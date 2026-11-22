"""Offline intake and blinded label checks; recorded attestations are not certification."""
from collections import Counter,defaultdict
from datetime import datetime
import hashlib
import json
import re

SOURCES={'consented_activity','public_dataset','human_authored','synthetic','inspected_development'}
LABELS={'safe','unsafe','uncertain','out_of_scope'}
STATUSES=('eligible','pending_provenance','pending_review','disputed','unsupported','development_only')
CASE_FIELDS={'schema_version','case_id','family_id','origin_id','source_kind','source_reference',
             'source_revision','collected_at','evaluation_rights','privacy_review','messages'}
BLIND_FORBIDDEN={'unsafe_score','predicted_label','confidence','score','model_output','label','labels',
    'annotation','annotations','prediction','predictions','assessment','label_rationale','reviewer_id','expected','ground_truth'}
REVIEW_FIELDS={'case_id','content_sha256','reviewer_id','label','rationale','independent'}


def context_hash(messages):
    return hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def identifier(value):
    return isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}',value) is not None


def reject_prediction_metadata(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key.casefold() in BLIND_FORBIDDEN:raise ValueError('Prediction or annotation metadata is not blinded')
            reject_prediction_metadata(item)
    elif isinstance(value,list):
        for item in value:reject_prediction_metadata(item)


def validate_case(case):
    if not isinstance(case,dict) or set(case)!=CASE_FIELDS:raise ValueError('Unknown or missing case fields')
    if type(case['schema_version']) is not int or case['schema_version']!=1:raise ValueError('Invalid case version')
    if any(not identifier(case[k]) for k in ('case_id','family_id','origin_id')):raise ValueError('Invalid case/group identity')
    if not isinstance(case['source_kind'],str) or case['source_kind'] not in SOURCES:raise ValueError('Invalid source kind')
    for field in ('source_reference','source_revision'):
        if not isinstance(case[field],str) or len(case[field])>1024:raise ValueError('Invalid source reference')
    revision=case['source_revision']
    if revision and not re.fullmatch(r'(sha256:[a-f0-9]{64}|git:[a-f0-9]{40})',revision):raise ValueError('Immutable source revision required')
    try:
        when=datetime.fromisoformat(case['collected_at'])
        if when.tzinfo is None:raise ValueError('Timezone required')
    except (TypeError,ValueError):raise ValueError('Invalid collection timestamp') from None
    for field in ('evaluation_rights','privacy_review'):
        value=case[field]
        if not isinstance(value,dict) or set(value)!={'confirmed','evidence_reference'}:raise ValueError('Invalid recorded attestation')
        if type(value['confirmed']) is not bool or not isinstance(value['evidence_reference'],str):raise ValueError('Invalid attestation values')
    messages=case['messages']
    if not isinstance(messages,list) or not messages:raise ValueError('Full nonempty message context required')
    for message in messages:
        if not isinstance(message,dict) or not isinstance(message.get('role'),str) or 'content' not in message:raise ValueError('Invalid context message')
        reject_prediction_metadata({k:v for k,v in message.items() if k!='content'})
    context_hash(messages)  # Also rejects non-JSON/nonfinite context, without flattening it.


def provenance_ready(case):
    return bool(case['source_reference'] and case['source_revision'] and all(
        case[k]['confirmed'] and case[k]['evidence_reference'].strip() for k in ('evaluation_rights','privacy_review')))


def blind_cases(cases):
    output=[];seen=set()
    for case in cases:
        validate_case(case)
        if case['case_id'] in seen:raise ValueError('Duplicate case identity')
        seen.add(case['case_id'])
        if provenance_ready(case):
            output.append({'case_id':case['case_id'],'content_sha256':context_hash(case['messages']),'messages':case['messages']})
    return output


def audit(cases,reviews,adjudications=()):
    by_id={};annotations=defaultdict(list);decisions={}
    for case in cases:
        validate_case(case)
        if case['case_id'] in by_id:raise ValueError('Duplicate case identity')
        by_id[case['case_id']]=case
    def check_review(review):
        if not isinstance(review,dict) or set(review)!=REVIEW_FIELDS:raise ValueError('Unknown or missing annotation fields')
        if not identifier(review['case_id']) or review['case_id'] not in by_id:raise ValueError('Annotation references unknown case')
        if not identifier(review['reviewer_id']) or not isinstance(review['label'],str) or review['label'] not in LABELS:raise ValueError('Invalid reviewer or label')
        if not isinstance(review['rationale'],str) or not review['rationale'].strip() or type(review['independent']) is not bool:raise ValueError('Annotation rationale/independence required')
        if review['content_sha256']!=context_hash(by_id[review['case_id']]['messages']):raise ValueError('Reviewed content has changed')
    for review in reviews:
        check_review(review);group=annotations[review['case_id']]
        if any(r['reviewer_id']==review['reviewer_id'] for r in group) or len(group)>=2:raise ValueError('Exactly two distinct initial reviewers required')
        group.append(review)
    for review in adjudications:
        check_review(review);ident=review['case_id']
        if ident in decisions or any(r['reviewer_id']==review['reviewer_id'] for r in annotations[ident]):raise ValueError('Distinct adjudicator required')
        decisions[ident]=review
    result=[]
    from .classifier import RESERVED_TOKENS
    for ident,case in by_id.items():
        label=None;reason='';status='pending_review'
        group=annotations[ident];adjudication=decisions.get(ident)
        if not provenance_ready(case):status='pending_provenance';reason='recorded_rights_privacy_or_source_incomplete'
        elif case['source_kind'] in {'synthetic','inspected_development'}:status='development_only';reason='not_fresh_independent_activity'
        else:
            messages=case['messages']
            supported=(len(messages)==1 and set(messages[0])=={'role','content'} and messages[0]['role']=='user'
                and isinstance(messages[0]['content'],str) and bool(messages[0]['content'].strip()))
            if supported:
                text=messages[0]['content']
                supported=len(text.encode())<=8000 and not any(token in text for token in RESERVED_TOKENS)
            if not supported:status='unsupported';reason='current_single_user_profile_boundary'
            elif len(group)!=2 or not all(r['independent'] for r in group):reason='two_independent_reviews_required'
            else:
                if group[0]['label']==group[1]['label'] and group[0]['label'] in {'safe','unsafe','out_of_scope'}:
                    label=group[0]['label']
                elif adjudication and adjudication['independent'] and adjudication['label'] in {'safe','unsafe','out_of_scope'}:
                    label=adjudication['label']
                else:status='disputed';reason='independent_adjudication_required'
                if label=='out_of_scope':status='unsupported';reason='human_scope_exclusion'
                elif label in {'safe','unsafe'}:status='eligible';reason='pending_group_freeze_and_tokenizer_check'
        result.append({'case_id':ident,'content_sha256':context_hash(case['messages']),'status':status,'reason':reason,'label':label})
    counts=Counter(r['status'] for r in result)
    return {'schema_version':1,'total':len(cases),'counts':{s:counts[s] for s in STATUSES},'cases':result,
        'ready_for_group_freeze':bool(counts['eligible']) and not any(counts[s] for s in ('pending_provenance','pending_review','disputed')),
        'held_out_certified':False,'production_approved':False,'training_overlap_verified':False,
        'scope':'structural audit of recorded attestations; no verification of actual consent, reviewer independence, labels or training membership'}
