"""Bounded candidate index for normalized exact/near checks against streamed references."""
from collections import Counter,defaultdict
import hashlib
import json
import math
from .evaluation import normalize


def shingles(text):
    words=normalize(text).split()
    return {tuple(words[i:i+5]) for i in range(len(words)-4)}


def audit_references(cases,references,threshold=.85):
    if not math.isfinite(threshold) or not 0<threshold<=1:raise ValueError('Invalid overlap threshold')
    normalized=[];grams=[];exact=defaultdict(list);index=defaultdict(list);seen=set();result=[]
    for position,case in enumerate(cases):
        ident=case.get('id')
        if not isinstance(ident,str) or not ident or ident in seen:raise ValueError('Unique case identities required')
        seen.add(ident);messages=case.get('messages',[])
        text=messages[0].get('content') if len(messages)==1 and isinstance(messages[0],dict) and messages[0].get('role')=='user' else None
        norm=normalize(text) if isinstance(text,str) else '';g=shingles(text) if norm else set()
        normalized.append(norm);grams.append(g)
        if norm:exact[norm].append(position)
        for shingle in g:index[shingle].append(position)
        result.append({'id':ident,'normalized_exact_matches':0,'near_matches':0,'reference_sources':set(),'evidence':[]})
    total=with_text=near_eligible=excluded=0;fingerprint=hashlib.sha256()
    def record(i,reference,kind,similarity):
        entry=result[i];entry[kind]+=1;entry['reference_sources'].add(reference['source'])
        if len(entry['evidence'])<3:entry['evidence'].append({'reference_id':reference['id'],'source':reference['source'],
            'match':'normalized_exact' if kind=='normalized_exact_matches' else 'near','jaccard':similarity})
    for reference in references:
        total+=1
        if not isinstance(reference.get('id'),str) or not isinstance(reference.get('source'),str):raise ValueError('Reference identity required')
        text=reference.get('text');norm=normalize(text) if isinstance(text,str) else ''
        include=reference.get('include',True)
        fingerprint.update(json.dumps({'id':reference['id'],'source':reference['source'],'include':include,
            'normalized_sha256':hashlib.sha256(norm.encode()).hexdigest()},sort_keys=True).encode()+b'\n')
        if not include:excluded+=1;continue
        if not norm:continue
        with_text+=1;exact_positions=set(exact.get(norm,()))
        for i in exact_positions:record(i,reference,'normalized_exact_matches',1.)
        g=shingles(text)
        if not g:continue
        near_eligible+=1;intersections=Counter();n=len(g)
        for shingle in g:
            for i in index.get(shingle,()):
                if i not in exact_positions and min(n,len(grams[i]))/max(n,len(grams[i]))>=threshold:intersections[i]+=1
        for i,intersection in intersections.items():
            similarity=intersection/(n+len(grams[i])-intersection)
            if similarity>=threshold:record(i,reference,'near_matches',similarity)
    for entry in result:entry['reference_sources']=sorted(entry['reference_sources'])
    return {'schema_version':1,'method':'NFKC casefold word normalization; unique five-word-shingle Jaccard',
        'near_threshold':threshold,'candidate_count':len(cases),'reference_rows':total,'reference_rows_excluded':excluded,
        'reference_rows_with_text':with_text,'reference_rows_near_eligible':near_eligible,
        'reference_stream_sha256':fingerprint.hexdigest(),'cases':result,
        'matched_candidates':sum(bool(r['normalized_exact_matches'] or r['near_matches']) for r in result),
        'training_exclusion_verified':False,'scope':'available references only; not original final training membership certification'}
