"""Frozen synthetic development cases and correctly identified offline model variants."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

from .evaluation import prepare_cases
from .calibration import split_rows
from .schemas import PROFILE,ENSEMBLE_PROFILE

VARIANTS_BY_PROFILE={PROFILE:('lora','base'),ENSEMBLE_PROFILE:('ensemble-q8',)}


VARIATIONS=('Cedar-17','Harbor-29','Maple-43','Willow-61')


def freeze_cases(scenarios,destination):
    scenarios=Path(scenarios);destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    definitions=json.loads(scenarios.read_text());rows=[];metadata=[]
    for family in definitions:
        for name in VARIATIONS:
            for label in ('safe','unsafe'):
                rows.append({'label':['benign' if label=='safe' else 'injection'],'source':family['id'],
                    'messages':[{'role':'user','content':family[label].format(name=name)}]})
                metadata.append({'category':family['category'],'label_rationale':
                    'Authorized defensive/quoted-data task; no override or secret disclosure requested' if label=='safe' else
                    'Requests policy override, unauthorized action, secret disclosure or falsification'})
    with tempfile.TemporaryDirectory(prefix='tw-cases-') as tmp:
        corpus=Path(tmp)/'cases.jsonl';corpus.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        plan=prepare_cases(corpus)
    for case,meta in zip(plan['cases'],metadata,strict=True):
        case.update(meta,partition=case['partition'].removeprefix('diagnostic_'))
    split_rows(plan['cases'])
    plan.update(dataset_kind='new_ai_authored_synthetic_development',created_at=datetime.now(timezone.utc).isoformat(),
        definitions_sha256=hashlib.sha256(scenarios.read_bytes()).hexdigest(),previously_evaluated=False,
        human_labels_reviewed=False,training_overlap_verified=False,calibrated_probability_available=False,
        label_scope='proposed binary safety/injection policy; not independently adjudicated',
        effective_independent_units='scenario families; variations are not independent samples')
    with destination.open('x') as stream:stream.write(json.dumps(plan,indent=2)+'\n')
    return plan


def score_variant(classifier,text,variant):
    if variant in {'lora','ensemble-q8'}:return classifier.score(text)
    if variant=='base':
        with classifier.model.disable_adapter():return classifier.score(text)
    raise ValueError('Unknown model variant')


def overlap_audit(cases,reference):
    """Exact/near checks against available standalone historical inputs only."""
    from .evaluation import normalize
    reference=Path(reference);raw=reference.read_bytes();known=[]
    def shingles(text):
        words=normalize(text).split()
        return {tuple(words[i:i+5]) for i in range(len(words)-4)}
    for line in raw.decode().splitlines():
        if not line.strip():continue
        messages=json.loads(line).get('messages',[])
        if len(messages)==1 and messages[0].get('role')=='user':
            text=messages[0]['content'];known.append((normalize(text),shingles(text)))
    matches=[]
    for case in cases:
        a=shingles(case['text']);norm=normalize(case['text'])
        for other,b in known:
            if norm==other or (a and b and len(a&b)/len(a|b)>=.85):
                matches.append(case['id']);break
    return {'reference_sha256':hashlib.sha256(raw).hexdigest(),'reference_standalone_inputs':len(known),
        'matched_case_ids':matches,'comparison':'NFKC word normalization or >=0.85 five-word shingle Jaccard',
        'training_nonoverlap_certified':False}


def assess(classifier,text,variant,profile=PROFILE):
    """Keep input rejection, model failure and completed-over-deadline distinct."""
    import math
    import time
    from uuid import uuid4
    from pydantic import ValidationError
    from .classifier import InputTooLarge,RESERVED_TOKENS
    from .schemas import ClassifyRequest
    if profile not in VARIANTS_BY_PROFILE or variant not in VARIANTS_BY_PROFILE[profile]:
        raise ValueError('Unknown model profile/variant pair')
    started=time.perf_counter()
    def response(status,reason,**values):
        return {'status':status,'reason':reason,'latency_ms':(time.perf_counter()-started)*1000,**values}
    try:
        ClassifyRequest(schema_version=1,profile=profile,request_id=str(uuid4()),
            event_kind='prompt',messages=[{'role':'user','content':text}],timeout_ms=1000)
        if len(text.encode())>8000:return response('unsupported','input_too_large')
        if any(token in text for token in RESERVED_TOKENS):return response('unsupported','reserved_framing')
    except (ValidationError,UnicodeError):return response('unsupported','invalid_input')
    try:
        result=score_variant(classifier,text,variant)
        if (type(result.unsafe_score) not in (float,int) or not math.isfinite(result.unsafe_score)
            or not 0<=result.unsafe_score<=1 or type(result.input_tokens) is not int
            or not 1<=result.input_tokens<=1024 or not math.isfinite(result.inference_ms) or result.inference_ms<0):
            return response('error','invalid_model_output')
    except InputTooLarge:return response('unsupported','input_too_large')
    except Exception:return response('error','model_error')
    if (time.perf_counter()-started)*1000>1000:return response('error','deadline_exceeded')
    return response('classified','',unsafe_score=result.unsafe_score,input_tokens=result.input_tokens,inference_ms=result.inference_ms)


def attempt_counts(rows):
    from collections import Counter
    counts=Counter(r['status'] for r in rows)
    if set(counts)-{'classified','error','unsupported'}:raise ValueError('Unfinished attempt')
    return {'total':len(rows),**{s:counts[s] for s in ('classified','error','unsupported')},
        'reasons':dict(Counter(r['reason'] for r in rows if r['status']!='classified'))}


def development_summary(rows):
    from .evaluation import summarize
    result=summarize(rows)
    result['metric_scope']='AI-authored synthetic development; conditional on successful scores; experimental calibration fitted separately, not promoted'
    return result
