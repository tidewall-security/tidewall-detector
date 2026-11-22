"""Wire and accounting checks for bounded local serving qualification probes."""
from collections import Counter
import math
from .profiles import PROFILES
from .schemas import PROFILE
from .api import STATUS,RETRYABLE


def validate_reply(status,data,request_id,expected_statuses,*,error_codes=None,profile=PROFILE):
    if status not in expected_statuses or not isinstance(data,dict):raise ValueError('Unexpected qualification response')
    if data.get('schema_version')!=1 or type(data.get('schema_version')) is not int or (profile not in PROFILES or data.get('profile')!=profile):
        raise ValueError('Invalid response profile')
    if status==200:
        fields={'schema_version','profile','request_id','status','unsafe_score','score_kind','model','input_tokens','inference_ms'}
        if set(data)!=fields or data['request_id']!=request_id or data['status']!='classified' or data['model']!=PROFILES[profile]['identity']():raise ValueError('Invalid classified identity')
        score=data['unsafe_score'];duration=data['inference_ms']
        if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid score')
        if data['score_kind']!=PROFILES[profile]['score_kind'] or type(data['input_tokens']) is not int or not 1<=data['input_tokens']<=1024:
            raise ValueError('Invalid score semantics/coverage')
        if type(duration) not in (int,float) or not math.isfinite(duration) or duration<0:raise ValueError('Invalid inference duration')
    else:
        fields={'schema_version','profile','request_id','status','code','retryable'};code=data.get('code')
        if error_codes is not None and (not isinstance(code,str) or code not in error_codes):raise ValueError('Unexpected qualification failure code')
        if (set(data)!=fields or data.get('status')!='error' or not isinstance(code,str) or code not in STATUS
            or STATUS[code]!=status or data['request_id'] not in (None,request_id) or type(data['retryable']) is not bool
            or data['retryable']!=(code in RETRYABLE)):raise ValueError('Invalid typed failure')


def summarize_calls(rows):
    success=[r['duration_ms'] for r in rows if r['http_status']==200 and r['valid']]
    def percentile(p):
        if not success:return None
        values=sorted(success);index=(len(values)-1)*p;lo=math.floor(index);hi=math.ceil(index)
        return values[lo]+(values[hi]-values[lo])*(index-lo)
    return {'total':len(rows),'classified':len(success),'invalid':sum(not r['valid'] for r in rows),
        'http_counts':dict(Counter(str(r['http_status']) for r in rows)),
        'successful_latency_ms':{'p50':percentile(.5),'p95':percentile(.95)},
        'scope':'HTTP responses; error latency never counted as successful inference latency'}
