"""Offline, experimental score calibration. Never installed into the serving API."""
import hashlib
import json
import math

EPSILON=1e-6


def clipped(score):
    if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1:
        raise ValueError('Invalid raw score')
    return min(1-EPSILON,max(EPSILON,score))


def logit(score):
    p=clipped(score)
    return math.log(p)-math.log1p(-p)


def sigmoid(value):
    if value>=0:return 1/(1+math.exp(-value))
    e=math.exp(value);return e/(1+e)


def split_rows(rows):
    ids=set();groups={};parts={'calibration':[],'test':[]}
    for row in rows:
        ident=row.get('id');group=row.get('group_id');part=row.get('partition')
        if not ident or not group or part not in parts or ident in ids:raise ValueError('Invalid or repeated split identity')
        if group in groups and groups[group]!=part:raise ValueError('Group leaks across partitions')
        ids.add(ident);groups[group]=part;parts[part].append(row)
    if not all(parts.values()):raise ValueError('Both partitions required')
    return parts['calibration'],parts['test']


def classified(rows):
    selected=[]
    for row in rows:
        if row.get('status') not in {'classified','error','unsupported'}:raise ValueError('Unfinished or unknown status')
        if row['status']=='classified':
            clipped(row['unsafe_score'])
            if row.get('expected') not in {'safe','unsafe'}:raise ValueError('Invalid label')
            selected.append(row)
    return selected


def probability_metrics(rows):
    data=classified(rows);bins=[[] for _ in range(10)];brier=0.;nll=0.
    for row in data:
        raw=row['unsafe_score'];p=clipped(raw);y=int(row['expected']=='unsafe')
        brier+=(raw-y)**2;nll-=y*math.log(p)+(1-y)*math.log1p(-p)
        bins[min(9,int(raw*10))].append((raw,y))
    n=len(data)
    return {'total':len(rows),'classified':n,'coverage':n/len(rows) if rows else None,
        'brier':brier/n if n else None,'nll':nll/n if n else None,
        'ece_10_equal_width':sum(abs(sum(p for p,y in b)-sum(y for p,y in b)) for b in bins)/n if n else None,
        'clipped_scores':sum(r['unsafe_score']!=clipped(r['unsafe_score']) for r in data),
        'epsilon':EPSILON,'scope':'conditional on classified examples; source and calibration qualification supplied by caller'}


def fit_temperature(rows):
    if any(r.get('partition')!='calibration' for r in rows):raise ValueError('Fitting requires calibration partition only')
    if len({r.get('id') for r in rows})!=len(rows) or any(not r.get('id') or not r.get('group_id') for r in rows):
        raise ValueError('Calibration identities required and unique')
    data=classified(rows)
    if {r['expected'] for r in data}!={'safe','unsafe'}:raise ValueError('Both labels required for fitting')
    values=[(logit(r['unsafe_score']),int(r['expected']=='unsafe')) for r in data]
    def loss(log_temperature):
        t=math.exp(log_temperature);total=0.
        for margin,y in values:
            z=margin/t
            total+=max(z,0)-y*z+math.log1p(math.exp(-abs(z)))
        return total/len(values)
    # Fixed bounds and iteration count, chosen before seeing the test set.
    lo,hi=-4.,4.;ratio=(math.sqrt(5)-1)/2
    for _ in range(100):
        a=hi-ratio*(hi-lo);b=lo+ratio*(hi-lo)
        if loss(a)<loss(b):hi=b
        else:lo=a
    chosen=(lo+hi)/2;temperature=math.exp(chosen)
    fingerprint=[{k:r[k] for k in ('id','group_id','expected','unsafe_score')} for r in data]
    return {'schema_version':1,'method':'binary_temperature_scaling','temperature':temperature,'epsilon':EPSILON,
        'log_temperature_bounds':[-4.,4.],'bound_reached':min(abs(chosen+4),abs(chosen-4))<1e-5,
        'fit_rows':len(data),'fit_groups':len({r['group_id'] for r in data}),
        'fit_sha256':hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest(),
        'clipped_calibration_scores':sum(r['unsafe_score']!=clipped(r['unsafe_score']) for r in data),
        'deployment_approved':False,'training_overlap_verified':False,'human_labels_reviewed':False}


def apply_temperature(score,artifact):
    t=artifact.get('temperature')
    if (artifact.get('schema_version')!=1 or artifact.get('method')!='binary_temperature_scaling'
        or artifact.get('epsilon')!=EPSILON or type(t) not in (int,float) or not math.isfinite(t) or t<=0):
        raise ValueError('Invalid calibration artifact')
    return sigmoid(logit(score)/t)
