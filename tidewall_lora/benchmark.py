"""Freeze verified imported source and overlap receipts for fixed-test diagnostics."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

from .evaluation import summarize
from .calibration import probability_metrics
from .comparison import VARIANTS_BY_PROFILE
from .schemas import PROFILE


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_benchmark(import_dir,audit_path,output,profile=PROFILE):
    if profile not in VARIANTS_BY_PROFILE: raise ValueError('Unknown benchmark profile')
    from .external_benchmark import WILDGUARD,verify_file,prepare_rows
    import pyarrow.parquet as pq
    import_dir=Path(import_dir);audit_path=Path(audit_path);output=Path(output)
    if output.exists():raise FileExistsError(output)
    imported=json.loads((import_dir/'report.json').read_text());casespath=import_dir/'cases.jsonl'
    if imported.get('status')!='imported' or imported.get('source')!=WILDGUARD or sha(casespath)!=imported['cases_sha256']:
        raise ValueError('Invalid imported source receipt')
    identity=verify_file(import_dir/'source.parquet',WILDGUARD)
    if identity!=imported['file_identity']:raise ValueError('Imported source identity changed')
    raw=pq.read_table(import_dir/'source.parquet').to_pylist()
    cases=[json.loads(line) for line in casespath.read_text().splitlines()]
    if len(raw)!=WILDGUARD['rows'] or cases!=prepare_rows(raw,WILDGUARD):raise ValueError('Cases differ from source ground truth')
    audit=json.loads(audit_path.read_text())
    if (audit.get('status')!='complete' or audit.get('candidate_cases_sha256')!=sha(casespath)
        or audit.get('source_identity')!=identity or audit.get('training_exclusion_verified') is not False):
        raise ValueError('Invalid overlap audit receipt')
    entries=audit['analysis']['cases'];ids=[r['id'] for r in entries]
    if len(set(ids))!=len(ids) or set(ids)!={c['id'] for c in cases} or audit['analysis']['candidate_count']!=len(cases):
        raise ValueError('Overlap audit does not cover every case exactly once')
    by_id={r['id']:r for r in entries}
    for case,source_row in zip(cases,raw,strict=True):
        if case['partition']!='test' or case['calibration_allowed'] is not False:raise ValueError('Official source must remain test-only')
        entry=by_id[case['id']]
        if any(type(entry[k]) is not int or entry[k]<0 for k in ('normalized_exact_matches','near_matches')):
            raise ValueError('Invalid overlap counts')
        case.update(available_reference_match=bool(entry['normalized_exact_matches'] or entry['near_matches']),
            available_reference_sources=entry['reference_sources'],source_adversarial=source_row.get('adversarial'),
            source_subcategory=source_row.get('subcategory'))
    # Recheck source before storing the frozen snapshot.
    if verify_file(import_dir/'source.parquet',WILDGUARD)!=identity:raise ValueError('Source changed during freezing')
    protocol={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),'threshold':.5,'timeout_ms':1000,
        'profile':profile,'models':list(VARIANTS_BY_PROFILE[profile]),'calibration_fitting':False,
        'model_order':' then '.join(VARIANTS_BY_PROFILE[profile])+'; serial; no speed superiority claim',
        'purpose':'source-annotated prompt-harm benchmark; source-declared test, partial reference screening only',
        'source':dict(WILDGUARD),'source_identity':identity,'import_cases_sha256':sha(casespath),'audit_sha256':sha(audit_path),
        'training_exclusion_verified':False,'production_approved':False,
        'authorization_scope':'user-authorized private local evaluation; source-account access already approved; no raw redistribution',
        'privacy_scope':'protected local source processing, no identification or prompt publication; no independent privacy certification'}
    plan={'protocol':protocol,'cases':cases}
    output.mkdir(parents=True,mode=0o700,exist_ok=False)
    shutil.copyfile(import_dir/'source.parquet',output/'source.parquet');(output/'source.parquet').chmod(0o400)
    if verify_file(output/'source.parquet',WILDGUARD)!=identity:raise ValueError('Frozen source copy differs')
    shutil.copyfile(audit_path,output/'overlap-audit.json')
    (output/'frozen-cases.json').write_text(json.dumps(plan,indent=2)+'\n')
    return plan


def metric_bundle(rows):
    def one(selected):
        value=summarize(selected);c=value['confusion'];denom=2*c['tp']+c['fp']+c['fn']
        value.update(f1=2*c['tp']/denom if denom else None,raw_probability_metrics=probability_metrics(selected),
            metric_scope='external source prompt-harm labels; conditional classification metrics; raw scores, no calibration fitted')
        return value
    return {'whole_source':one(rows),'available_reference_screened':one([r for r in rows if not r['available_reference_match']]),
        'screening_scope':'known reference matches removed only from secondary metrics; all source rows remain in primary accounting; unknown original training membership persists'}
