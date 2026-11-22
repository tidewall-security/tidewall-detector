"""Offline source-annotated WildGuardTest import. No downloader, scorer or gate bypass."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import shutil

from .classifier import RESERVED_TOKENS
from .evaluation import normalize

WILDGUARD={'repo':'allenai/wildguardmix','revision':'d29c47f41c8b51348b5c8e8c81c039b3132b66d1',
    'path':'test/wildguard_test.parquet','bytes':2258878,
    'git_lfs_pointer_oid':'f347cada395638467cbc5ce0f29222752c672e40','rows':1725}


def verify_file(path,spec):
    path=Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size!=spec['bytes']:
        raise ValueError('File does not match pinned benchmark size/type')
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):sha.update(chunk)
    pointer=(f'version https://git-lfs.github.com/spec/v1\noid sha256:{sha.hexdigest()}\nsize {spec["bytes"]}\n').encode()
    oid=hashlib.sha1(f'blob {len(pointer)}\0'.encode()+pointer).hexdigest()
    if oid!=spec['git_lfs_pointer_oid']:raise ValueError('File does not match pinned benchmark pointer identity')
    return {'sha256':sha.hexdigest(),'bytes':path.stat().st_size,'git_lfs_pointer_oid':oid}


def prepare_rows(source_rows,spec):
    output=[];groups=defaultdict(list)
    for index,row in enumerate(source_rows):
        if not isinstance(row,dict):raise ValueError('Invalid source row')
        prompt=row.get('prompt');label=row.get('prompt_harm_label')
        expected={'harmful':'unsafe','unharmful':'safe'}.get(label) if isinstance(label,str) else None
        reason='';text=prompt if isinstance(prompt,str) else None
        if expected is None:reason='missing_or_unknown_prompt_harm_label'
        elif text is None or not text.strip():reason='missing_or_empty_prompt'
        else:
            try:
                if len(text.encode())>8000:reason='input_too_large'
            except UnicodeError:reason='invalid_unicode'
            if not reason and any(t in text for t in RESERVED_TOKENS):reason='reserved_framing'
        context_hash=hashlib.sha256(json.dumps(prompt,ensure_ascii=True).encode()).hexdigest()
        ident=hashlib.sha256(f'{spec["repo"]}:{spec["revision"]}:{spec["path"]}:{index}:{context_hash}'.encode()).hexdigest()
        norm=normalize(text) if text is not None else ''
        group=hashlib.sha256(norm.encode()).hexdigest()
        record={'id':ident,'source':spec['repo'],'source_revision':spec['revision'],'source_row':index,
            'source_prompt_harm_label':label,'source_content_sha256':context_hash,
            'messages':[{'role':'user','content':text}] if text is not None else [],'expected':expected,
            'group_id':group,'partition':'test','calibration_allowed':False,
            'status':'unsupported' if reason else 'pending','reason':reason,
            'label_provenance':'publisher prompt_harm_label; source documents human annotations',
            'tokenizer_check':'pending','training_exclusion_verified':False}
        output.append(record)
        if text is not None:groups[norm].append(record)
    for records in groups.values():
        if len({r['expected'] for r in records if r['expected'] is not None})>1:
            for r in records:
                r.update(status='unsupported',reason='conflicting_labels')
    return output


def import_file(source,output):
    source=Path(source);output=Path(output)
    output.mkdir(parents=True,mode=0o700,exist_ok=False)
    report={'schema_version':1,'status':'in_progress','source':dict(WILDGUARD),
        'source_task':'prompt harmfulness; response labels are not model inputs or fallback ground truth',
        'annotation_provenance_url':'https://huggingface.co/datasets/allenai/wildguardmix',
        'original_file_preserved':False,'partition':'test','calibration_allowed':False,
        'training_exclusion_verified':False,'scoring_ready':False,'production_approved':False,
        'remaining_reviews':['source-use rights and privacy','cross-source train/development overlap','tokenizer limit'],
        'model_inference_performed':False}
    def save():(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        if not source.is_file() or source.is_symlink() or source.stat().st_size!=WILDGUARD['bytes']:
            raise ValueError('Approved source has wrong size/type')
        snapshot=output/'source.parquet'
        shutil.copyfile(source,snapshot);snapshot.chmod(0o400)
        report['file_identity']=verify_file(snapshot,WILDGUARD);save()
        try:
            import pyarrow.parquet as pq
        except ImportError:raise RuntimeError('Install the optional inference[evaluation] dependencies') from None
        parquet=pq.ParquetFile(snapshot)
        required={'prompt','prompt_harm_label'}
        if not required.issubset(parquet.schema_arrow.names):raise ValueError('Source columns do not match prompt-harm task')
        if parquet.metadata.num_rows!=WILDGUARD['rows']:raise ValueError('Source row count differs from pinned benchmark')
        rows=prepare_rows((r for batch in parquet.iter_batches(batch_size=128) for r in batch.to_pylist()),WILDGUARD)
        if len(rows)!=WILDGUARD['rows']:raise ValueError('Imported row count differs from source')
        # Ensure the immutable snapshot still matches the originally verified bytes.
        if verify_file(snapshot,WILDGUARD)!=report['file_identity']:raise ValueError('Source changed during import')
        destination=output/'cases.jsonl'
        with destination.open('x') as stream:
            for row in rows:stream.write(json.dumps(row)+'\n')
        counts=Counter(r['status'] for r in rows)
        report.update(status='imported',original_file_preserved=True,
            counts={'total':len(rows),'pending':counts['pending'],'unsupported':counts['unsupported']},
            reasons=dict(Counter(r['reason'] for r in rows if r['reason'])),
            cases_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__);raise
    finally:save()
    return report
