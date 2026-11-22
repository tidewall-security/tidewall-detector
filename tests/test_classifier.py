import ast
import hashlib
import json
from pathlib import Path
import shutil

import pytest
from tidewall_lora.classifier import verify_manifest, identity, ArtifactError

ROOT=Path(__file__).resolve().parents[1]


def test_scoring_recipe_is_the_existing_source():
    text=(ROOT/'tidewall_lora/recipe.py').read_text()
    provenance=json.loads((ROOT/'tidewall_lora/recipe-provenance.json').read_text())
    actual={n.name:hashlib.sha256(ast.get_source_segment(text,n).encode()).hexdigest()
            for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in provenance['functions']}
    assert actual==provenance['functions']


def test_identity_pins_existing_assets():
    value=identity()
    assert value['base_revision']=='c1899de289a04d12100db370d81485cdf75e47ca'
    assert value['adapter_revision']=='5500c3bef53729ed4661b5cfae1b01adca96ca4c'
    assert value['tokenizer_revision']==value['base_revision']
    assert value['prompt_sha256']=='3a0e693064509eb24773ba53af4831ecb4c6ca8cf741a28d2d10080384df5041'


def test_missing_artifacts_refuse_readiness(tmp_path):
    with pytest.raises(ArtifactError): verify_manifest(tmp_path)


def test_modified_metadata_is_rejected(tmp_path):
    manifest=json.loads((ROOT/'tidewall_lora/model-manifest.json').read_text())
    for name in manifest['files']:
        p=tmp_path/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'changed')
    with pytest.raises(ArtifactError): verify_manifest(tmp_path)


def test_extra_loader_file_is_rejected(tmp_path,monkeypatch):
    from tidewall_lora import classifier
    manifest=json.loads(classifier.MANIFEST_PATH.read_text())
    manifest['files']={}
    for name in ('base/config.json','adapter/adapter_config.json'):
        path=tmp_path/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'{}')
        manifest['files'][name]={'bytes':2,'sha256':hashlib.sha256(b'{}').hexdigest()}
    trusted=tmp_path.parent/(tmp_path.name+'-manifest.json'); trusted.write_text(json.dumps(manifest))
    monkeypatch.setattr(classifier,'MANIFEST_PATH',trusted)
    assert classifier.verify_manifest(tmp_path)
    (tmp_path/'base/chat_template.jinja').write_text('Unexpected alternate template')
    with pytest.raises(ArtifactError): classifier.verify_manifest(tmp_path)


def test_staging_copies_only_pinned_files(tmp_path,monkeypatch):
    from tidewall_lora import classifier
    source=tmp_path/'download'; source.mkdir()
    manifest=json.loads(classifier.MANIFEST_PATH.read_text()); manifest['files']={}
    for name in ('base/config.json','adapter/adapter_config.json'):
        path=source/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'{}')
        manifest['files'][name]={'bytes':2,'sha256':hashlib.sha256(b'{}').hexdigest()}
    cache=source/'base/.cache/huggingface'; cache.mkdir(parents=True)
    (cache/'metadata').write_text('Downloader bookkeeping')
    (source/'base/chat_template.jinja').write_text('Unlisted template must never be copied')
    trusted=tmp_path/'manifest.json'; trusted.write_text(json.dumps(manifest))
    monkeypatch.setattr(classifier,'MANIFEST_PATH',trusted)
    destination=tmp_path/'verified'
    classifier.stage_models(source,destination)
    assert classifier.verify_manifest(destination)
    assert not (destination/'base/chat_template.jinja').exists() and not (destination/'base/.cache').exists()
    assert (source/'base/chat_template.jinja').exists()


def test_last_token_projection_keeps_full_input_context():
    from tidewall_lora.classifier import LastTokenModel
    seen=[]
    class Model:
        def __call__(self,**kwargs): seen.append(kwargs); return 'original output'
    assert LastTokenModel(Model())(input_ids='full input',attention_mask='full mask')=='original output'
    assert seen==[{'input_ids':'full input','attention_mask':'full mask','logits_to_keep':1}]
