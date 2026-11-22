"""Pinned-artifact wrapper around the preserved existing scoring recipe."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import shutil

from . import recipe

MANIFEST_PATH=Path(__file__).with_name('model-manifest.json')
RESERVED_TOKENS=('<|endoftext|>','<|im_start|>','<|im_end|>','<|object_ref_start|>',
    '<|object_ref_end|>','<|box_start|>','<|box_end|>','<|quad_start|>','<|quad_end|>',
    '<|vision_start|>','<|vision_end|>','<|vision_pad|>','<|image_pad|>','<|video_pad|>')


class ArtifactError(RuntimeError): pass
class InputTooLarge(ValueError): pass


def identity():
    data=json.loads(MANIFEST_PATH.read_text())
    return {k:v for k,v in data.items() if k!='files'}


def verify_manifest(root):
    root=Path(root).resolve()
    manifest=json.loads(MANIFEST_PATH.read_text())
    if hashlib.sha256(recipe.SYSTEM_PROMPT.encode()).hexdigest()!=manifest['prompt_sha256']:
        raise ArtifactError('Classifier prompt differs from the pinned recipe')
    try:
        expected_files=set(manifest['files'])
        expected_dirs={str(Path(name).parent) for name in expected_files}
        found_files=set()
        for path in root.rglob('*'):
            name=str(path.relative_to(root))
            if path.is_symlink(): raise ArtifactError('Symlinked model artifact rejected')
            if path.is_file(): found_files.add(name)
            elif name not in expected_dirs: raise ArtifactError('Unexpected model directory')
        if found_files!=expected_files: raise ArtifactError('Missing or unexpected model artifact')
        for name,expected in manifest['files'].items():
            path=root/name
            if path.resolve().parent not in (root/'base',root/'adapter') or not path.is_file():
                raise ArtifactError('Missing or unsafe model artifact')
            if path.stat().st_size!=expected['bytes']: raise ArtifactError('Model artifact size mismatch')
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(4*1024*1024),b''): digest.update(chunk)
            if digest.hexdigest()!=expected['sha256']: raise ArtifactError('Model artifact hash mismatch')
    except OSError:
        raise ArtifactError('Model artifacts unavailable') from None
    return manifest


def stage_models(source,destination):
    """Explicitly stage only pinned regular files; preserve downloader metadata."""
    source=Path(source).resolve(); destination=Path(destination)
    manifest=json.loads(MANIFEST_PATH.read_text())
    destination.mkdir(parents=True,exist_ok=False)
    for name in manifest['files']:
        original=source/name
        if (not original.is_file() or original.is_symlink()
                or original.resolve().parent!=source/Path(name).parent):
            raise ArtifactError('Missing or unsafe source artifact')
        target=destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,target,follow_symlinks=False)
        target.chmod(0o444)
    verify_manifest(destination)
    return destination


@dataclass(frozen=True)
class ScoreResult:
    unsafe_score: float
    input_tokens: int
    inference_ms: float


class LastTokenModel:
    """Keep full attention context; omit vocabulary projections we never read."""
    def __init__(self,model): self.model=model
    def __call__(self,**inputs): return self.model(**inputs,logits_to_keep=1)


class Classifier:
    ready=False
    def __init__(self,root,device='cpu'):
        self.device=device
        verify_manifest(root)
        # Imports deliberately occur after verification. Local paths and offline
        # runtime flags forbid automatic retrieval of absent model artifacts.
        self.tokenizer,self.model=recipe.load_model(str(Path(root)/'base'),Path(root)/'adapter',device)
        ids=recipe.resolve_class_token_ids(self.tokenizer)
        if list(ids)!=json.loads(MANIFEST_PATH.read_text())['class_token_ids']:
            raise ArtifactError('Class-token identities differ from the pinned tokenizer')
        self.unsafe,self.safe,_,_=ids
        self.score('Hello.')
        self.ready=True

    def score(self,text):
        prompt=self.tokenizer.apply_chat_template(
            [{'role':'system','content':recipe.SYSTEM_PROMPT},{'role':'user','content':text}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        tokens=len(self.tokenizer(prompt)['input_ids'])
        if tokens>1024: raise InputTooLarge('Templated input exceeds 1024 tokens')
        import torch
        if self.device=='mps': torch.mps.synchronize()
        started=time.perf_counter()
        value=recipe.score_prompt(self.tokenizer,LastTokenModel(self.model),text,self.unsafe,self.safe,False,self.device)
        if self.device=='mps': torch.mps.synchronize()
        return ScoreResult(value,tokens,(time.perf_counter()-started)*1000)

    def close(self):
        self.ready=False
        self.model=None
        self.tokenizer=None
