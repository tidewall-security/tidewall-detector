"""Local Q8 implementation of the existing two-specialist class-token recipe.

Recipe: prompt-injection-ensemble, commit b9bc67ae6109a3f7bd1c81f73e218a69acf60f36.
Models run sequentially in the API's single physical worker. Scores are not
calibrated probabilities or policy decisions.
"""
import hashlib
import json
import math
from pathlib import Path
import time

from .classifier import ArtifactError, InputTooLarge, ScoreResult
from .recipe import SYSTEM_PROMPT

MANIFEST_PATH=Path(__file__).with_name('ensemble-manifest.json')


def identity():
    return {k:v for k,v in json.loads(MANIFEST_PATH.read_text()).items() if k!='files'}


def verify_manifest(root):
    root=Path(root)
    manifest=json.loads(MANIFEST_PATH.read_text())
    if hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()!=manifest['prompt_sha256']:
        raise ArtifactError('Ensemble prompt differs from the pinned recipe')
    try:
        if root.is_symlink() or not root.is_dir(): raise ArtifactError('Unsafe model root')
        expected=set(manifest['files'])
        dirs={str(parent) for name in expected for parent in Path(name).parents if str(parent)!='.'}
        found=set()
        for path in root.rglob('*'):
            name=str(path.relative_to(root))
            if path.is_symlink(): raise ArtifactError('Symlinked model artifact rejected')
            if path.is_file(): found.add(name)
            elif not path.is_dir() or name not in dirs: raise ArtifactError('Unexpected model directory')
        if found!=expected: raise ArtifactError('Missing or unexpected model artifact')
        for name,pin in manifest['files'].items():
            path=root/name
            if Path(name).is_absolute() or '..' in Path(name).parts: raise ArtifactError('Unsafe artifact path')
            if path.stat().st_size!=pin['bytes']: raise ArtifactError('Model artifact size mismatch')
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(4*1024*1024),b''): digest.update(chunk)
            if digest.hexdigest()!=pin['sha256']: raise ArtifactError('Model artifact hash mismatch')
    except OSError:
        raise ArtifactError('Model artifacts unavailable') from None
    return manifest


class EnsembleClassifier:
    ready=False

    def __init__(self,root,device='cpu'):
        self.models=[]; self.tokenizers=[]
        # Resolve once: rotating a parent deployment symlink must not redirect
        # either loader after the release directory has been verified.
        root=Path(root)
        if root.is_symlink(): raise ArtifactError('Symlinked model root rejected')
        try: root=root.resolve(strict=True)
        except OSError: raise ArtifactError('Model artifacts unavailable') from None
        verify_manifest(root)
        import llama_cpp
        import transformers
        import numpy
        from transformers import AutoTokenizer
        if transformers.__version__!='4.51.3' or numpy.__version__!='2.4.4':
            raise ArtifactError('Unqualified tokenizer or numeric runtime version')
        if llama_cpp.__version__!='0.3.37': raise ArtifactError('Unqualified GGUF runtime version')
        if device not in {'cpu','metal'}: raise ArtifactError('Unsupported ensemble device')
        if device=='metal':
            import sys
            if sys.platform!='darwin' or not llama_cpp.llama_supports_gpu_offload():
                raise ArtifactError('Metal offload unavailable')
        try:
            for name in ('broad','multishot'):
                self.tokenizers.append(AutoTokenizer.from_pretrained(
                    str(Path(root)/name/'tokenizer'),local_files_only=True,trust_remote_code=False))
                self.models.append(llama_cpp.Llama(model_path=str(Path(root)/name/'model.gguf'),
                    n_ctx=4096,n_batch=4096,n_gpu_layers=-1 if device=='metal' else 0,
                    logits_all=False,verbose=False))
            self.score('Hello.')
            self.ready=True
        except BaseException:
            self.close()
            raise

    def score(self,text):
        started=time.perf_counter(); inputs=[]
        # Preflight both tokenizers before doing either model's inference.
        for model,tokenizer in zip(self.models,self.tokenizers):
            rendered=tokenizer.apply_chat_template(
                [{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':text}],
                tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=model.tokenize(rendered.encode('utf-8'),add_bos=False,special=True)
            if len(ids)>1024: raise InputTooLarge('Templated input exceeds 1024 tokens')
            if ids!=tokenizer(rendered)['input_ids']:
                raise ArtifactError('GGUF tokenizer differs from pinned tokenizer')
            inputs.append(ids)
        import numpy as np
        scores=[]
        for model,ids in zip(self.models,inputs):
            model.reset(); model.eval(ids)
            logits=np.ctypeslib.as_array(model._ctx.get_logits(),shape=(model._n_vocab,))
            a,b=float(logits[38157]),float(logits[18675])
            if not math.isfinite(a) or not math.isfinite(b): raise ArtifactError('Invalid class logits')
            if a>b: value=1/(1+math.exp(b-a))
            else:
                exp=math.exp(a-b); value=exp/(1+exp)
            scores.append(value)
        return ScoreResult(sum(scores)/2,max(map(len,inputs)),(time.perf_counter()-started)*1000)

    def close(self):
        self.ready=False
        # Release every successfully loaded context, including partial startup.
        models,self.models=self.models,[]
        self.tokenizers=[]
        for model in models:
            try: model.close()
            except Exception: pass
