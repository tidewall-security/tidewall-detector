"""The native ensemble preserves the recovered recipe and bounded input contract."""
import ctypes
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
from tidewall_lora import ensemble
from tidewall_lora.classifier import ArtifactError, InputTooLarge


def test_manifest_rejects_tampering_extra_files_and_symlinks(tmp_path, monkeypatch):
    root=tmp_path/'models'; (root/'broad').mkdir(parents=True)
    path=root/'broad/model.gguf'; path.write_bytes(b'weights')
    manifest={'prompt_sha256':hashlib.sha256(ensemble.SYSTEM_PROMPT.encode()).hexdigest(),
              'files':{'broad/model.gguf':{'bytes':7,'sha256':hashlib.sha256(b'weights').hexdigest()}}}
    pin=tmp_path/'pin.json'; pin.write_text(json.dumps(manifest)); monkeypatch.setattr(ensemble,'MANIFEST_PATH',pin)
    assert ensemble.verify_manifest(root)==manifest
    path.write_bytes(b'changed')
    with pytest.raises(ArtifactError): ensemble.verify_manifest(root)
    path.write_bytes(b'weights'); extra=root/'broad/extra'; extra.write_text('extra')
    with pytest.raises(ArtifactError): ensemble.verify_manifest(root)
    extra.unlink(); path.unlink(); path.symlink_to(pin)
    with pytest.raises(ArtifactError): ensemble.verify_manifest(root)


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    models=[]
    class Model:
        def __init__(self,**kw):
            self.kw=kw; self.closed=False; self.evaluations=0; self.tokens=[1,2]; self._n_vocab=38158
            self.logits=(ctypes.c_float*self._n_vocab)(); self.logits[38157]=2 if not models else -2
            self._ctx=SimpleNamespace(get_logits=lambda:self.logits); models.append(self)
        def tokenize(self,*a,**kw): return self.tokens
        def reset(self): pass
        def eval(self,tokens): self.evaluations+=1
        def close(self): self.closed=True
    class Tokenizer:
        def apply_chat_template(self,messages,**kw): return 'rendered'
        def __call__(self,text): return {'input_ids':[1,2]}
    monkeypatch.setattr(ensemble,'verify_manifest',lambda root: {})
    monkeypatch.setitem(sys.modules,'llama_cpp',SimpleNamespace(__version__='0.3.37',Llama=Model,
        llama_supports_gpu_offload=lambda:True))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(__version__='4.51.3',AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *a,**kw:Tokenizer())))
    return models,tmp_path


def test_sequential_mean_and_cleanup(runtime,monkeypatch):
    monkeypatch.setattr(sys,'platform','linux')
    models,root=runtime; scorer=ensemble.EnsembleClassifier(root,'cpu')
    result=scorer.score('Hello')
    assert result.unsafe_score==pytest.approx(.5) and result.input_tokens==2
    assert len(models)==2 and all(m.kw['n_gpu_layers']==0 for m in models)
    scorer.close(); assert all(m.closed for m in models) and not scorer.ready


def test_all_tokenizers_preflight_before_either_evaluation(runtime):
    models,root=runtime; scorer=ensemble.EnsembleClassifier(root)
    before=[m.evaluations for m in models]; models[1].tokens=list(range(1025))
    with pytest.raises(InputTooLarge): scorer.score('Too large')
    assert [m.evaluations for m in models]==before
    scorer.close()


def test_tokenizer_divergence_is_unavailable(runtime):
    models,root=runtime; scorer=ensemble.EnsembleClassifier(root)
    models[1].tokens=[3,4]
    with pytest.raises(ArtifactError): scorer.score('Hello')
    scorer.close()


def test_partial_startup_closes_loaded_contexts(runtime,monkeypatch):
    models,root=runtime; module=sys.modules['llama_cpp']; original=module.Llama
    def load(**kwargs):
        if models: raise RuntimeError('load failed')
        return original(**kwargs)
    monkeypatch.setattr(module,'Llama',load)
    with pytest.raises(RuntimeError): ensemble.EnsembleClassifier(root)
    assert len(models)==1 and models[0].closed


def test_wrong_runtime_and_unavailable_gpu_rejected(runtime,monkeypatch):
    models,root=runtime; module=sys.modules['llama_cpp']
    monkeypatch.setattr(module,'__version__','different')
    with pytest.raises(ArtifactError): ensemble.EnsembleClassifier(root)
    module.__version__='0.3.37'; module.llama_supports_gpu_offload=lambda:False
    with pytest.raises(ArtifactError): ensemble.EnsembleClassifier(root,'metal')
    assert not models


def test_parent_symlink_rotation_does_not_change_verified_release(runtime,monkeypatch):
    models,root=runtime
    first=root/'first'; second=root/'second'
    (first/'models').mkdir(parents=True); (second/'models').mkdir(parents=True)
    current=root/'current'; current.symlink_to(first,target_is_directory=True)
    def verify(path):
        assert path==first/'models'
        current.unlink(); current.symlink_to(second,target_is_directory=True)
    monkeypatch.setattr(ensemble,'verify_manifest',verify)
    scorer=ensemble.EnsembleClassifier(current/'models')
    assert all(str(first/'models') in m.kw['model_path'] for m in models)
    scorer.close()


@pytest.mark.parametrize('library',['transformers','numpy'])
def test_changed_rendering_or_numeric_runtime_refuses_readiness(runtime,monkeypatch,library):
    models,root=runtime
    if library=='numpy':
        import numpy
    monkeypatch.setattr(sys.modules[library],'__version__','unqualified')
    with pytest.raises(ArtifactError): ensemble.EnsembleClassifier(root)
    assert not models


def test_metal_argument_routing_is_explicit(runtime,monkeypatch):
    monkeypatch.setattr(sys,'platform','darwin')
    models,root=runtime; scorer=ensemble.EnsembleClassifier(root,'metal')
    assert all(m.kw['n_gpu_layers']==-1 for m in models)
    scorer.close()
