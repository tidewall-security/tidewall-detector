"""Contract tests use a controllable scorer; real-weight parity is separate."""
import threading
import time
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from tidewall_lora.api import create_app, InferenceSettings
from tidewall_lora.classifier import ScoreResult

TENANT=str(uuid4())
PROFILE='tidewall.prompt-injection.1'


class Scorer:
    ready=True
    calls=0
    entered=threading.Event()
    release=threading.Event()
    block=False
    def score(self,text):
        self.calls+=1; self.entered.set()
        if self.block: self.release.wait(2)
        return ScoreResult(.9,20,1.)
    def close(self): pass


@pytest.fixture
def system(tmp_path):
    key=tmp_path/'key'; key.write_text('a'*64)
    scorer=Scorer(); scorer.entered=threading.Event(); scorer.release=threading.Event()
    app=create_app(InferenceSettings(tenant_id=TENANT,api_key_file=str(key)),scorer)
    with TestClient(app) as c: yield c,scorer


def headers(**changes):
    return {'Authorization':'Bearer '+'a'*64,'X-Tidewall-Tenant-ID':TENANT,**changes}


def body(**changes):
    return {'schema_version':1,'profile':PROFILE,'request_id':str(uuid4()),'event_kind':'prompt',
            'messages':[{'role':'user','content':'Hello'}],'timeout_ms':1000,**changes}


def post(system,value=None,**kwargs):
    return system[0].post('/v1/classify',json=value or body(),headers=kwargs.pop('headers',headers()),**kwargs)


def test_native_result_has_identity_and_no_policy(system):
    value=body(); r=post(system,value)
    assert r.status_code==200,r.text
    x=r.json()
    assert x['profile']==PROFILE and x['schema_version']==1 and x['request_id']==value['request_id']
    assert x['status']=='classified' and x['unsafe_score']==.9
    assert x['score_kind']=='two_class_softmax_uncalibrated'
    assert x['model']['base_revision'] and x['model']['tokenizer_revision'] and x['model']['adapter_revision']
    assert not {'decision','threshold','confidence','score','category','level'} & x.keys()


@pytest.mark.parametrize('h,code,status', [({},'unauthenticated',401),
    ({'Authorization':'Bearer bad','X-Tidewall-Tenant-ID':TENANT},'unauthenticated',401),
    (headers(**{'X-Tidewall-Tenant-ID':str(uuid4())}),'tenant_mismatch',403)])
def test_auth_is_tenant_bound(system,h,code,status):
    r=post(system,headers=h)
    assert r.status_code==status and r.json()['code']==code
    assert system[1].calls==0


@pytest.mark.parametrize('changes', [{'extra':'field'},{'timeout_ms':True},{'timeout_ms':0},
    {'schema_version':2},{'request_id':'invalid'},{'profile':'guardrail.1'},
    {'messages':[{'role':'system','content':'Hello'}]},{'messages':[]},
    {'messages':[{'role':'user','content':'one'},{'role':'user','content':'two'}]},
    {'messages':[{'role':'user','content':'  '}]}, {'event_kind':'command'}])
def test_strict_request_has_safe_machine_error(system,changes):
    r=post(system,body(**changes))
    assert r.status_code==422,r.text
    assert r.json()['status']=='error' and r.json()['code']=='invalid_request'
    assert 'messages' not in r.text and system[1].calls==0


@pytest.mark.parametrize('text,code,status',[('é'*4001,'input_too_large',413),
    ('Documentation for <|im_start|>.','unsupported_input',422)])
def test_coverage_limits(system,text,code,status):
    r=post(system,body(messages=[{'role':'user','content':text}]))
    assert r.status_code==status and r.json()['code']==code
    assert system[1].calls==0


def test_replay_is_bound_to_content(system):
    value=body(); first=post(system,value); second=post(system,{**value,'timeout_ms':500})
    assert first.status_code==second.status_code==200
    assert first.json()==second.json() and system[1].calls==1
    value['messages'][0]['content']='Changed'
    r=post(system,value)
    assert r.status_code==409 and r.json()['code']=='request_conflict'


def test_deadline_does_not_release_model_slot(system):
    from concurrent.futures import ThreadPoolExecutor
    c,scorer=system; scorer.block=True
    value=body(timeout_ms=40)
    with ThreadPoolExecutor() as pool:
        pending=pool.submit(post,system,value)
        assert scorer.entered.wait(1)
        timed_out=pending.result(timeout=1)
        assert timed_out.status_code==504 and timed_out.json()['code']=='deadline_exceeded'
        busy=post(system)
        assert busy.status_code==503 and busy.json()['code']=='capacity_exceeded'
        assert scorer.calls==1
        scorer.release.set()
    for _ in range(50):
        retry=post(system,value)
        if retry.status_code!=503: break
        time.sleep(.01)
    assert retry.status_code==409  # same timed-out ID cannot silently become fresh work


def test_not_ready_has_no_safe_verdict(system):
    system[1].ready=False
    r=post(system)
    assert r.status_code==503 and r.json()['code']=='model_unavailable'
    assert 'unsafe_score' not in r.json()


def test_error_does_not_echo_large_body(system):
    r=system[0].post('/v1/classify',content=b'x'*13000,headers={**headers(),'Content-Type':'application/json'})
    assert r.status_code==413 and 'xxxx' not in r.text


def test_duplicate_auth_header_is_rejected(system):
    h=list(headers().items())+[('Authorization','Bearer '+'a'*64)]
    r=system[0].post('/v1/classify',json=body(),headers=h)
    assert r.status_code==401 and system[1].calls==0


def test_template_token_overflow_is_not_scored(system,monkeypatch):
    from tidewall_lora.classifier import InputTooLarge
    def large(text): raise InputTooLarge('private prompt must not be echoed')
    monkeypatch.setattr(system[1],'score',large)
    r=post(system)
    assert r.status_code==413 and r.json()['code']=='input_too_large' and 'private' not in r.text


def test_cache_capacity_is_bounded(tmp_path):
    key=tmp_path/'key'; key.write_text('a'*64)
    scorer=Scorer()
    with TestClient(create_app(InferenceSettings(TENANT,str(key),cache_size=1),scorer)) as client:
        s=(client,scorer)
        assert post(s).status_code==200
        r=post(s)
        assert r.status_code==503 and r.json()['code']=='capacity_exceeded'
        assert scorer.calls==1


def test_model_error_is_not_retried_under_same_identity(system,monkeypatch):
    def broken(text): raise RuntimeError('private model failure')
    monkeypatch.setattr(system[1],'score',broken)
    value=body(); r=post(system,value)
    assert r.status_code==503 and 'private' not in r.text
    again=post(system,value)
    assert again.status_code==409 and again.json()['code']=='request_conflict'


def test_cancelled_replay_does_not_erase_success(tmp_path,monkeypatch):
    import asyncio
    import contextlib
    import httpx
    key=tmp_path/'key'; key.write_text('a'*64)
    async def run():
        app=create_app(InferenceSettings(TENANT,str(key)),Scorer())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app),base_url='http://test') as client:
                value=body()
                first=await client.post('/v1/classify',json=value,headers=headers())
                assert first.status_code==200
                # Hold the await boundary so cancellation deterministically lands
                # after a cached operation is selected, before its reply returns.
                from tidewall_lora import api
                with monkeypatch.context() as patch:
                    patch.setattr(api.asyncio,'wrap_future',lambda future:asyncio.get_running_loop().create_future())
                    replay=asyncio.create_task(client.post('/v1/classify',json=value,headers=headers()))
                    await asyncio.sleep(0)
                    replay.cancel()
                    with contextlib.suppress(asyncio.CancelledError): await replay
                retry=await client.post('/v1/classify',json=value,headers=headers())
                assert retry.status_code==200 and retry.json()==first.json()
    asyncio.run(run())


def test_inherited_nonloopback_socket_cannot_bypass_tls(tmp_path,monkeypatch):
    import socket
    import sys
    import uvicorn
    from tidewall_lora.__main__ import main
    key=tmp_path/'key'; key.write_text('a'*64)
    for k,v in {'TIDEWALL_TENANT_ID':TENANT,'TIDEWALL_API_KEY_FILE':str(key),'TIDEWALL_MODEL_DIR':str(tmp_path)}.items():
        monkeypatch.setenv(k,v)
    def run(app,**kwargs): app.state.runtime.shutdown()
    monkeypatch.setattr(uvicorn,'run',run)
    with socket.socket() as listener:
        listener.bind(('0.0.0.0',0)); listener.listen()
        monkeypatch.setattr(sys,'argv',['tidewall_lora','--fd',str(listener.fileno())])
        with pytest.raises(SystemExit): main()


def test_unpaired_unicode_is_a_machine_error(system):
    import json
    value=body(messages=[{'role':'user','content':'\ud800'}])
    r=system[0].post('/v1/classify',content=json.dumps(value),headers={**headers(),'Content-Type':'application/json'})
    assert r.status_code==422 and r.json()['code']=='invalid_request'
    assert system[1].calls==0


def test_inherited_loopback_listener_is_accepted(tmp_path,monkeypatch):
    import socket
    import sys
    import uvicorn
    from tidewall_lora.__main__ import main
    key=tmp_path/'key'; key.write_text('a'*64)
    for k,v in {'TIDEWALL_TENANT_ID':TENANT,'TIDEWALL_API_KEY_FILE':str(key),'TIDEWALL_MODEL_DIR':str(tmp_path)}.items():
        monkeypatch.setenv(k,v)
    calls=[]
    def run(app,**kwargs): calls.append(kwargs); app.state.runtime.shutdown()
    monkeypatch.setattr(uvicorn,'run',run)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0)); listener.listen()
        monkeypatch.setattr(sys,'argv',['tidewall_lora','--fd',str(listener.fileno())])
        main()
    assert len(calls)==1


def test_ensemble_profile_is_explicit_and_errors_keep_selected_identity(tmp_path):
    from tidewall_lora.ensemble import identity
    profile='tidewall.prompt-injection.ensemble-q8.1'
    key=tmp_path/'key'; key.write_text('a'*64)
    scorer=Scorer(); scorer.calls=0
    with TestClient(create_app(InferenceSettings(TENANT,str(key),profile=profile),scorer)) as client:
        assert client.get('/health/ready').json()=={'status':'ready','profile':profile}
        result=client.post('/v1/classify',json=body(profile=profile),headers=headers())
        assert result.status_code==200
        assert result.json()['model']==identity()
        assert result.json()['score_kind']=='mean_of_two_class_softmax_uncalibrated'
        mismatch=client.post('/v1/classify',json=body(),headers=headers())
        assert mismatch.status_code==422 and mismatch.json()['profile']==profile
        assert mismatch.json()['code']=='unsupported_input' and scorer.calls==1
        denied=client.post('/v1/classify',json=body(profile=profile))
        assert denied.status_code==401 and denied.json()['profile']==profile
        invalid=client.post('/v1/classify',json=body(profile=profile,messages=[]),headers=headers())
        assert invalid.status_code==422 and invalid.json()['profile']==profile


def test_ensemble_keeps_single_user_boundary(tmp_path):
    key=tmp_path/'key'; key.write_text('a'*64)
    scorer=Scorer(); scorer.calls=0
    profile='tidewall.prompt-injection.ensemble-q8.1'
    with TestClient(create_app(InferenceSettings(TENANT,str(key),profile=profile),scorer)) as client:
        response=client.post('/v1/classify',json=body(profile=profile,messages=[
            {'role':'user','content':'one'},{'role':'assistant','content':'two'}]),headers=headers())
        assert response.status_code==422 and scorer.calls==0


def test_ensemble_deadline_holds_physical_slot_and_replay_is_content_bound(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    key=tmp_path/'key'; key.write_text('a'*64)
    profile='tidewall.prompt-injection.ensemble-q8.1'
    scorer=Scorer(); scorer.calls=0; scorer.entered=threading.Event(); scorer.release=threading.Event(); scorer.block=True
    with TestClient(create_app(InferenceSettings(TENANT,str(key),profile=profile),scorer)) as client:
        value=body(profile=profile,timeout_ms=20)
        with ThreadPoolExecutor() as pool:
            pending=pool.submit(client.post,'/v1/classify',json=value,headers=headers())
            assert scorer.entered.wait(1)
            assert pending.result(timeout=1).status_code==504
        busy=client.post('/v1/classify',json=body(profile=profile),headers=headers())
        assert busy.status_code==503 and busy.json()['code']=='capacity_exceeded'
        assert busy.json()['profile']==profile
        scorer.release.set()
        for _ in range(100):
            if not client.app.state.runtime.active: break
            time.sleep(.01)
        abandoned=client.post('/v1/classify',json=value,headers=headers())
        assert abandoned.status_code==409
        scorer.block=False; fresh=body(profile=profile)
        first=client.post('/v1/classify',json=fresh,headers=headers())
        second=client.post('/v1/classify',json=fresh,headers=headers())
        assert first.status_code==second.status_code==200 and first.json()==second.json()
        assert scorer.calls==2


@pytest.mark.parametrize('size,ttl,expected',[(None,None,(1024,300)),('4096','300.5',(4096,300.5)),
    ('1','1',(1,1)),('100000','3600',(100000,3600))])
def test_entry_point_configures_bounded_replay_cache(tmp_path,monkeypatch,size,ttl,expected):
    import sys,uvicorn
    from tidewall_lora.__main__ import main
    key=tmp_path/'key';key.write_text('a'*64)
    for name,value in {'TIDEWALL_TENANT_ID':TENANT,'TIDEWALL_API_KEY_FILE':str(key),
                       'TIDEWALL_MODEL_DIR':str(tmp_path),'TIDEWALL_PROFILE':PROFILE,'TIDEWALL_DEVICE':'cpu'}.items():
        monkeypatch.setenv(name,value)
    for name,value in [('TIDEWALL_CACHE_SIZE',size),('TIDEWALL_CACHE_TTL_SECONDS',ttl)]:
        if value is None:monkeypatch.delenv(name,raising=False)
        else:monkeypatch.setenv(name,value)
    received=[]
    def run(app,**kwargs):
        settings=app.state.runtime.settings
        received.append((settings.cache_size,settings.cache_ttl_seconds))
        app.state.runtime.shutdown()
    monkeypatch.setattr(uvicorn,'run',run);monkeypatch.setattr(sys,'argv',['tidewall_lora'])
    main()
    assert received==[expected]


@pytest.mark.parametrize('name,value',[('TIDEWALL_CACHE_SIZE',value) for value in ['0','-1','100001','1.5','nan','inf','invalid']]+
    [('TIDEWALL_CACHE_TTL_SECONDS',value) for value in ['0','3601','nan','inf','-inf','invalid']])
def test_entry_point_rejects_invalid_cache_before_launch(tmp_path,monkeypatch,name,value):
    import sys,uvicorn
    from tidewall_lora.__main__ import main
    key=tmp_path/'key';key.write_text('a'*64)
    for setting,text in {'TIDEWALL_TENANT_ID':TENANT,'TIDEWALL_API_KEY_FILE':str(key),
                         'TIDEWALL_MODEL_DIR':str(tmp_path),'TIDEWALL_PROFILE':PROFILE,'TIDEWALL_DEVICE':'cpu'}.items():
        monkeypatch.setenv(setting,text)
    monkeypatch.delenv('TIDEWALL_CACHE_SIZE',raising=False);monkeypatch.delenv('TIDEWALL_CACHE_TTL_SECONDS',raising=False)
    monkeypatch.setenv(name,value);launched=[]
    def run(app,**kwargs):launched.append(True);app.state.runtime.shutdown()
    monkeypatch.setattr(uvicorn,'run',run);monkeypatch.setattr(sys,'argv',['tidewall_lora'])
    with pytest.raises(ValueError):main()
    assert not launched


@pytest.mark.parametrize('profile',[PROFILE,'tidewall.prompt-injection.ensemble-q8.1'])
def test_configured_cache_preserves_replay_conflict_tenant_and_expiry(tmp_path,monkeypatch,profile):
    from types import SimpleNamespace
    from tidewall_lora import api
    clock=[1000.0]
    # Replace this module's clock, not asyncio/httpx's process-global clock.
    monkeypatch.setattr(api,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    key=tmp_path/'key';key.write_text('a'*64);scorer=Scorer();scorer.calls=0
    app=create_app(InferenceSettings(TENANT,str(key),cache_size=2,cache_ttl_seconds=1,profile=profile),scorer)
    with TestClient(app) as client:
        first=body(profile=profile);second=body(profile=profile);third=body(profile=profile)
        original=client.post('/v1/classify',json=first,headers=headers())
        assert original.status_code==200
        assert client.post('/v1/classify',json=second,headers=headers()).status_code==200
        assert len(app.state.runtime.records)==2 and scorer.calls==2
        clock[0]+=0.999
        replay=client.post('/v1/classify',json=first,headers=headers())
        assert replay.status_code==200 and replay.json()==original.json() and scorer.calls==2
        changed={**first,'messages':[{'role':'user','content':'Changed content'}]}
        assert client.post('/v1/classify',json=changed,headers=headers()).status_code==409
        foreign=headers(**{'X-Tidewall-Tenant-ID':str(uuid4())})
        assert client.post('/v1/classify',json=first,headers=foreign).status_code==403
        full=client.post('/v1/classify',json=third,headers=headers())
        assert full.status_code==503 and full.json()['code']=='capacity_exceeded'
        assert len(app.state.runtime.records)==2 and scorer.calls==2
        clock[0]+=0.002
        assert client.post('/v1/classify',json=third,headers=headers()).status_code==200
        assert len(app.state.runtime.records)==1 and scorer.calls==3
        assert client.post('/v1/classify',json=changed,headers=headers()).status_code==200
        assert len(app.state.runtime.records)==2 and scorer.calls==4
