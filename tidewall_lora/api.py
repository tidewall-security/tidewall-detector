"""Single-worker inference with bounded admission and tenant/content-bound replay."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
import threading
import time
from uuid import UUID

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .classifier import InputTooLarge, RESERVED_TOKENS
from .schemas import ClassifyRequest, PROFILE
from .profiles import PROFILES

STATUS={'invalid_request':422,'unsupported_input':422,'input_too_large':413,'unauthenticated':401,
    'tenant_mismatch':403,'model_unavailable':503,'capacity_exceeded':503,'deadline_exceeded':504,'request_conflict':409}
RETRYABLE={'model_unavailable','capacity_exceeded','deadline_exceeded'}


def failure(code,request_id=None,profile=PROFILE):
    return {'schema_version':1,'profile':profile,'status':'error','code':code,
            'request_id':request_id,'retryable':code in RETRYABLE}


def failed_response(code,request_id=None,profile=PROFILE):
    return JSONResponse(failure(code,request_id,profile),status_code=STATUS[code],headers={'Cache-Control':'no-store'})


@dataclass(frozen=True)
class InferenceSettings:
    tenant_id: str
    api_key_file: str
    model_dir: str=''
    device: str='cpu'
    max_timeout_ms: int=1500
    cache_size: int=1024
    cache_ttl_seconds: float=300

    profile: str=PROFILE

    def validate(self):
        if str(UUID(self.tenant_id))!=self.tenant_id: raise ValueError('Canonical tenant UUID required')
        key=Path(self.api_key_file).read_text().strip()
        if len(key)<32 or not key.isascii() or any(c.isspace() for c in key): raise ValueError('Invalid service key')
        if type(self.max_timeout_ms) is not int or not 1<=self.max_timeout_ms<=10000: raise ValueError('Invalid timeout')
        if type(self.cache_size) is not int or not 1<=self.cache_size<=100000: raise ValueError('Invalid cache size')
        if not math.isfinite(self.cache_ttl_seconds) or not 1<=self.cache_ttl_seconds<=3600: raise ValueError('Invalid cache TTL')
        if self.profile not in PROFILES or self.device not in PROFILES[self.profile]['devices']: raise ValueError('Unsupported device')
        return key


class InputBoundary:
    """Authenticate and bound raw body before the framework parses it."""
    def __init__(self,app,*,key,tenant,profile=PROFILE):
        self.app,self.key,self.tenant,self.profile=app,key,tenant,profile

    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope['path']!='/v1/classify':
            return await self.app(scope,receive,send)
        headers={}
        for name,value in scope['headers']: headers.setdefault(name,[]).append(value)
        auth=headers.get(b'authorization',[])
        if len(auth)!=1 or not hmac.compare_digest(auth[0],('Bearer '+self.key).encode()):
            return await failed_response('unauthenticated',profile=self.profile)(scope,receive,send)
        tenant=headers.get(b'x-tidewall-tenant-id',[])
        if tenant!=[self.tenant.encode()]:
            return await failed_response('tenant_mismatch',profile=self.profile)(scope,receive,send)
        if (headers.get(b'content-encoding',[b'identity'])!=[b'identity']
                or len(headers.get(b'content-type',[]))!=1
                or headers[b'content-type'][0].split(b';',1)[0].strip().lower()!=b'application/json'):
            return await failed_response('invalid_request',profile=self.profile)(scope,receive,send)
        scope.setdefault('state',{})['admitted_at']=time.monotonic()
        content=bytearray()
        try:
            async with asyncio.timeout(2):
                while True:
                    message=await receive()
                    if message['type']=='http.disconnect': return
                    chunk=message.get('body',b'')
                    if len(chunk)>12000-len(content):
                        return await failed_response('input_too_large',profile=self.profile)(scope,receive,send)
                    content.extend(chunk)
                    if not message.get('more_body'): break
        except TimeoutError:
            return await failed_response('deadline_exceeded',profile=self.profile)(scope,receive,send)
        # Parse once to reject pathological JSON without echoing parser details.
        try: json.loads(content)
        except (ValueError,RecursionError,OverflowError):
            return await failed_response('invalid_request',profile=self.profile)(scope,receive,send)
        sent=False
        async def bounded_receive():
            nonlocal sent
            if not sent:
                sent=True
                return {'type':'http.request','body':bytes(content),'more_body':False}
            return await receive()
        return await self.app(scope,bounded_receive,send)


class Runtime:
    def __init__(self,settings,scorer):
        self.settings,self.scorer=settings,scorer
        self.lock=threading.Lock()
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='tidewall-model')
        self.records={}
        self.active=False
        self.accepting=True
        self.load_failed=False

    def load(self):
        try:
            scorer=PROFILES[self.settings.profile]['loader'](self.settings.model_dir,self.settings.device)
            with self.lock: self.scorer=scorer
        except Exception:
            with self.lock: self.load_failed=True

    def ready(self): return self.accepting and self.scorer is not None and self.scorer.ready

    def work(self,ident,text,deadline):
        try:
            if time.monotonic()>=deadline: return failure('deadline_exceeded',ident,self.settings.profile),504
            result=self.scorer.score(text)
            score=result.unsafe_score
            if (type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1
                    or type(result.input_tokens) is not int or not 1<=result.input_tokens<=1024
                    or not math.isfinite(result.inference_ms) or result.inference_ms<0):
                return failure('model_unavailable',ident,self.settings.profile),503
            if time.monotonic()>=deadline: return failure('deadline_exceeded',ident,self.settings.profile),504
            return {'schema_version':1,'profile':self.settings.profile,'request_id':ident,'status':'classified',
                'unsafe_score':score,'score_kind':PROFILES[self.settings.profile]['score_kind'],
                'model':PROFILES[self.settings.profile]['identity'](),
                'input_tokens':result.input_tokens,'inference_ms':result.inference_ms},200
        except InputTooLarge: return failure('input_too_large',ident,self.settings.profile),413
        except Exception: return failure('model_unavailable',ident,self.settings.profile),503

    def done(self,ident,future):
        with self.lock:
            self.active=False
            entry=self.records.get(ident)
            if entry is None: return
            if entry['abandoned'] or future.result()[1]!=200: entry['abandoned']=True

    def admit(self,body,deadline):
        ident=body.request_id
        digest=hashlib.sha256(body.model_dump_json(exclude={'timeout_ms','request_id'}).encode()).hexdigest()
        with self.lock:
            now=time.monotonic()
            for old,entry in list(self.records.items()):
                if entry['expires']<now and entry['future'].done(): del self.records[old]
            if ident in self.records:
                entry=self.records[ident]
                if entry['digest']!=digest or entry['abandoned']: return None,'request_conflict'
                if not entry['future'].done(): return None,'capacity_exceeded'
                return entry['future'],None
            if not self.ready(): return None,'model_unavailable'
            if self.active or len(self.records)>=self.settings.cache_size: return None,'capacity_exceeded'
            self.active=True
            future=self.executor.submit(self.work,ident,body.messages[0].content,deadline)
            self.records[ident]={'digest':digest,'future':future,'abandoned':False,
                'expires':now+self.settings.cache_ttl_seconds}
        future.add_done_callback(lambda f:self.done(ident,f))
        return future,None

    def abandon(self,ident):
        with self.lock:
            if ident in self.records and not self.records[ident]['future'].done():
                self.records[ident]['abandoned']=True

    def shutdown(self):
        with self.lock: self.accepting=False
        self.executor.shutdown(wait=True,cancel_futures=False)
        if self.scorer is not None: self.scorer.close()


def create_app(settings,classifier=None):
    key=settings.validate()
    runtime=Runtime(settings,classifier)
    @asynccontextmanager
    async def lifespan(app):
        if classifier is None: runtime.executor.submit(runtime.load)
        try: yield
        finally: await asyncio.to_thread(runtime.shutdown)
    app=FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
    app.state.runtime=runtime
    app.add_middleware(InputBoundary,key=key,tenant=settings.tenant_id,profile=settings.profile)

    @app.exception_handler(RequestValidationError)
    async def invalid(request,exc):
        ident=None
        if isinstance(exc.body,dict):
            try:
                value=exc.body.get('request_id')
                if isinstance(value,str) and str(UUID(value))==value: ident=value
            except (ValueError,TypeError): pass
        return failed_response('invalid_request',ident,settings.profile)

    @app.get('/health/live')
    async def live(): return {'status':'alive'}

    @app.get('/health/ready')
    async def ready():
        state='ready' if runtime.ready() else 'unavailable' if runtime.load_failed else 'loading'
        return JSONResponse({'status':state,'profile':settings.profile},status_code=200 if state=='ready' else 503)

    from fastapi import Request
    @app.post('/v1/classify')
    async def classify(body:ClassifyRequest,request:Request):
        if body.profile!=settings.profile:
            return failed_response('unsupported_input',body.request_id,settings.profile)
        text=body.messages[0].content
        if len(text.encode())>8000: return failed_response('input_too_large',body.request_id,settings.profile)
        if any(token in text for token in RESERVED_TOKENS): return failed_response('unsupported_input',body.request_id,settings.profile)
        deadline=request.state.admitted_at+min(body.timeout_ms,settings.max_timeout_ms)/1000
        if time.monotonic()>=deadline: return failed_response('deadline_exceeded',body.request_id,settings.profile)
        future,code=runtime.admit(body,deadline)
        if code: return failed_response(code,body.request_id,settings.profile)
        if future.done():
            result,status=future.result()
            return JSONResponse(result,status_code=status,headers={'Cache-Control':'no-store'})
        try:
            result,status=await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)),max(0,deadline-time.monotonic()))
            if time.monotonic()>=deadline:
                runtime.abandon(body.request_id)
                return failed_response('deadline_exceeded',body.request_id,settings.profile)
            return JSONResponse(result,status_code=status,headers={'Cache-Control':'no-store'})
        except TimeoutError:
            runtime.abandon(body.request_id)
            return failed_response('deadline_exceeded',body.request_id,settings.profile)
        except asyncio.CancelledError:
            runtime.abandon(body.request_id)
            raise
    return app
