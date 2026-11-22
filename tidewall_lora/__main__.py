"""Run one resident model worker; CPU by default, loopback by default."""
import argparse
import os
import socket
import ipaddress
import errno
from pathlib import Path


def env_file(name):
    value=os.environ.get(name,'')
    path=os.environ.get(name+'_FILE','')
    if value and path: raise ValueError('Set only one of '+name+' and its file setting')
    return Path(path).read_text().strip() if path else value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8107)
    parser.add_argument('--fd',type=int)
    parser.add_argument('--ssl-certfile')
    parser.add_argument('--ssl-keyfile')
    parser.add_argument('--config', type=Path, help='Standalone service.json; do not combine with setting environment variables')
    args=parser.parse_args()
    host=args.host
    if args.fd is not None:
        try:
            with socket.socket(fileno=os.dup(args.fd)) as listener:
                if listener.family not in (socket.AF_INET,socket.AF_INET6) or listener.type!=socket.SOCK_STREAM:
                    parser.error('Inherited descriptor must be a TCP socket')
                bound=listener.getsockname()
                if not bound[1]: parser.error('Inherited descriptor must be bound')
                try:
                    if not listener.getsockopt(socket.SOL_SOCKET,socket.SO_ACCEPTCONN):
                        parser.error('Inherited descriptor must be listening')
                except OSError as exc:
                    # Darwin exposes the constant but rejects querying this option.
                    # Family, type and actual bound address remain validated.
                    if exc.errno!=errno.ENOPROTOOPT: raise
                host=bound[0]
        except OSError:
            parser.error('Invalid inherited listener')
    try: loopback=ipaddress.ip_address(host).is_loopback
    except ValueError: loopback=False
    if not loopback and not (args.ssl_certfile and args.ssl_keyfile):
        parser.error('Non-loopback listeners require TLS')
    # Set before model-library imports. Only staged local artifacts are loaded.
    for name in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','TOKENIZERS_PARALLELISM'):
        os.environ[name]='false' if name=='TOKENIZERS_PARALLELISM' else '1'
    from .api import create_app,InferenceSettings
    import uvicorn
    if args.config:
        from .prepare import load_settings
        try:
            settings=load_settings(args.config)
        except (OSError,ValueError,RecursionError):
            parser.error('Invalid service configuration; check file fields, key and conflicting environment settings')
    else:
        settings=InferenceSettings(tenant_id=env_file('TIDEWALL_TENANT_ID'),
            api_key_file=os.environ['TIDEWALL_API_KEY_FILE'],model_dir=os.environ['TIDEWALL_MODEL_DIR'],
            device=os.environ.get('TIDEWALL_DEVICE','cpu'),
            profile=os.environ.get('TIDEWALL_PROFILE','tidewall.prompt-injection.1'),
            cache_size=int(os.environ.get('TIDEWALL_CACHE_SIZE','1024')),
            cache_ttl_seconds=float(os.environ.get('TIDEWALL_CACHE_TTL_SECONDS','300')))
    uvicorn.run(create_app(settings),host=args.host,port=args.port,fd=args.fd,workers=1,
        ssl_certfile=args.ssl_certfile,ssl_keyfile=args.ssl_keyfile,access_log=False,limit_concurrency=32,
        timeout_graceful_shutdown=15)


if __name__=='__main__': main()
