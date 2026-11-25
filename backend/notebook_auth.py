"""Optional HTTP Basic authentication for temporary Kaggle notebook tunnels."""
import base64
import hmac
from urllib.parse import urlsplit
from starlette.responses import PlainTextResponse


class NotebookAuth:
    def __init__(self,app,password):
        self.app,self.password=app,password

    async def __call__(self,scope,receive,send):
        if scope['type'] not in ('http','websocket'):
            return await self.app(scope,receive,send)
        headers={key.lower():value for key,value in scope.get('headers',[])}
        valid=False
        try:
            scheme,encoded=headers.get(b'authorization',b'').decode('ascii').split(' ',1)
            user,password=base64.b64decode(encoded,validate=True).decode('utf-8').split(':',1)
            valid=scheme.lower()=='basic' and hmac.compare_digest(user.encode(),b'nanoforge') and hmac.compare_digest(password.encode(),self.password.encode())
        except (ValueError,UnicodeError):pass
        if not valid:
            if scope['type']=='websocket':return await send({'type':'websocket.close','code':1008})
            response=PlainTextResponse('Sign in with the notebook username/password.',status_code=401,
                                       headers={'WWW-Authenticate':'Basic realm="NanoForge", charset="UTF-8"','Cache-Control':'no-store'})
            return await response(scope,receive,send)
        if scope['type']=='http' and scope['method'] not in ('GET','HEAD','OPTIONS'):
            origin=headers.get(b'origin')
            host=headers.get(b'host',b'').decode('ascii').lower()
            if origin:
                try:
                    source=urlsplit(origin.decode('ascii'))
                    target=urlsplit(scope['scheme']+'://'+host)
                    def port(url):return url.port or (443 if url.scheme=='https' else 80)
                    same_origin=(source.scheme==target.scheme and source.hostname==target.hostname and port(source)==port(target))
                except (ValueError,UnicodeError):same_origin=False
                if not same_origin:
                    return await PlainTextResponse('Cross-origin writes are blocked.',status_code=403)(scope,receive,send)
        await self.app(scope,receive,send)
