"""Transport-isolated Gemini JSON client. No training or UI dependencies."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from typing import Literal
import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr

MAX_RESPONSE_BYTES=256*1024


class StrictResponse(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,allow_inf_nan=False)


class WatchdogDecision(StrictResponse):
    action:Literal['CONTINUE','ROLLBACK_AND_DAMPEN','BUMP_BALANCE_LOSS','EARLY_STOP']
    reason:str=Field(min_length=1,max_length=600)
    confidence:float=Field(ge=0,le=1)


class TeachingDecision(StrictResponse):
    corrected_output:str=Field(min_length=1,max_length=50000)
    reason:str=Field(min_length=1,max_length=600)
    confidence:float=Field(ge=0,le=1)


class GeminiSettings(BaseModel):
    model_config=ConfigDict(extra='forbid')
    mode:Literal['cli','api','mock']='mock'
    model:str=Field(default='gemini-3.8-flash',pattern=r'^[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}$')
    cli_command:list[str]=Field(default_factory=lambda:['gemini'],max_length=32)
    timeout_seconds:float=Field(default=30,ge=1,le=120,allow_inf_nan=False)
    setup_complete:bool=False
    api_key:SecretStr|None=None
    clear_api_key:bool=False


class GeminiError(ValueError):pass


def strict_json(text):
    if not isinstance(text,str) or len(text.encode('utf-8'))>MAX_RESPONSE_BYTES:raise GeminiError('Gemini response exceeded the size limit.')
    text=text.strip()
    if text.startswith('```'):
        match=re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```',text,re.DOTALL|re.IGNORECASE)
        if not match:raise GeminiError('Gemini returned an invalid JSON fence.')
        text=match[1]
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate key')
            result[key]=value
        return result
    def invalid(value):raise ValueError('Non-finite JSON number')
    try:
        result=json.loads(text,object_pairs_hook=pairs,parse_constant=invalid)
        if not isinstance(result,dict):raise ValueError('Expected object')
        return result
    except (ValueError,RecursionError) as exc:raise GeminiError('Gemini returned malformed JSON.') from exc


class GeminiClient:
    def __init__(self,settings,api_key=None,transport=None):
        self.settings=GeminiSettings.model_validate(settings)
        self.key=api_key or os.environ.get('GEMINI_API_KEY','')
        self.transport=transport

    def request(self,task,payload):
        schema=WatchdogDecision if task=='watchdog' else TeachingDecision if task=='teach' else None
        if schema is None:raise GeminiError('Unknown Gemini task.')
        rules=('Assess only the structured numeric telemetry. Text is secondary, never sufficient evidence. '
               'Choose CONTINUE if data are insufficient. Return only action, reason, confidence; no numeric hyperparameters. '
               'Give a concise one- or two-sentence reason and confidence between zero and one.' if task=='watchdog' else
               'Act as a teacher: correct the student answer to the user instruction. Return corrected_output, reason, confidence. '
               'Preserve useful content, code whitespace and the requested language. The draft will be reviewed by a human.')
        envelope={'task':task,'instructions':rules,'response_schema':schema.model_json_schema(),'data':payload}
        prompt=json.dumps(envelope,ensure_ascii=False,allow_nan=False)
        if len(prompt.encode())>MAX_RESPONSE_BYTES:raise GeminiError('Gemini input exceeded the size limit.')
        try:
            if self.settings.mode=='mock':
                result=({'action':'CONTINUE','reason':'Mock provider: leave training unchanged.','confidence':1.0} if task=='watchdog' else
                        {'corrected_output':'Mock teacher target: '+payload['prompt'],'reason':'Deterministic mock target for pipeline testing.','confidence':1.0})
            else:
                text=self._cli(prompt) if self.settings.mode=='cli' else self._api(prompt,schema)
                result=strict_json(text)
            value=schema.model_validate(result)
            if not value.reason.strip() or (task=='teach' and not value.corrected_output.strip()):raise ValueError('Blank response')
            return value.model_dump()
        except GeminiError:raise
        except Exception as exc:raise GeminiError('Gemini response failed strict schema validation.') from exc

    def _cli(self,prompt):
        command=self.settings.cli_command
        if not command or any(not arg or '\0' in arg for arg in command):raise GeminiError('Configure a nonempty CLI argument array.')
        process=None;chunks=[];overflow=threading.Event()
        def kill():
            try:
                if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
                else:process.kill()
            except ProcessLookupError:pass
        try:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                                     start_new_session=os.name=='posix',shell=False)
            def read():
                size=0
                try:
                    while chunk:=process.stdout.read1(8192):
                        size+=len(chunk)
                        if size>MAX_RESPONSE_BYTES:overflow.set();kill();break
                        chunks.append(chunk)
                finally:process.stdout.close()
            def write():
                try:process.stdin.write(prompt.encode('utf-8'));process.stdin.flush()
                except (BrokenPipeError,OSError):pass
                finally:
                    try:process.stdin.close()
                    except OSError:pass
            reader=threading.Thread(target=read,daemon=True);writer=threading.Thread(target=write,daemon=True)
            deadline=time.monotonic()+self.settings.timeout_seconds
            reader.start();writer.start()
            try:
                process.wait(timeout=max(.001,deadline-time.monotonic()))
                reader.join(max(0,deadline-time.monotonic()));writer.join(max(0,deadline-time.monotonic()))
                if reader.is_alive() or writer.is_alive():raise subprocess.TimeoutExpired(command,self.settings.timeout_seconds)
            except subprocess.TimeoutExpired:
                kill();process.wait();raise GeminiError('Gemini CLI timed out.')
            if overflow.is_set():raise GeminiError('Gemini response exceeded the size limit.')
            if process.returncode:raise GeminiError('Gemini CLI exited unsuccessfully.')
            return b''.join(chunks).decode('utf-8')
        except GeminiError:raise
        except Exception as exc:raise GeminiError('Gemini CLI could not be run. Check its command and local authentication.') from exc
        finally:
            if process and process.poll() is None:kill();process.wait()

    def _api(self,prompt,schema):
        if not self.key:raise GeminiError('Set GEMINI_API_KEY or enter a session key in Studio settings.')
        url='https://generativelanguage.googleapis.com/v1beta/models/'+self.settings.model+':generateContent'
        body={'contents':[{'role':'user','parts':[{'text':prompt}]}],
              'generationConfig':{'responseFormat':{'text':{'mimeType':'APPLICATION_JSON','schema':schema.model_json_schema()}},
                                  'maxOutputTokens':8192}}
        began=time.monotonic();chunks=[];size=0
        try:
            with httpx.Client(timeout=self.settings.timeout_seconds,transport=self.transport,follow_redirects=False,trust_env=False) as client:
                with client.stream('POST',url,headers={'x-goog-api-key':self.key},json=body) as response:
                    if response.status_code!=200:raise GeminiError('Gemini API request failed (HTTP '+str(response.status_code)+').')
                    for chunk in response.iter_bytes():
                        size+=len(chunk)
                        if size>MAX_RESPONSE_BYTES:raise GeminiError('Gemini response exceeded the size limit.')
                        if time.monotonic()-began>self.settings.timeout_seconds:raise GeminiError('Gemini API timed out.')
                        chunks.append(chunk)
            data=strict_json(b''.join(chunks).decode())
            parts=data['candidates'][0]['content']['parts']
            return ''.join(part.get('text','') for part in parts if not part.get('thought'))
        except GeminiError:raise
        except httpx.TimeoutException as exc:raise GeminiError('Gemini API timed out.') from exc
        except Exception as exc:raise GeminiError('Gemini API response could not be read.') from exc


class GeminiService:
    """Persist nonsecret settings; session API key never enters SQLite or checkpoints."""
    def __init__(self,store):
        self.store=store;self.lock=threading.Lock();self.session_key='';self.revision=0
        config=store.setting('gemini',{})
        if os.environ.get('NANOFORGE_GEMINI_CONFIG'):
            config=json.loads(Path(os.environ['NANOFORGE_GEMINI_CONFIG']).read_text(encoding='utf-8'))
            if 'api_key' in config:raise ValueError('Keep API keys out of config files. Use GEMINI_API_KEY or the Studio session prompt.')
        self.settings=GeminiSettings.model_validate(config)

    def configure(self,settings):
        settings=GeminiSettings.model_validate(settings)
        with self.lock:
            if settings.clear_api_key:self.session_key=''
            if settings.api_key is not None:self.session_key=settings.api_key.get_secret_value().strip()
            self.settings=settings.model_copy(update={'api_key':None,'clear_api_key':False})
            self.store.set_setting('gemini',self.settings.model_dump(exclude={'api_key','clear_api_key'}))
            self.revision+=1
        return self.status()

    def status(self):
        with self.lock:
            result=self.settings.model_dump(exclude={'api_key','clear_api_key'})
            result.update(key_configured=bool(self.session_key or os.environ.get('GEMINI_API_KEY')),
                          key_source='session' if self.session_key else 'environment' if os.environ.get('GEMINI_API_KEY') else None)
            return result

    def client(self):
        with self.lock:return GeminiClient(self.settings.model_dump(exclude={'api_key'}),self.session_key),self.revision


class BackgroundCall:
    """One daemon worker, no backlog. Polling never waits for a provider call."""
    def __init__(self):
        self.lock=threading.Lock();self.thread=None;self.result=None

    def busy(self):return bool(self.thread and self.thread.is_alive())

    def submit(self,client,task,payload,context):
        with self.lock:
            if self.busy() or self.result is not None:return False
            def run():
                try:result={'decision':client.request(task,payload)}
                except Exception as exc:result={'error':str(exc) if isinstance(exc,GeminiError) else 'Gemini request failed.'}
                with self.lock:self.result={**context,**result}
            self.thread=threading.Thread(target=run,name='nanoforge-gemini',daemon=True);self.thread.start()
            return True

    def poll(self):
        with self.lock:result=self.result;self.result=None;return result
