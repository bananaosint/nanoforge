import { useEffect, useState } from 'react'
import { api } from './api'
import type { Status } from './types'

type Settings={mode:'api'|'cli'|'mock';model:string;cli_command:string[];timeout_seconds:number;setup_complete:boolean;key_configured:boolean;key_source:string|null}
type Job={id:string;state:string;prompt:string;student_output:string;checkpoint:string|null;error:string|null;feedback_id:number|null;approved_output?:string;provider:{mode:string;model:string};draft:{corrected_output:string;reason:string;confidence:number}|null}
const message=(error:unknown)=>error instanceof Error?error.message:String(error)

export function GeminiSetup(){
 const [settings,setSettings]=useState<Settings|null>(null),[open,setOpen]=useState(false),[key,setKey]=useState(''),[command,setCommand]=useState('["gemini"]'),[busy,setBusy]=useState(false),[notice,setNotice]=useState('')
 useEffect(()=>{api<Settings>('/gemini/settings').then(value=>{setSettings({...value,mode:!value.setup_complete?'api':value.mode});setCommand(JSON.stringify(value.cli_command));setOpen(!value.setup_complete||(value.mode==='api'&&!value.key_configured))}).catch(()=>{})},[])
 async function save(skip=false){
  if(!settings)return
  setBusy(true);setNotice('')
  try{
   const {key_configured,key_source,...config}=settings
   const value=await api<Settings>('/gemini/settings','PUT',{...config,mode:skip?'mock':settings.mode,cli_command:skip?['gemini']:JSON.parse(command),setup_complete:true,...(!skip&&settings.mode==='api'&&key?{api_key:key}:{})})
   setSettings(value);setKey('');setOpen(false)
  }catch(error){setNotice(message(error))}finally{setBusy(false)}
 }
 return <><button className="button secondary small" onClick={()=>setOpen(true)}>Gemini settings</button>{open&&<div className="modal-backdrop"><section className="panel modal gemini-dialog" role="dialog" aria-modal="true" aria-labelledby="gemini-title"><div className="panel-title"><h2 id="gemini-title">Set up Gemini assistance</h2><button className="text-button" onClick={()=>{setOpen(false);setKey('')}}>Close</button></div><div className="modal-content gemini-form">
  <p>Connect a teacher and checkpoint watchdog. API mode sends enabled watchdog metrics and requested teaching examples to Google. CLI uses your configured local command; mock stays local.</p>
  {settings?<><label className="field"><span>Gemini provider</span><select value={settings.mode} onChange={e=>setSettings({...settings,mode:e.target.value as Settings['mode']})}><option value="api">Gemini API</option><option value="cli">Local CLI</option><option value="mock">Mock · no external calls</option></select></label>
  {settings.mode==='api'&&<><label className="field"><span>Gemini API key</span><input type="password" autoComplete="off" value={key} onChange={e=>setKey(e.target.value)} placeholder={settings.key_configured?'Configured via '+settings.key_source:'Enter Gemini API key'}/></label><small>Kept in backend memory for this session. No browser storage, database or checkpoint copy. You can also set GEMINI_API_KEY in the backend environment.</small>{settings.key_source==='session'&&<button className="text-button" disabled={busy} onClick={async()=>{try{const {key_configured,key_source,...config}=settings;setSettings(await api<Settings>('/gemini/settings','PUT',{...config,clear_api_key:true}));setKey('')}catch(error){setNotice(message(error))}}}>Forget session key</button>}</>}
  {settings.mode==='cli'&&<label className="field"><span>CLI command · JSON argument array</span><input value={command} onChange={e=>setCommand(e.target.value)}/><small>Prompt envelope arrives on stdin. Output must be the requested JSON object. No shell expansion; local CLI authentication applies.</small></label>}
  <label className="field"><span>Gemini model</span><input value={settings.model} onChange={e=>setSettings({...settings,model:e.target.value})}/></label>
  <label className="field"><span>Request timeout (seconds)</span><input type="number" min="1" max="120" value={settings.timeout_seconds} onChange={e=>setSettings({...settings,timeout_seconds:+e.target.value})}/></label>
  <div className="row"><button className="button primary" disabled={busy||(settings.mode==='api'&&!key&&!settings.key_configured)} onClick={()=>save()}>Save Gemini settings</button><button className="button secondary" disabled={busy} onClick={()=>save(true)}>Use local mock instead</button></div>
  <button className="text-button" disabled={busy} onClick={async()=>{setBusy(true);try{await api('/gemini/test','POST');setNotice('Saved provider returned valid JSON.')}catch(error){setNotice(message(error))}finally{setBusy(false)}}}>Test saved provider (sends synthetic metrics)</button></>:<p>Connecting to backend…</p>}
  {notice&&<p role="status">{notice}</p>}
 </div></section></div>}</>
}

export function WatchdogPanel({status,enabled,setEnabled,interval,setInterval,disabled}:{status:Status;enabled:boolean;setEnabled:(value:boolean)=>void;interval:number;setInterval:(value:number)=>void;disabled:boolean}){
 const watchdog=status.watchdog,last=watchdog?.events.at(-1),metric=status.metrics.at(-1)
 return <section className="panel gemini-panel"><div className="panel-title"><h2>Checkpoint watchdog · Circuit Breaker</h2><span className="tag">{status.mode!=='pretrain'?'INACTIVE DURING '+status.mode.toUpperCase():watchdog?.pending?'GEMINI REQUEST PENDING':'PRETRAINING'}</span></div><div className="gemini-form">
  <label className="row"><input type="checkbox" checked={enabled} disabled={disabled} onChange={e=>setEnabled(e.target.checked)}/>Enable Gemini watchdog for this run</label>
  <label className="field"><span>Watchdog interval (steps)</span><select value={interval} disabled={disabled} onChange={e=>setInterval(+e.target.value)}>{Array.from(new Set([250,500,1000,interval])).sort((a,b)=>a-b).map(value=><option key={value} value={value}>{value}</option>)}</select></label>
  <small>Uses the saved Gemini provider. Runs asynchronously; numeric guards gate interventions. Rollback restores the best validation checkpoint. Default policy: LR ×0.5, balance coefficient ×2 (maximum 1), minimum confidence 0.8, and at most three rollbacks per run. Policy values loaded from training JSON/API are preserved.</small>
  <div className="row gemini-telemetry"><span>Current LR: {metric?.learning_rate?.toExponential(2)??'—'}</span><span>Balance coefficient: {status.model_config?.architecture==='moe'?status.model_config.router_aux_loss:'—'}</span><span>Best validation: {watchdog?.best_val_loss?.toFixed(4)??'—'}</span><span>Grad norm: {metric?.grad_norm?.toFixed(3)??'—'}</span><span>Balance loss: {metric?.balance_loss?.toFixed(4)??'—'}</span><span>Z loss: {metric?.z_loss?.toFixed(4)??'—'}</span></div>
  {metric?.expert_load_pct?.length? <details><summary>Expert routing · % of tokens per expert</summary>{metric.expert_load_pct.map((layer,index)=><p key={index}>Layer {index+1}: {layer.map((value,expert)=>`E${expert+1} ${value.toFixed(1)}%`).join(' · ')}</p>)}<small>Top-k routing totals 100 × k percent per layer.</small></details>:null}
  {last&&<p role="status">{last.action}: {last.reason}{last.applied===false?' · '+last.guard:''}</p>}
  {watchdog?.events.length? <details><summary>Watchdog history ({watchdog.events.length})</summary>{watchdog.events.map((event,index)=><p key={index}>Step {event.step} · {event.action} · {event.reason}{event.guard?' · '+event.guard:''}</p>)}</details>:null}
 </div></section>
}

export function GeminiTeach({prompt,studentOutput,checkpoint,onSaved}:{prompt:string;studentOutput:string;checkpoint:string|null;onSaved:()=>Promise<unknown>}){
 const [jobs,setJobs]=useState<Job[]>([]),[job,setJob]=useState<Job|null>(null),[target,setTarget]=useState(''),[busy,setBusy]=useState(false),[error,setError]=useState('')
 useEffect(()=>{api<Job[]>('/teaching').then(values=>{setJobs(values);setJob(values[0]||null)}).catch(()=>{})},[])
 useEffect(()=>{if(job?.state==='draft')setTarget(job.draft?.corrected_output||'')},[job?.id,job?.state])
 useEffect(()=>{
  if(job?.state!=='running')return
  let live=true
  const timer=setInterval(()=>api<Job>('/teaching/'+job.id).then(value=>{if(live)setJob(value)}).catch(reason=>{if(live)setError(message(reason))}),1000)
  return()=>{live=false;clearInterval(timer)}
 },[job?.id,job?.state])
 async function request(){setBusy(true);setError('');try{const value=await api<Job>('/teaching','POST',{prompt,student_output:studentOutput,checkpoint});setJob(value);setJobs(old=>[value,...old])}catch(reason){setError(message(reason))}finally{setBusy(false)}}
 async function approve(){if(!job)return;setBusy(true);setError('');try{const value=await api<{job:Job}>('/teaching/'+job.id+'/approve','POST',{corrected_output:target});setJob(value.job);await onSaved()}catch(reason){setError(message(reason))}finally{setBusy(false)}}
 return <section className="panel gemini-panel"><div className="panel-title"><h2>Gemini · Edit & Teach</h2><span className="tag">HUMAN REVIEW</span></div><div className="gemini-form">
  <p>Ask the saved provider to improve the current prompt’s student response. Review/edit its draft, then approve it into the existing SFT library. Training uses the Studio’s completion mask and prompt template.</p>
  <button className="button secondary" disabled={busy||job?.state==='running'||!prompt.trim()} onClick={request}>{job?.state==='running'?'Teacher is working…':'Ask Gemini to edit this response'}</button>
  <small>API mode sends this prompt and student completion to Google only when you request a draft. Existing manual editing remains available.</small>
  {jobs.length>0&&<label className="field"><span>Teacher drafts and history</span><select value={job?.id||''} onChange={e=>{const selected=jobs.find(value=>value.id===e.target.value);if(selected)api<Job>('/teaching/'+selected.id).then(setJob).catch(reason=>setError(message(reason)))}}>{jobs.map(value=><option key={value.id} value={value.id}>{value.prompt.slice(0,60)} · {value.id.slice(0,6)}</option>)}</select></label>}
  {job&&<><small>Request: {job.provider.mode} · {job.provider.model} · {job.state}</small><pre className="teacher-prompt">{job.prompt}</pre>{job.error&&<p role="alert">{job.error}</p>}
   {job.draft&&<><p>{job.draft.reason} · confidence {(job.draft.confidence*100).toFixed(0)}%</p><label className="field"><span>Teacher target · edit before approval</span><textarea className="target-input" value={job.state==='approved'?(job.approved_output??job.draft.corrected_output):target} readOnly={job.state==='approved'} onChange={e=>setTarget(e.target.value)}/></label><button className="button primary" disabled={busy||job.state!=='draft'||!target.trim()} onClick={approve}>{job.state==='approved'?'Added to SFT library':'Approve & add to SFT'}</button><small>After approval, use “Train on feedback → Start SFT” below to distill approved targets into the active model.</small></>}
  </>}{error&&<p role="alert">{error}</p>}
 </div></section>
}
