import { useEffect, useRef, useState } from 'react'
import { Check, Download, Globe2, Loader2, Pause, RefreshCw } from 'lucide-react'
import { api } from './api'

type Job={state:string;error?:string;rows?:number;source_records?:number;needs_review?:number;output?:string;bytes_downloaded?:number;bytes_total?:number;current_record?:number;max_records?:number}
type Source={id:string;repo:string;label:string;kind:string;size:string;license:string;note?:string;config?:string;massive?:boolean;needs_access?:boolean}
type State={active:boolean;catalog:Source[];jobs:Record<string,Job>}
const mb=(n:number)=>(n/1_000_000).toFixed(1)+' MB'

export default function HubImports({onComplete}:{onComplete:(names:string[])=>void}){
 const [data,setData]=useState<State|null>(null),[pending,setPending]=useState(false),[error,setError]=useState('')
 const [selected,setSelected]=useState<string[]>([]),[limit,setLimit]=useState(0)
 const seen=useRef(new Set<string>()),initialized=useRef(false),callback=useRef(onComplete)
 callback.current=onComplete
 useEffect(()=>{
  let alive=true
  async function poll(){
   try{
    const next=await api<State>('/imports')
    if(!alive)return
    setData(next)
    if(!initialized.current){setSelected(next.catalog.filter(s=>!s.massive&&!s.needs_access).map(s=>s.id));initialized.current=true}
    const completed=new Set(Object.values(next.jobs).filter(j=>['completed','sampled'].includes(j.state)).map(j=>j.output))
    for(const name of seen.current){if(!completed.has(name))seen.current.delete(name)}
    const ready=Object.values(next.jobs).filter(j=>['completed','sampled'].includes(j.state)&&j.output&&!seen.current.has(j.output))
    if(ready.length){ready.forEach(j=>seen.current.add(j.output!));callback.current(ready.map(j=>j.output!))}
   }catch(e){if(alive)setError(String(e))}
  }
  poll();const timer=setInterval(poll,1800)
  return()=>{alive=false;clearInterval(timer)}
 },[])
 async function start(ids:string[],maxRecords=0){
  setPending(true);setError('')
  try{setData(await api<State>('/imports/start','POST',{dataset_ids:ids,max_records:maxRecords}))}
  catch(e){setError(e instanceof Error?e.message:String(e))}
  finally{setPending(false)}
 }
 async function pause(){setPending(true);try{setData(await api<State>('/imports/pause','POST'))}catch(e){setError(String(e))}finally{setPending(false)}}
 return <section className="panel hub-panel">
  <div className="panel-title"><div><Globe2 size={18}/><h2>Direct dataset imports</h2><span className="muted">{data?.catalog.length??'…'} sources · no upload limit</span></div>
   <div className="row hub-actions"><select aria-label="Import record limit" value={limit} disabled={data?.active||pending} onChange={e=>setLimit(+e.target.value)}><option value={0}>Full training split</option><option value={1000}>1,000 source-row sample</option><option value={10000}>10,000 source-row sample</option></select>
   {data?.active?<button className="button secondary small" disabled={pending} onClick={pause}><Pause size={14}/>Pause imports</button>:<button className="button primary small" disabled={!data||pending||!selected.length} onClick={()=>start(selected,limit)}><Download size={14}/>Import selected ({selected.length})</button>}</div>
  </div>
  <div className="hub-intro"><p>Select sources, then download full training splits or explicit samples directly from Hugging Face. Assign completed sources a weight in the mixer below. A completed sample can later be upgraded to its full corpus.</p><p>Large corpora start unselected: Cosmopedia v2 (~122 GB), Wikipedia (~12 GB), FineWeb-Edu 10BT (~29 GB), and FineMath 4+ (~18 GB), all compressed. Full imports process one shard at a time; normalized data and token caches need additional disk space. Chinese R1 remains skipped.</p></div>
  {error&&<div className="inline-error">{error}</div>}
  <div className="hub-grid">{data?.catalog.map(item=>{
   const job=data.jobs[item.id],ready=job?.state==='completed',sampled=job?.state==='sampled'
   const running=job&&['queued','downloading','converting','translating','finalizing'].includes(job.state)
   const interrupted=job&&['paused','error','blocked'].includes(job.state)
   const locked=pending||data.active
   return <article className={'hub-source '+(selected.includes(item.id)?'chosen':'')} key={item.id}>
    <div className="hub-source-title"><label><input type="checkbox" aria-label={'Select '+item.label} checked={selected.includes(item.id)} disabled={locked} onChange={e=>setSelected(old=>e.target.checked?[...old,item.id]:old.filter(id=>id!==item.id))}/><strong>{item.label}</strong></label><span className="tag">{item.kind}</span></div>
    <a href={'https://huggingface.co/datasets/'+item.repo} target="_blank" rel="noreferrer" className="hub-repo">{item.repo}</a>
    <div className="hub-meta">{item.size} · {item.license}{item.config?' · '+item.config:''}</div>
    {item.note&&<p className="hub-note">{item.note}</p>}
    {job&&<div className={'hub-job '+(['error','blocked'].includes(job.state)?'error':'')}>
     <span>{job.state}{sampled?' · '+job.max_records?.toLocaleString()+' source-row limit':''}</span>
     {job.state==='downloading'&&job.bytes_downloaded!==undefined?<small>Current file: {mb(job.bytes_downloaded)}{job.bytes_total?' / '+mb(job.bytes_total):''}</small>:<small>{job.output?.endsWith('.txt')?'Full text corpus ready':(job.rows||0).toLocaleString()+' training records'}{job.source_records?' · '+job.source_records.toLocaleString()+' source rows':''}</small>}
     {!!job.needs_review&&<small>{job.needs_review.toLocaleString()} held for review</small>}{job.error&&<p>{job.error}</p>}
     {job.bytes_total&&job.state==='downloading'?<div className="progress"><i style={{width:Math.min(100,(job.bytes_downloaded||0)/job.bytes_total*100)+'%'}}/></div>:null}
    </div>}
    <div className="row hub-card-actions"><button className={'button small '+(ready?'secondary':'primary')} disabled={locked||ready} onClick={()=>start([item.id],interrupted?job.max_records||0:0)}>{ready?<Check size={14}/>:running?<Loader2 size={14} className="spin"/>:interrupted?<RefreshCw size={14}/>:<Download size={14}/>} {ready?'Ready in mixer':running?'Working…':interrupted?'Resume':sampled?'Import full':'Import full'}</button>
    <button className="button secondary small" disabled={locked||ready||sampled} onClick={()=>start([item.id],1000)}>Sample 1,000</button></div>
   </article>
  })}</div>
  <div className="chart-footer"><span>Token caches use uint16 for vocabularies up to 65,536 IDs; larger vocabularies use uint32.</span><span>Original train splits · explicit configuration selection</span></div>
 </section>
}
