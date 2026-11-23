import { useEffect, useState } from 'react'
import { api } from './api'
import type { ModelConfig, Source } from './types'

type Preview = {model_config:ModelConfig;parameters:number;weights_bytes:number;training_state_bytes:number}
type Props = {
  custom:boolean; dimensions:ModelConfig; tokenizer:string; sources:Source[]; revision:string;
  disabled:boolean; onChange:(model:ModelConfig)=>void; onValid:(valid:boolean)=>void
}
const fields = [
  ['n_layer','Layers',1], ['n_head','Heads',1], ['n_embd','Embedding width',1],
  ['block_size','Context window',8], ['dropout','Dropout',0],
] as const
const gib = (bytes:number)=>(bytes/1024**3).toFixed(2)+' GiB'

export default function ModelSizing({custom,dimensions,tokenizer,sources,revision,disabled,onChange,onValid}:Props){
  const [preview,setPreview]=useState<Preview|null>(null),[error,setError]=useState('')
  const [budget,setBudget]=useState('100'),[fitting,setFitting]=useState(false),[fitError,setFitError]=useState('')
  const request=JSON.stringify({preset:'Custom',tokenizer,model:dimensions,datasets:sources})
  useEffect(()=>{
    let current=true
    setPreview(null);setError('');onValid(false)
    const timer=setTimeout(()=>{
      api<Preview>('/models/preview','POST',JSON.parse(request)).then(result=>{
        if(current){setPreview(result);onValid(true)}
      }).catch(e=>{if(current)setError(e instanceof Error?e.message:String(e))})
    },350)
    return()=>{current=false;clearTimeout(timer)}
  },[request,revision,onValid])
  const fit=async()=>{
    const target=Math.floor(Number(budget)*1_000_000)
    if(!Number.isFinite(target)||target<1||target>=1_000_000_000){setFitError('Enter a budget from 0.000001 to 999.999999 million.');return}
    setFitting(true);setFitError('')
    try{
      const {n_embd,...fixed}=dimensions
      const result=await api<Preview>('/models/preview','POST',{preset:'Custom',tokenizer,model:fixed,datasets:sources,target_parameters:target})
      const {vocab_size,...model}=result.model_config
      onChange(model)
    }catch(e){setFitError(e instanceof Error?e.message:String(e))}finally{setFitting(false)}
  }
  return <div className="model-sizing">
    {custom&&<>
      <div className="field-pair">{fields.map(([key,label,min])=><label className="field" key={key}><span>{label}</span><input aria-label={label} disabled={disabled||fitting} type="number" min={min} max={key==='dropout'?0.8:999999999} step={key==='dropout'?0.05:1} value={dimensions[key]} onChange={e=>onChange({...dimensions,[key]:+e.target.value})}/></label>)}</div>
      <label className="field"><span>Parameter budget (millions)</span><input aria-label="Parameter budget in millions" type="number" min="0.000001" max="999.999999" step="1" value={budget} disabled={disabled||fitting} onChange={e=>setBudget(e.target.value)}/></label>
      <button className="button secondary small" disabled={disabled||fitting} onClick={fit}>{fitting?'Sizing…':'Fit width to budget'}</button>
      <p className="sizing-note">Keeps layers, heads and context fixed. Finds the largest width that fits your budget; whole dimensions may leave a small gap.</p>
      {fitError&&<p role="alert" className="sizing-error">{fitError}</p>}
    </>}
    <div className="parameter-preview" aria-live="polite">
      {error?<p className="sizing-error">{error}</p>:preview?<>
        <strong>{preview.parameters.toLocaleString('en-US')} parameters</strong>
        <span>{preview.model_config.vocab_size?.toLocaleString('en-US')} vocabulary · limit &lt;1 billion</span>
        <span>FP32 weights {gib(preview.weights_bytes)} · training state ~{gib(preview.training_state_bytes)}</span>
        <small>Training also needs activations and working memory. A model below the limit may still exceed your GPU’s memory.</small>
      </>:<span>Counting parameters…</span>}
    </div>
  </div>
}
