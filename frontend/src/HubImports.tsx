import { useEffect, useRef, useState } from 'react'
import { Download, Globe2, Pause, Search } from 'lucide-react'
import { api } from './api'

type Job = {
  state: string; error?: string; rows?: number; needs_review?: number
  output?: string; bytes_downloaded?: number; bytes_total?: number; max_records?: number
  previous_outputs?: Record<string, unknown>
}
type CatalogSource = {
  id: string; repo: string; label: string; kind: string; size: string; license: string
  note?: string; config?: string; massive?: boolean; streaming: boolean
}
type State = {active: boolean; catalog: CatalogSource[]; jobs: Record<string, Job>}
type Access = 'stream' | 'local'
const mb = (bytes: number) => (bytes / 1_000_000).toFixed(1) + ' MB'
const pageSize = 8
const ready = (job?: Job) => job?.state === 'completed' || job?.state === 'sampled'
const resumable = (job?: Job) => ['paused', 'error', 'blocked'].includes(job?.state || '')
const kindLabel: Record<string, string> = {text: 'Pretraining text', sft: 'Instruction', preference: 'Preference'}

export default function HubImports({onComplete, onMix, mix}: {
  onComplete: () => void
  onMix: (name: string, alternatives?: string[]) => void
  mix: Record<string, number>
}) {
  const [data, setData] = useState<State | null>(null)
  const [pending, setPending] = useState(false), [error, setError] = useState('')
  const [query, setQuery] = useState(''), [kind, setKind] = useState('all')
  const [page, setPage] = useState(0), [expanded, setExpanded] = useState('')
  const [access, setAccess] = useState<Record<string, Access>>({})
  const [selected, setSelected] = useState<string[]>([]), [limit, setLimit] = useState(0)
  const seen = useRef(''), callback = useRef(onComplete), mutating = useRef(false)
  callback.current = onComplete

  useEffect(() => {
    let alive = true
    async function poll() {
      try {
        if (mutating.current) return
        const next = await api<State>('/imports')
        if (!alive || mutating.current) return
        setData(next)
        const completed = JSON.stringify(Object.values(next.jobs).filter(ready).map(job => job.output))
        if (completed !== seen.current) {
          seen.current = completed
          callback.current()
        }
      } catch (reason) {
        if (alive) setError(String(reason))
      }
    }
    poll()
    const timer = setInterval(poll, 1800)
    return () => {alive = false; clearInterval(timer)}
  }, [])

  async function mutate(path: string, body?: unknown) {
    mutating.current = true
    setPending(true)
    setError('')
    try { setData(await api<State>(path, 'POST', body)) }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
    finally { mutating.current = false; setPending(false) }
  }

  const download = (ids: string[], maxRecords: number) => mutate('/imports/start', {
    dataset_ids: ids, max_records: maxRecords,
  })
  const matching = (data?.catalog || []).filter(source =>
    (kind === 'all' || source.kind === kind) &&
    [source.label, source.repo, source.config, source.id].join(' ').toLowerCase().includes(query.trim().toLowerCase()),
  )
  const locked = pending || data?.active
  const queued = selected.filter(id => !ready(data?.jobs[id]) && !resumable(data?.jobs[id]))
  const jobs = Object.entries(data?.jobs || {}).filter(([, job]) => !ready(job))

  return <section className="panel dataset-catalog">
    <div className="panel-title"><div><Globe2 size={18}/><h2>Hugging Face catalog</h2></div></div>
    <div className="catalog-body">
      <div className="catalog-filters">
        <label className="catalog-search"><Search size={16}/><input aria-label="Search datasets" placeholder="Search dataset, repository or subset" value={query} onChange={event => {setQuery(event.target.value); setPage(0)}}/></label>
        <select aria-label="Filter dataset type" value={kind} onChange={event => {setKind(event.target.value); setPage(0)}}>
          <option value="all">All types</option>
          {Object.entries(kindLabel).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
      </div>
      <div className="catalog-toolbar">
        <span className="muted">{matching.length} datasets</span>
        <div className="row">
          {!!queued.length && <><select aria-label="Bulk download size" value={limit} disabled={locked} onChange={event => setLimit(+event.target.value)}>
            <option value={0}>Full training split</option><option value={1000}>Sample 1,000 rows</option><option value={10000}>Sample 10,000 rows</option>
          </select><button className="button secondary small" disabled={locked} onClick={() => download(queued, limit)}>Download selected ({queued.length})</button></>}
          {data?.active && <button className="button secondary small" disabled={pending} onClick={() => mutate('/imports/pause')}><Pause size={14}/>Pause downloads</button>}
        </div>
      </div>
      {error && <p className="inline-error" role="alert">{error}</p>}
      {!!jobs.length && <details className="catalog-jobs" open={data?.active || undefined}>
        <summary>Download activity ({jobs.length})</summary>
        {jobs.map(([id, job]) => <p key={id}>{data?.catalog.find(s => s.id === id)?.label || id} · {job.state}
          {job.state === 'downloading' && job.bytes_downloaded != null ? ' · ' + mb(job.bytes_downloaded) + (job.bytes_total ? ' / ' + mb(job.bytes_total) : '') : ''}
          {job.error ? ' · ' + job.error : ''}
        </p>)}
      </details>}
      <div className="catalog-list">
        {matching.slice(page * pageSize, (page + 1) * pageSize).map(source => {
          const job = data?.jobs[source.id]
          const stream = 'hf:' + source.id
          const cached = [job?.output, ...Object.keys(job?.previous_outputs || {})].filter((name): name is string => !!name)
          const inMix = !!mix[stream] || cached.some(name => !!mix[name])
          const open = expanded === source.id
          const mode = access[source.id] ?? (mix[stream] || source.massive && source.streaming ? 'stream' : 'local')
          return <article className="catalog-item" key={source.id}>
            <div className="catalog-summary">
              <input type="checkbox" aria-label={'Select ' + source.label + ' for download'} checked={queued.includes(source.id)}
                disabled={locked || ready(job) || resumable(job)}
                onChange={event => setSelected(previous => event.target.checked ? [...previous, source.id] : previous.filter(id => id !== source.id))}/>
              <button type="button" className="catalog-toggle" aria-expanded={open} onClick={() => setExpanded(open ? '' : source.id)}>
                <span><strong>{source.label}</strong><small>{source.repo}{source.config ? ' · ' + source.config : ''}</small></span>
                <span className="catalog-kind">{kindLabel[source.kind] || source.kind}</span>
                <span className="catalog-state">{inMix ? 'In mix' : ready(job) ? 'Cached' : job?.state || source.size}</span>
                <span aria-hidden="true">{open ? '−' : '+'}</span>
              </button>
            </div>
            {open && <div className="catalog-detail">
              <p><a href={'https://huggingface.co/datasets/' + source.repo} target="_blank" rel="noreferrer">Source card</a> · {source.license} · {source.size}</p>
              {source.note && <p>{source.note}</p>}
              <label className="field"><span>Access for {source.label}</span>
                <select value={mode} onChange={event => setAccess(previous => ({...previous, [source.id]: event.target.value as Access}))}>
                  <option value="local">Local cache</option><option value="stream" disabled={!source.streaming}>Streaming</option>
                </select>
              </label>
              {mode === 'stream' ? <>
                <small>Reads from Hugging Face during training. Use Magic 8k or GPT-2 BPE. Download a local cache to use instruction or preference pairs.</small>
                <button className="button primary small" disabled={!!mix[stream]} onClick={() => onMix(stream, cached)}>{mix[stream] ? 'Streaming in mix' : 'Add stream to mix'}</button>
              </> : <>
                {!source.streaming && <small>This dataset requires Local cache.</small>}
                {job?.error && <p className="inline-error">{job.error}</p>}
                {!!job?.needs_review && <small>{job.needs_review.toLocaleString()} rows held for review</small>}
                <div className="row catalog-actions">
                  {ready(job) && job?.output && <button className="button primary small" disabled={!!mix[job.output]} onClick={() => onMix(job.output!, [stream, ...cached])}>Add cached file to mix</button>}
                  {job?.state !== 'completed' && <button className="button secondary small" disabled={locked} onClick={() => download([source.id], resumable(job) ? job?.max_records || 0 : 0)}><Download size={14}/>{resumable(job) ? 'Resume download' : 'Download full split'}</button>}
                  {!ready(job) && !resumable(job) && <button className="button secondary small" disabled={locked} onClick={() => download([source.id], 1000)}>Sample 1,000</button>}
                </div>
              </>}
            </div>}
          </article>
        })}
        {!matching.length && <p className="muted">{data ? 'No matching datasets.' : 'Loading datasets…'}</p>}
      </div>
      {matching.length > pageSize && <div className="catalog-pagination">
        <button className="button secondary small" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
        <span>Page {page + 1} of {Math.ceil(matching.length / pageSize)}</span>
        <button className="button secondary small" disabled={(page + 1) * pageSize >= matching.length} onClick={() => setPage(page + 1)}>Next</button>
      </div>}
    </div>
  </section>
}
