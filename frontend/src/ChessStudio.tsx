import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { Activity, Check, ChevronDown, CircleAlert, Cpu, Crosshair, FileUp, Loader2, Pause, Play, Settings2, Square, X } from 'lucide-react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from './api'
import type { Checkpoint, Status } from './types'

const startingFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
const activeStates = ['running', 'paused', 'stopping']
const message = (error: unknown) => error instanceof Error ? error.message : String(error)
const series = [
  { key: 'composite_chess_accuracy', label: 'Composite chess accuracy', color: '#adf57c' },
  { key: 'piece_hallucination_rate', label: 'Piece hallucination rate', color: '#fb9887' },
  { key: 'move_legality_rate', label: 'Move legality rate', color: '#a99bfa' },
  { key: 'tactical_fact_accuracy', label: 'Tactical fact accuracy', color: '#78d6dd' },
  { key: 'theme_f1', label: 'Theme F1', color: '#e8bd79' },
] as const
const categoryLabels: Record<string, string> = {
  grounding: 'Piece grounding', legality: 'Move legality', tactical: 'Tactical facts',
  stockfish: 'Stockfish verdict', theme: 'Tactical themes', coverage: 'Claim coverage',
}
const defaultRun = {
  dataset: 'starter', preset: 'Nano', checkpoint: '', max_steps: 250, batch_size: 4,
  learning_rate: 0.0003, eval_interval: 50, benchmark_size: 50, use_stockfish: false,
}
type RunOptions = typeof defaultRun
type Dataset = { id: string; label: string; hub?: boolean; streaming?: boolean; ready: boolean }
type Settings = { stockfish_path: string; resolved_path: string | null; stockfish_available: boolean; source: string }
type Claim = { category: string; claim: string; start: number; end: number; passed: boolean; reason: string | null }
type Verification = {
  passed: boolean; checks: Claim[]; errors: Claim[];
  recommendations: { move: string; fen: string; centipawn_loss: number | null }[];
}
type Score = { cp: number | null; mate: number | null; mate_color: string | null }
type Context = {
  fen: string; after_fen: string; side_to_move: string; pv: string[];
  move: { uci: string; san: string; origin: string; destination: string };
  eval_before?: Score; eval_after?: Score;
  multipv?: { rank: number; score: Score; pv: string[] }[];
}
type Metrics = Record<string, number | null> & { sample_count: number; passed_samples: number }
type Summary = {
  checkpoint: string; run_id: string | null; step: number; benchmark_id: string; created_at: string; source: 'automatic' | 'manual';
  protocol: { size: number; engine: string | null }; metrics: Metrics;
}
type Sample = { fen: string; move: string; explanation: string; reference: string; verification: Verification }
type Report = Summary & { samples: Sample[] }
type Position = {
  fen: string; context: Context | null;
  legal_moves: { uci: string; san: string }[]; pgn_moves: { ply: number; uci: string; san: string }[];
}
type TestResult = {
  checkpoint: string; context: Context; explanation: string; prompt: string;
  stockfish_available: boolean; verification: Verification;
}
type PositionInput = { fen: string; move?: string } | { pgn: string; ply: number }
export type ChessTab = 'chess-train' | 'chess-accuracy' | 'chess-sandbox'

function Field({ label, children }: { label: string; children: ReactNode }) {
  return <label className="field"><span>{label}</span>{children}</label>
}

function Section({ title, icon, action, children }: { title: string; icon: ReactNode; action?: ReactNode; children: ReactNode }) {
  return <section className="panel chess-panel">
    <div className="panel-title"><div>{icon}<h2>{title}</h2></div>{action}</div>
    {children}
  </section>
}

function CheckpointPicker({ checkpoints, value, onChange, placeholder = 'Select a checkpoint' }: {
  checkpoints: Checkpoint[]; value: string; onChange: (name: string) => void; placeholder?: string;
}) {
  return <select value={value} onChange={e => onChange(e.target.value)}>
    <option value="">{placeholder}</option>
    {checkpoints.map(c => <option key={c.name} value={c.name}>Step {c.step} · {c.name.replace(/\.pt$/, '')}</option>)}
  </select>
}

export default function ChessStudio({ tab, status, onStatus }: { tab: ChessTab; status: Status; onStatus: (s: Status) => void }) {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [settings, setSettings] = useState<Settings | null>(null)
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [run, setRun] = useState(defaultRun)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState('')
  const [sandboxCheckpoint, setSandboxCheckpoint] = useState('')
  const chessActive = status.mode === 'chess'
  const training = activeStates.includes(status.state)
  const refreshCheckpoints = useCallback(async () => setCheckpoints(await api<Checkpoint[]>('/chess/checkpoints')), [])

  useEffect(() => {
    Promise.all([
      api<Dataset[]>('/chess/datasets').then(setDatasets),
      api<Settings>('/chess/settings').then(setSettings),
    ]).catch(error => setError(message(error)))
  }, [])
  useEffect(() => {
    refreshCheckpoints().catch(error => setError(message(error)))
  }, [refreshCheckpoints, status.checkpoint, status.state])
  const selectedSandboxCheckpoint = checkpoints.some(item => item.name === sandboxCheckpoint)
    ? sandboxCheckpoint : checkpoints[0]?.name ?? ''

  const perform = async (label: string, action: () => Promise<void>) => {
    setError('')
    setNotice('')
    setBusy(label)
    try { await action() }
    catch (error) { setError(message(error)) }
    finally { setBusy('') }
  }
  const launch = () => perform('train', async () => {
    const { checkpoint, ...options } = run
    onStatus(await api<Status>('/chess/runs', 'POST', { ...options, ...(checkpoint ? { checkpoint } : {}) }))
  })
  const control = (action: string) => perform(action, async () => onStatus(await api<Status>('/chess/training/' + action, 'POST')))
  const pageTitle = tab === 'chess-train' ? 'Chess Training' : tab === 'chess-accuracy' ? 'Chess Accuracy' : 'Position Sandbox'
  const pageDescription = tab === 'chess-train' ? 'Train a small model to explain moves from verified position facts.'
    : tab === 'chess-accuracy' ? 'Compare checkpoint evaluations on a fixed set of held-out positions.'
      : 'Test an explanation and inspect each claim against the board.'

  return <main className="chess-workspace">
    <div className="page-heading">
      <div><h1>{pageTitle}</h1><p>{pageDescription}</p></div>
      {settings && <span className={settings.stockfish_available ? 'chess-pill good' : 'chess-pill'}><Cpu size={14}/>
        {settings.stockfish_available ? 'Stockfish available' : 'Stockfish unavailable'}
      </span>}
    </div>
    {error && <p className="inline-error chess-notice" role="alert">{error}</p>}
    {notice && <p className="chess-notice chess-success" role="status"><Check size={15}/>{notice}</p>}
    {tab === 'chess-train' && <TrainingView
      run={run} setRun={setRun} datasets={datasets} checkpoints={checkpoints} settings={settings}
      status={status} chessActive={chessActive} training={training} busy={busy} launch={launch} control={control}
      savePath={stockfish_path => perform('settings', async () => {
        setSettings(await api<Settings>('/chess/settings', 'PUT', { stockfish_path }))
        setNotice('Stockfish setting saved.')
      })}
      importDataset={preset => perform('import', async () => {
        const result = await api<{ examples: number }>('/chess/datasets/import', 'POST', { preset })
        setDatasets(await api<Dataset[]>('/chess/datasets'))
        setNotice('Imported ' + result.examples.toLocaleString() + ' chess examples.')
      })}
      upload={file => perform('upload', async () => {
        const data = new FormData()
        data.append('file', file)
        const result = await api<{ id: string; examples: number }>('/chess/datasets/upload', 'POST', data)
        setDatasets(await api<Dataset[]>('/chess/datasets'))
        setRun(previous => ({ ...previous, dataset: result.id }))
        setNotice('Uploaded ' + result.examples.toLocaleString() + ' chess examples.')
      })}
    />}
    {tab === 'chess-accuracy' && <AccuracyView checkpoints={checkpoints} status={status} onError={setError} onImported={refreshCheckpoints}/>}
    {tab === 'chess-sandbox' && <Sandbox checkpoints={checkpoints} checkpoint={selectedSandboxCheckpoint} setCheckpoint={setSandboxCheckpoint}
      engineAvailable={settings?.stockfish_available ?? false}/>}
  </main>
}

function TrainingView({ run, setRun, datasets, checkpoints, settings, status, chessActive, training, busy,
  launch, control, savePath, importDataset, upload }: {
  run: RunOptions; setRun: (next: RunOptions | ((old: RunOptions) => RunOptions)) => void; datasets: Dataset[];
  checkpoints: Checkpoint[]; settings: Settings | null; status: Status; chessActive: boolean; training: boolean; busy: string;
  launch: () => void; control: (action: string) => void; savePath: (path: string) => void;
  importDataset: (preset: string) => void; upload: (file: File) => void;
}) {
  const uploadRef = useRef<HTMLInputElement>(null)
  const [path, setPath] = useState(settings?.stockfish_path ?? '')
  const [preview, setPreview] = useState<{ prompt: string; target: string } | null>(null)
  const [previewError, setPreviewError] = useState('')
  const selected = datasets.find(item => item.id === run.dataset)
  const last = status.mode === 'chess' ? status.metrics.at(-1) : undefined
  const accuracy = status.mode === 'chess'
    ? [...status.metrics].reverse().find(item => item.chess_accuracy)?.chess_accuracy?.composite_chess_accuracy
    : null
  useEffect(() => setPath(settings?.stockfish_path ?? ''), [settings?.stockfish_path])
  useEffect(() => {
    api<{ prompt: string; target: string }>('/chess/prompt-preview').then(setPreview).catch(error => setPreviewError(message(error)))
  }, [])

  const update = <K extends keyof RunOptions>(key: K, value: RunOptions[K]) => setRun({ ...run, [key]: value })
  const numberField = (key: 'max_steps' | 'batch_size' | 'learning_rate' | 'eval_interval' | 'benchmark_size',
    label: string, min: number, max: number, step: number | 'any' = 1) => <Field label={label}>
      <input type="number" required value={run[key] || ''} min={min} max={max} step={step}
        onChange={e => update(key, +e.target.value)}/>
    </Field>

  return <div className="chess-training-grid">
    <Section title="Run configuration" icon={<Settings2 size={17}/>}>
      <form className="chess-panel-body" onSubmit={e => { e.preventDefault(); launch() }}>
        <fieldset disabled={training || !!busy} className="chess-fields">
          <Field label="Chess dataset">
            <select value={run.dataset} onChange={e => update('dataset', e.target.value)}>
              {datasets.map(item => <option key={item.id} value={item.id}>{item.label}{item.streaming ? ' · Streaming' : !item.ready ? ' · import required' : ''}</option>)}
            </select>
          </Field>
          {selected?.streaming && <small>Reads chess examples from Hugging Face during training.</small>}
          <div className="chess-actions">
            {selected?.hub && <button className="button secondary small" type="button" onClick={() => importDataset(selected.id)}>
              {busy === 'import' ? <Loader2 className="spin" size={14}/> : <FileUp size={14}/>} {selected.ready ? 'Import again' : 'Import 2,000 examples'}
            </button>}
            <input ref={uploadRef} type="file" accept=".pgn,.jsonl" hidden onChange={e => {
              const file = e.target.files?.[0]
              if (file) upload(file)
              e.target.value = ''
            }}/>
            <button className="button secondary small" type="button" onClick={() => uploadRef.current?.click()}>
              {busy === 'upload' ? <Loader2 className="spin" size={14}/> : <FileUp size={14}/>} Upload JSONL or PGN
            </button>
          </div>
          <div className="chess-field-grid">
            <Field label="Chess model">
              <select disabled={!!run.checkpoint} value={run.preset} onChange={e => update('preset', e.target.value)}>
                <option>Nano</option><option>Micro</option><option>Mini</option>
              </select>
            </Field>
            {numberField('max_steps', 'Training steps', 1, 1000000)}
            {numberField('batch_size', 'Batch size', 1, 64)}
            {numberField('learning_rate', 'Learning rate', 0.000001, 0.1, 'any')}
            {numberField('eval_interval', 'Evaluate every N steps', 1, 10000)}
            {numberField('benchmark_size', 'Benchmark positions', 1, 200)}
          </div>
          <Field label="Starting weights"><CheckpointPicker checkpoints={checkpoints} value={run.checkpoint}
            onChange={checkpoint => update('checkpoint', checkpoint)} placeholder="Train from scratch"/></Field>
          <label className="chess-checkbox">
            <input type="checkbox" checked={run.use_stockfish} onChange={e => update('use_stockfish', e.target.checked)}/>
            Use Stockfish during training and evaluation
          </label>
        </fieldset>
        <div className="chess-actions">
          <button type="submit" className="button primary" disabled={training || !!busy || !selected?.ready}>
            {busy === 'train' ? <Loader2 size={15} className="spin"/> : <Play size={15}/>}
            {run.checkpoint ? 'Continue training' : 'Start chess training'}
          </button>
          {training && chessActive && <>
            <button type="button" className="button secondary" disabled={!!busy || status.state === 'stopping'}
              onClick={() => control(status.state === 'paused' ? 'resume' : 'pause')}>
              {status.state === 'paused' ? <Play size={14}/> : <Pause size={14}/>}
              {status.state === 'paused' ? 'Resume' : 'Pause'}
            </button>
            <button type="button" className="button secondary" disabled={!!busy || status.state === 'stopping'} onClick={() => control('stop')}>
              <Square size={14}/>Stop
            </button>
          </>}
        </div>
        {training && !chessActive && <p className="muted" role="status">Text training is active.</p>}
      </form>
    </Section>

    <div className="chess-stack">
      {chessActive && status.run_id && <Section title="Current chess run" icon={<Activity size={17}/>}
        action={<span className={'state-badge ' + status.state}>{status.state}</span>}>
        <div className="chess-panel-body">
          <div className="chess-run-stats">
            <div><span>Step</span><strong>{status.step} <small>/ {status.target_step ?? status.step}</small></strong></div>
            <div><span>Training loss</span><strong>{last?.train_loss.toFixed(3) ?? '—'}</strong></div>
            <div><span>Composite accuracy</span><strong>{accuracy == null ? '—' : accuracy.toFixed(1) + '%'}</strong></div>
          </div>
          <progress aria-label="Chess training progress" value={status.step} max={status.target_step || status.step || 1}/>
          {status.state === 'running' && status.step === 0 && selected?.streaming && <p className="muted" role="status">Opening the chess stream…</p>}
          {status.chess_evaluating_step != null && <p className="muted" role="status">Evaluating checkpoint at step {status.chess_evaluating_step}…</p>}
          {status.error && <p className="inline-error" role="alert">{status.error}</p>}
        </div>
      </Section>}
      <Section title="Prompt preview" icon={<Crosshair size={17}/>}>
        <div className="chess-panel-body">
          {previewError ? <p className="inline-error">{previewError}</p> : preview
            ? <><pre className="chess-code">{preview.prompt.split('\n').map((line, index) => {
                const marker = line.match(/^<[^>]+>/)?.[0]
                return <span key={index}>{marker && <strong>{marker}</strong>}{marker ? line.slice(marker.length) : line}{'\n'}</span>
              })}</pre><div className="chess-target"><small>Target explanation</small><p>{preview.target}</p></div></>
            : <p className="muted">Loading prompt…</p>}
        </div>
      </Section>
      <Section title="Stockfish settings" icon={<Cpu size={17}/>}>
        <form className="chess-panel-body" onSubmit={e => { e.preventDefault(); savePath(path) }}>
          <Field label="Executable path"><input placeholder="Use STOCKFISH_PATH or find stockfish on PATH" value={path}
            onChange={e => setPath(e.target.value)}/></Field>
          {settings?.resolved_path && <small className="chess-break">{settings.resolved_path}</small>}
          <button type="submit" className="button secondary small chess-fit" disabled={!!busy}>Save setting</button>
        </form>
      </Section>
    </div>
  </div>
}

function AccuracyView({ checkpoints, status, onError, onImported }: {
  checkpoints: Checkpoint[]; status: Status; onError: (error: string) => void; onImported: () => Promise<void>;
}) {
  const checkpointFile = useRef<HTMLInputElement>(null)
  const [history, setHistory] = useState<Summary[]>([])
  const [run, setRun] = useState('')
  const [protocol, setProtocol] = useState('')
  const [checkpoint, setCheckpoint] = useState('')
  const [report, setReport] = useState<Report | null>(null)
  const [selectedReport, setSelectedReport] = useState('')
  const [running, setRunning] = useState(false)
  const [uploading, setUploading] = useState(false)

  const refresh = useCallback(() => api<Summary[]>('/chess/accuracy').then(setHistory), [])
  useEffect(() => {
    refresh().catch(error => onError(message(error)))
    if (!activeStates.includes(status.state) || status.mode !== 'chess') return
    const timer = window.setInterval(() => refresh().catch(error => onError(message(error))), 3000)
    return () => window.clearInterval(timer)
  }, [refresh, onError, status.checkpoint, status.mode, status.state])

  const byDate = [...history].sort((a, b) => b.created_at.localeCompare(a.created_at))
  const runOptions = Array.from(new Set(byDate.map(item => item.run_id ?? '')))
  const activeRun = runOptions.includes(run) ? run : status.run_id && runOptions.includes(status.run_id) ? status.run_id : runOptions[0]
  const runReports = history.filter(item => (item.run_id ?? '') === activeRun)
  const protocols = Array.from(new Map(runReports.map(item => [item.benchmark_id, item])).values()).reverse()
  const selectedProtocol = protocols.find(item => item.benchmark_id === protocol)
    ?? [...runReports].reverse().find(item => item.source === 'automatic') ?? protocols[0]
  const matchingReports = runReports.filter(item => item.benchmark_id === selectedProtocol?.benchmark_id)
  const chartReports = Array.from(new Map(matchingReports.map(item => [item.checkpoint, item])).values())
  const selected = activeStates.includes(status.state) && status.mode === 'chess' && !checkpoint
    ? status.checkpoint ?? checkpoints[0]?.name ?? ''
    : checkpoint || checkpoints[0]?.name || ''
  const reportItems = [...matchingReports].reverse()
  const reportKey = (item: Summary) => item.checkpoint + '/' + item.source
  const currentReport = reportItems.find(item => reportKey(item) === selectedReport) ?? reportItems[0]
  const detailName = currentReport?.checkpoint
  const detailSource = currentReport?.source
  const detailProtocol = currentReport?.benchmark_id
  const detailCreated = currentReport?.created_at

  useEffect(() => {
    let cancelled = false
    setReport(null)
    if (detailName && detailSource) {
      api<Report>('/chess/checkpoints/' + encodeURIComponent(detailName) + '/accuracy?source=' + detailSource + '&benchmark_id=' + detailProtocol)
        .then(value => { if (!cancelled) setReport(value) })
        .catch(error => { if (!cancelled) onError(message(error)) })
    }
    return () => { cancelled = true }
  }, [detailName, detailSource, detailProtocol, detailCreated, onError])

  const importCheckpoint = async (file: File) => {
    setUploading(true)
    onError('')
    try {
      const form = new FormData()
      form.append('file', file)
      const entry = await api<Checkpoint>('/checkpoints/upload?mode=chess', 'POST', form)
      await onImported()
      setCheckpoint(entry.name)
    } catch (error) { onError(message(error)) }
    finally { setUploading(false) }
  }

  const benchmark = async () => {
    setRunning(true)
    onError('')
    try {
      const result = await api<Report>('/chess/checkpoints/' + encodeURIComponent(selected) + '/accuracy', 'POST', {})
      await refresh()
      setRun(result.run_id ?? '')
      setProtocol(result.benchmark_id)
      setSelectedReport(reportKey(result))
      setReport(result)
    } catch (error) { onError(message(error)) }
    finally { setRunning(false) }
  }

  const displayedReport = report && currentReport && reportKey(report) === reportKey(currentReport)
    && report.benchmark_id === currentReport.benchmark_id ? report : null
  const last = chartReports.at(-1)
  const failed = displayedReport?.samples.filter(sample => !sample.verification.passed) ?? []
  return <div className="chess-stack">
    <Section title="Checkpoint benchmark" icon={<Activity size={17}/>}>
      <div className="chess-panel-body chess-toolbar">
        <Field label="Chess checkpoint"><CheckpointPicker checkpoints={checkpoints} value={selected} onChange={setCheckpoint}/></Field>
        <button type="button" className="button primary" disabled={!selected || running || uploading} onClick={benchmark}>
          {running ? <Loader2 size={15} className="spin"/> : <Play size={15}/>}
          {running ? 'Running benchmark…' : 'Run accuracy benchmark'}
        </button>
        <input ref={checkpointFile} type="file" accept=".pt" hidden onChange={e => {
          const file = e.target.files?.[0]
          if (file) void importCheckpoint(file)
          e.target.value = ''
        }}/>
        <button type="button" className="button secondary" disabled={running || uploading} onClick={() => checkpointFile.current?.click()}>
          {uploading ? <Loader2 size={15} className="spin"/> : <FileUp size={15}/>} Import .pt checkpoint
        </button>
      </div>
    </Section>
    <Section title="Checkpoint accuracy over time" icon={<Activity size={17}/>}>
      <div className="chess-panel-body">
        {!!history.length && <div className="chess-toolbar chess-curve-filters">
          <Field label="Experiment"><select value={activeRun ?? ''} onChange={e => { setRun(e.target.value); setProtocol(''); setSelectedReport('') }}>
            {runOptions.map(id => <option value={id} key={id}>{id ? 'Run ' + id : 'Imported checkpoints'}</option>)}
          </select></Field>
          <Field label="Benchmark"><select value={selectedProtocol?.benchmark_id ?? ''} onChange={e => { setProtocol(e.target.value); setSelectedReport('') }}>
            {protocols.map(item => <option key={item.benchmark_id} value={item.benchmark_id}>
              {item.protocol.size} positions · {item.protocol.engine ?? 'Board verification'} · {item.benchmark_id.slice(0, 5)}
            </option>)}
          </select></Field>
        </div>}
        {last ? <>
          <div className="chess-metric-grid">{series.map(item => <div key={item.key}>
            <span><i style={{ background: item.color }}/>{item.label}</span>
            <strong>{last.metrics[item.key] == null ? '—' : last.metrics[item.key]?.toFixed(1) + '%'}</strong>
          </div>)}</div>
          <div className="chess-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartReports.map(item => ({ step: item.step, ...item.metrics }))}>
                <CartesianGrid stroke="#2a3038" vertical={false} strokeDasharray="3 5"/>
                <XAxis dataKey="step" tick={{ fill: '#969eac', fontSize: 12 }} tickLine={false}/>
                <YAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} width={48}
                  tickFormatter={value => value + '%'} tick={{ fill: '#969eac', fontSize: 12 }} tickLine={false}/>
                <Tooltip labelFormatter={value => 'Step ' + value} contentStyle={{ background: '#1b1f26', border: '1px solid #363c46', borderRadius: 8 }}/>
                {series.map(item => <Line key={item.key} dataKey={item.key} name={item.label} stroke={item.color}
                  strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false}/>)}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </> : <p className="chess-empty">Accuracy curves appear after the first chess checkpoint evaluation.</p>}
      </div>
    </Section>
    <Section title="Failed validation examples" icon={<CircleAlert size={17}/>}>
      <div className="chess-panel-body">
        {reportItems.length > 0 && <Field label="Evaluation report">
          <select value={currentReport ? reportKey(currentReport) : ''} onChange={e => setSelectedReport(e.target.value)}>
            {reportItems.map(item => <option key={reportKey(item)} value={reportKey(item)}>Step {item.step} · {item.source}</option>)}
          </select>
        </Field>}
        {displayedReport && <p className="muted">{displayedReport.metrics.passed_samples} of {displayedReport.metrics.sample_count} positions passed
          {displayedReport.metrics.average_centipawn_loss != null && ' · Mean recommended-move loss ' + displayedReport.metrics.average_centipawn_loss + ' cp'}</p>}
        {failed.length ? <div className="chess-table-scroll"><table className="chess-table">
          <thead><tr><th>Position and move</th><th>Model explanation</th><th>Verification failures</th></tr></thead>
          <tbody>{failed.map((sample, index) => <tr key={sample.fen + index}>
            <td><strong>{sample.move}</strong><code>{sample.fen}</code></td>
            <td><HighlightedText text={sample.explanation} errors={sample.verification.errors}/></td>
            <td><ul>{sample.verification.errors.map((issue, i) => <li key={i}>
              <strong>{issue.claim || categoryLabels[issue.category]}</strong> {issue.reason}
            </li>)}</ul></td>
          </tr>)}</tbody>
        </table></div> : <p className="chess-empty">
          {displayedReport ? 'No failed examples in this report.' : currentReport ? 'Loading report…' : 'No evaluation report yet.'}
        </p>}
      </div>
    </Section>
  </div>
}

const pieceGlyphs: Record<string, string> = { k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟' }
function Chessboard({ fen, move }: { fen: string; move?: Context['move'] }) {
  const rows = fen.split(' ')[0].split('/').map(rank => [...rank].flatMap(char => /[1-8]/.test(char) ? Array<string>(+char).fill('') : [char]))
  return <svg viewBox="0 0 480 480" className="chess-board" role="img" aria-label="Chessboard, White at the bottom">
    {Array.from({ length: 64 }, (_, index) => {
      const file = index % 8, rank = Math.floor(index / 8)
      const square = 'abcdefgh'[file] + (8 - rank)
      const x = file * 60, y = rank * 60
      const light = (file + rank) % 2 === 0
      const fill = move?.destination === square || move?.origin === square ? light ? '#b4cb6c' : '#95b550' : light ? '#dce5cb' : '#6a8866'
      const piece = rows[rank]?.[file] ?? ''
      const white = piece === piece.toUpperCase()
      const coordinate = light ? '#476144' : '#edf1e2'
      return <g key={square}>
        <rect x={x} y={y} width="60" height="60" fill={fill}/>
        {rank === 7 && <text x={x + 52} y={y + 54} fill={coordinate} className="chess-coordinate">{'abcdefgh'[file]}</text>}
        {file === 0 && <text x={x + 8} y={y + 13} fill={coordinate} className="chess-coordinate">{8 - rank}</text>}
        {piece && <text x={x + 30} y={y + 46} className="chess-piece" stroke={white ? '#343c2e' : '#e5edd7'}
          fill={white ? '#fff8e8' : '#263228'}>{pieceGlyphs[piece.toLowerCase()]}</text>}
      </g>
    })}
  </svg>
}

function HighlightedText({ text, errors }: { text: string; errors: Claim[] }) {
  if (!text) return <span className="muted">No explanation generated</span>
  const chars = [...text]
  const ranges = errors.filter(item => item.end > item.start).sort((a, b) => a.start - b.start)
  const merged: { start: number; end: number }[] = []
  for (const item of ranges) {
    const previous = merged.at(-1)
    if (previous && previous.end >= item.start) previous.end = Math.max(previous.end, item.end)
    else merged.push({ start: item.start, end: item.end })
  }
  let cursor = 0
  const parts = merged.map(({ start, end }) => {
    const before = chars.slice(cursor, start).join('')
    cursor = end
    return <span key={start}>{before}<mark>{chars.slice(start, end).join('')}</mark></span>
  })
  return <span className="chess-highlighted">{parts}{chars.slice(cursor).join('')}</span>
}

function VerificationDetails({ verification }: { verification: Verification }) {
  const groups = [...new Set([...verification.checks, ...verification.errors].map(item => item.category))]
  return <div className="chess-verification">
    <div className="chess-actions">
      {groups.map(category => {
        const failed = verification.errors.some(item => item.category === category)
        return <span key={category} className={'chess-pill ' + (failed ? 'bad' : 'good')}>
          {failed ? <X size={13}/> : <Check size={13}/>} {categoryLabels[category] ?? category}
        </span>
      })}
    </div>
    {verification.errors.length > 0 && <ul className="chess-issues">
      {verification.errors.map((item, index) => <li key={index}>{item.claim && <strong>{item.claim} · </strong>}{item.reason}</li>)}
    </ul>}
    {verification.recommendations.some(move => move.centipawn_loss != null) && <p className="muted">
      {verification.recommendations.filter(move => move.centipawn_loss != null)
        .map(move => move.move + ' ' + move.centipawn_loss + ' cp loss').join(' · ')}
    </p>}
  </div>
}

const scoreText = (score: Score) => score.mate != null ? (score.mate_color === 'white' ? '+' : '−') + 'M' + Math.abs(score.mate)
  : ((score.cp ?? 0) >= 0 ? '+' : '') + ((score.cp ?? 0) / 100).toFixed(2)

function Sandbox({ checkpoints, checkpoint, setCheckpoint, engineAvailable }: {
  checkpoints: Checkpoint[]; checkpoint: string; setCheckpoint: (value: string) => void; engineAvailable: boolean;
}) {
  const [kind, setKind] = useState<'fen' | 'pgn'>('fen')
  const [fen, setFen] = useState(startingFen)
  const [pgn, setPgn] = useState('')
  const [move, setMove] = useState('e2e4')
  const [ply, setPly] = useState(1)
  const [position, setPosition] = useState<Position | null>(null)
  const [positionError, setPositionError] = useState('')
  const [error, setError] = useState('')
  const [result, setResult] = useState<TestResult | null>(null)
  const [resultInput, setResultInput] = useState<PositionInput | null>(null)
  const [useStockfish, setUseStockfish] = useState(engineAvailable)
  const [afterMove, setAfterMove] = useState(false)
  const [busy, setBusy] = useState('')
  const [edit, setEdit] = useState('')
  const [correction, setCorrection] = useState<{ saved: boolean; verification: Verification } | null>(null)
  const [editing, setEditing] = useState(false)

  useEffect(() => {
    let cancelled = false
    if (kind === 'pgn' && !pgn.trim()) return
    const payload = kind === 'fen' ? { fen } : { pgn, ply }
    const timer = window.setTimeout(() => {
      api<Position>('/chess/position', 'POST', payload)
        .then(data => {
          if (cancelled) return
          setPosition(data)
          setPositionError('')
          if (kind === 'fen') setMove(current => data.legal_moves.some(item => item.uci === current) ? current : data.legal_moves[0]?.uci ?? '')
        })
        .catch(error => { if (!cancelled) { setPositionError(message(error)); setPosition(null) } })
    }, 200)
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [kind, fen, pgn, ply])

  const input: PositionInput = kind === 'fen' ? { fen, move } : { pgn, ply }
  const generate = async () => {
    setBusy('generate')
    setError('')
    setCorrection(null)
    setEditing(false)
    try {
      const response = await api<TestResult>('/chess/test-position', 'POST', { ...input, checkpoint, use_stockfish: useStockfish })
      setResult(response)
      setResultInput(input)
      setEdit(response.explanation)
      setAfterMove(false)
    } catch (error) { setError(message(error)) }
    finally { setBusy('') }
  }
  const save = async () => {
    if (!result || !resultInput) return
    setBusy('save')
    setError('')
    try {
      const response = await api<{ saved: boolean; verification: Verification }>('/chess/corrections', 'POST', {
        ...resultInput, checkpoint: result.checkpoint, explanation: edit, original_output: result.explanation,
        use_stockfish: result.stockfish_available,
      })
      setCorrection(response)
    } catch (error) { setError(message(error)) }
    finally { setBusy('') }
  }

  const match = result && JSON.stringify(input) === JSON.stringify(resultInput) && result.checkpoint === checkpoint
  const visibleResult = match ? result : null
  const context = visibleResult?.context ?? position?.context
  const selectedMove = position?.legal_moves.find(item => item.uci === move)
  const boardMove = context?.move ?? (selectedMove && {
    ...selectedMove, origin: selectedMove.uci.slice(0, 2), destination: selectedMove.uci.slice(2, 4),
  })
  const score = afterMove ? context?.eval_after : context?.eval_before
  const whiteShare = score ? score.mate != null ? (score.mate_color === 'white' ? 100 : 0)
    : 100 / (1 + Math.exp(-(score.cp ?? 0) / 240)) : 50

  const resetPosition = () => { setPosition(null); setPositionError(''); setAfterMove(false) }
  return <div className="chess-sandbox-grid">
    <Section title="Position" icon={<Crosshair size={17}/>}>
      <div className="chess-panel-body">
        <div className="chess-board-wrap">
          {score && <div className="chess-eval-bar" role="img" aria-label={'Stockfish score ' + scoreText(score)}>
            <div style={{ height: whiteShare + '%' }}/><span>{scoreText(score)}</span>
          </div>}
          <Chessboard fen={(afterMove && context?.after_fen) || position?.fen || startingFen}
            move={boardMove}/>
        </div>
        {position && <div className="chess-board-caption">
          <span>{(afterMove ? context?.after_fen ?? position.fen : position.fen).split(' ')[1] === 'b' ? 'Black' : 'White'} to move</span>
          {context && <button type="button" className="button secondary small" onClick={() => setAfterMove(!afterMove)}>
            {afterMove ? 'Before move' : 'After ' + context.move.san}
          </button>}
        </div>}
        {context?.multipv?.length ? <details className="chess-pv" open>
          <summary>Stockfish principal variations <ChevronDown size={13}/></summary>
          <small>From the position before the move</small>
          {context.multipv.map(line => <div key={line.rank}>
            <strong>{scoreText(line.score)}</strong><code>{line.pv.join(' ') || '—'}</code>
          </div>)}
        </details> : null}
      </div>
    </Section>
    <div className="chess-stack">
      <Section title="Test an explanation" icon={<Play size={17}/>}>
        <form className="chess-panel-body" onSubmit={e => { e.preventDefault(); void generate() }}>
          <div className="chess-segment" role="group" aria-label="Position format">
            <button type="button" aria-pressed={kind === 'fen'} onClick={() => { if (kind !== 'fen') resetPosition(); setKind('fen') }}>FEN</button>
            <button type="button" aria-pressed={kind === 'pgn'} onClick={() => { if (kind !== 'pgn') resetPosition(); setKind('pgn') }}>PGN</button>
          </div>
          {kind === 'fen' ? <>
            <Field label="Position FEN"><textarea className="chess-input-mono" rows={2} value={fen}
              onChange={e => { setFen(e.target.value); resetPosition() }}/></Field>
            <Field label="Legal move"><select value={move} onChange={e => { setMove(e.target.value); setAfterMove(false) }}>
              {!position?.legal_moves.length && <option value="">No legal moves loaded</option>}
              {position?.legal_moves.map(item => <option key={item.uci} value={item.uci}>{item.san} ({item.uci})</option>)}
            </select></Field>
          </> : <>
            <Field label="Game PGN"><textarea className="chess-input-mono" rows={3} placeholder="1. e4 e5 2. Nf3 Nc6"
              value={pgn} onChange={e => { setPgn(e.target.value); setPly(1); resetPosition() }}/></Field>
            <Field label="PGN ply"><select value={ply} onChange={e => { setPly(+e.target.value); setAfterMove(false) }}>
              {!position?.pgn_moves.length && <option value={1}>Load a PGN game</option>}
              {position?.pgn_moves.map(item => <option key={item.ply} value={item.ply}>{item.ply} · {item.san}</option>)}
            </select></Field>
          </>}
          {positionError && <p className="inline-error" role="alert">{positionError}</p>}
          <Field label="Chess checkpoint"><CheckpointPicker checkpoints={checkpoints} value={checkpoint} onChange={setCheckpoint}/></Field>
          <label className="chess-checkbox">
            <input type="checkbox" checked={useStockfish} onChange={e => setUseStockfish(e.target.checked)}/>
            Include Stockfish analysis
          </label>
          <button type="submit" className="button primary chess-fit" disabled={!!busy || !checkpoint || !position || (kind === 'fen' && !move)}>
            {busy === 'generate' ? <Loader2 size={15} className="spin"/> : <Crosshair size={15}/>}
            {busy === 'generate' ? 'Checking position…' : 'Generate and verify'}
          </button>
          {!checkpoints.length && <p className="muted">Train a chess model to create a checkpoint.</p>}
          {error && !editing && <p className="inline-error" role="alert">{error}</p>}
        </form>
      </Section>
      {visibleResult && <Section title="Model explanation" icon={<Activity size={17}/>}
        action={<span className={'chess-pill ' + (visibleResult.verification.passed ? 'good' : 'bad')}>
          {visibleResult.verification.passed ? <Check size={13}/> : <X size={13}/>}
          {visibleResult.verification.passed ? 'Passed' : 'Needs review'}
        </span>}>
        <div className="chess-panel-body">
          <p className="chess-explanation"><HighlightedText text={visibleResult.explanation} errors={visibleResult.verification.errors}/></p>
          {useStockfish && !visibleResult.stockfish_available && <p className="muted">Stockfish unavailable. Board checks completed.</p>}
          <VerificationDetails verification={visibleResult.verification}/>
          {!editing ? <button className="button secondary small chess-fit" type="button" onClick={() => setEditing(true)}>Edit explanation</button>
            : <>
              <Field label="Corrected explanation"><textarea value={edit} onChange={e => { setEdit(e.target.value); setCorrection(null) }}/></Field>
              <div className="chess-actions">
                <button type="button" className="button secondary" onClick={save} disabled={!!busy || !edit.trim() || correction?.saved}>
                  {busy === 'save' ? <Loader2 size={14} className="spin"/> : <Check size={14}/>} Save Verified Chess Correction
                </button>
                <button type="button" className="text-button" onClick={() => setEditing(false)}>Close editor</button>
              </div>
              {error && <p className="inline-error" role="alert">{error}</p>}
              {correction?.saved && <p className="chess-success" role="status"><Check size={14}/> Saved to the chess fine-tuning dataset.</p>}
              {correction && !correction.saved && <VerificationDetails verification={correction.verification}/>}
            </>}
        </div>
      </Section>}
    </div>
  </div>
}
