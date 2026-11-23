"""Run from the repository root: uvicorn backend.main:app --port 8000."""
import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from .api.schemas import NewRun, Training, Generation, Arena, Feedback, CheckpointSelection, Prompts
from .ml.dataset import DatasetManager
from .ml.corpus import FileCorpus
from .ml.model import PRESETS, device_name
from .ml.trainer import Trainer
from .storage import Store
from .checkpoints import register_checkpoint
from .importers.hub import HubImports
from .api.schemas import HubImportRequest


def create_app(root=None):
    root = Path(root or os.environ.get('NANOFORGE_HOME', Path(__file__).resolve().parents[1]))
    root.mkdir(parents=True, exist_ok=True)
    store = Store(root / 'nanoforge.sqlite3')
    datasets = DatasetManager(root / 'data')
    trainer = Trainer(datasets, store, root / 'checkpoints')
    imports = HubImports(datasets)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(imports.shutdown)
        await asyncio.to_thread(trainer.shutdown)
        store.close()

    app = FastAPI(title='NanoForge', version='1.0.0', lifespan=lifespan)
    app.state.trainer, app.state.store = trainer, store
    app.state.imports = imports
    app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
                       allow_methods=['*'], allow_headers=['*'])

    @app.exception_handler(ValueError)
    async def bad_request(request, exc):
        return JSONResponse(status_code=400, content={'detail': str(exc)})

    @app.get('/api/health')
    def health():
        return {'ok': True, 'device': trainer.device, 'device_name': device_name(trainer.device)}

    @app.get('/api/status')
    def status():
        return trainer.status()

    @app.get('/api/presets')
    def presets():
        return PRESETS

    @app.get('/api/datasets')
    def list_datasets():
        return datasets.list()

    @app.post('/api/datasets/upload')
    async def upload(file: UploadFile = File(...)):
        name = Path(file.filename or '').name
        if name != file.filename or Path(name).suffix.lower() not in ('.txt', '.jsonl'):
            raise ValueError('Upload a .txt or .jsonl file.')
        raw = await file.read(20 * 1024 * 1024 + 1)
        if len(raw) > 20 * 1024 * 1024:
            raise ValueError('Files must be smaller than 20 MB.')
        try:
            text = raw.decode('utf-8')
        except UnicodeError:
            raise ValueError('Use a UTF-8 encoded file.')
        if not text.strip():
            raise ValueError('The uploaded dataset is empty.')
        path = datasets.directory / name
        if path.exists():
            raise ValueError('A dataset with that name already exists. Rename the file first.')
        path.write_text(text, encoding='utf-8')
        try:
            if not datasets.text(name).strip():
                raise ValueError('No usable text in the dataset.')
        except Exception:
            path.unlink()
            raise
        return {'name': name, 'bytes': len(raw)}

    @app.get('/api/imports')
    def import_status():
        return imports.status()

    @app.post('/api/imports/start')
    def import_start(body: HubImportRequest):
        return imports.start(body.dataset_ids if body.dataset_ids is not None else body.dataset_id, body.max_records)

    @app.post('/api/imports/pause')
    def import_pause():
        return imports.pause()

    @app.post('/api/datasets/tinyshakespeare')
    def download():
        return datasets.download()

    @app.get('/api/datasets/{name}/preview')
    def preview(name: str):
        corpus = FileCorpus(datasets.path(name))
        text = corpus.preview()
        size = corpus.path.stat().st_size
        return {'text': text, 'bytes': size, 'characters': len(text) if size <= 5000 else None}

    @app.delete('/api/datasets/{name}')
    def delete_dataset(name: str):
        if trainer.busy() and any(s['name'] == name and s['weight'] > 0 for s in trainer.config.get('datasets', [])):
            raise ValueError('Stop training before deleting a dataset used by the active run.')
        datasets.path(name).unlink()
        imports.forget_output(name)
        return {'ok': True}

    @app.post('/api/runs')
    def create_run(body: NewRun):
        return trainer.create(body.model_dump(exclude_none=True))

    @app.get('/api/runs')
    def runs():
        return [{**row, 'config': json.loads(row['config'])} for row in store.rows('SELECT * FROM runs ORDER BY created_at DESC')]

    @app.get('/api/runs/{run_id}/metrics')
    def metrics(run_id: str):
        return [json.loads(row['payload']) for row in store.rows('SELECT payload FROM metrics WHERE run_id=? ORDER BY id', (run_id,))]

    @app.post('/api/training/start')
    def start(body: Training):
        return trainer.start(body.model_dump())

    @app.post('/api/training/pause')
    def pause():
        return trainer.pause()

    @app.post('/api/training/resume')
    def resume():
        return trainer.resume()

    @app.post('/api/training/stop')
    def stop():
        return trainer.stop()

    @app.get('/api/checkpoints')
    def checkpoints():
        return trainer.checkpoints()

    @app.post('/api/checkpoints/upload')
    async def upload_checkpoint(file: UploadFile = File(...)):
        name=file.filename or ''
        if Path(name).name!=name or '\\' in name or not name.endswith('.pt') or name.startswith('.'):
            raise ValueError('Upload a .pt checkpoint filename.')
        if (trainer.directory/name).exists():
            raise ValueError('A checkpoint with that filename already exists. Rename the upload first.')
        handle,pending=tempfile.mkstemp(prefix='.upload-',suffix='.pt',dir=trainer.directory)
        os.close(handle)
        try:
            size=0
            with open(pending,'wb') as target:
                while chunk:=await file.read(1024*1024):
                    size+=len(chunk)
                    if size>2*1024*1024*1024:
                        raise ValueError('Checkpoint uploads must be smaller than 2 GiB.')
                    target.write(chunk)
            return await asyncio.to_thread(register_checkpoint,pending,trainer.directory,name)
        finally:
            Path(pending).unlink(missing_ok=True)

    @app.post('/api/checkpoints/save')
    def save_checkpoint():
        return trainer.save()

    @app.post('/api/checkpoints/select')
    def select(body: CheckpointSelection):
        return trainer.load(body.name)

    @app.delete('/api/checkpoints/{name}')
    def delete_checkpoint(name: str):
        with trainer.lock:
            trainer.require_idle()
            if name == trainer.checkpoint:
                raise ValueError('Select another checkpoint before deleting the active checkpoint.')
            path = trainer.checkpoint_path(name)
            path.unlink()
            path.with_suffix('.json').unlink(missing_ok=True)
        return {'ok': True}

    @app.get('/api/timeline')
    def timeline(run_id: str | None = None):
        return store.rows('SELECT * FROM timeline WHERE run_id=? ORDER BY step,id', (run_id or trainer.run_id,))

    @app.put('/api/prompts')
    def prompts(body: Prompts):
        with trainer.lock:
            trainer.prompts = body.prompts
            store.set_setting('prompts', body.prompts)
        return {'prompts': body.prompts}

    def generation_kwargs(body):
        return body.model_dump(exclude={'layer', 'head'})

    @app.post('/api/generate')
    def generate(body: Generation):
        if trainer.step == 0:
            raise ValueError('Train at least one step before generating. New models start with random weights.')
        result = trainer.generate(**generation_kwargs(body))
        if body.layer is not None:
            if body.layer >= len(result['attention']):
                raise ValueError('Layer is outside the model or telemetry is disabled.')
            result['attention'] = [result['attention'][body.layer]]
        if body.head is not None:
            if not result['attention'] or body.head >= len(result['attention'][0]):
                raise ValueError('Head is outside the model or telemetry is disabled.')
            result['attention'] = [[layer[body.head]] for layer in result['attention']]
        return result

    @app.post('/api/arena')
    def arena(body: Arena):
        if trainer.step == 0 and (not body.checkpoint_a or not body.checkpoint_b):
            raise ValueError('Train or select a checkpoint before using the Arena.')
        kwargs = body.model_dump(exclude={'checkpoint_a', 'checkpoint_b', 'seed_b', 'layer', 'head'})
        kwargs['telemetry'] = False
        a = trainer.generate_from(body.checkpoint_a, **kwargs)
        kwargs['seed'] = body.seed_b
        b = trainer.generate_from(body.checkpoint_b, **kwargs)
        return {'a': a, 'b': b}

    @app.get('/api/feedback')
    def feedback(kind: str | None = None):
        if kind and kind not in ('sft', 'preference'):
            raise ValueError('Unknown feedback kind.')
        return store.feedback(kind)

    @app.post('/api/feedback')
    def add_feedback(body: Feedback):
        payload = body.model_dump(exclude={'kind'})
        return {'id': store.save_feedback(body.kind, payload)}

    @app.put('/api/feedback/{item_id}')
    def edit_feedback(item_id: int, body: Feedback):
        if not store.rows('SELECT id FROM feedback WHERE id=?', (item_id,)):
            raise HTTPException(404, 'Feedback not found.')
        return {'id': store.save_feedback(body.kind, body.model_dump(exclude={'kind'}), item_id)}

    @app.delete('/api/feedback/{item_id}')
    def delete_feedback(item_id: int):
        store.execute('DELETE FROM feedback WHERE id=?', (item_id,))
        return {'ok': True}

    @app.get('/api/feedback-export')
    def export(kind: str | None = None):
        text = '\n'.join(json.dumps(row, ensure_ascii=False) for row in store.feedback(kind))
        return Response(text + ('\n' if text else ''), media_type='application/x-ndjson',
                        headers={'Content-Disposition': 'attachment; filename="nanoforge-feedback.jsonl"'})

    @app.websocket('/ws/training')
    async def websocket(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                await ws.send_json(trainer.status())
                await asyncio.sleep(0.6)
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app


app = create_app()
