"""CPU preparation + synchronous, bounded GPU calls for the ZeroGPU Gradio adapter."""
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import re
import shutil
import time
import uuid
import torch
from .api.schemas import NewRun
from .checkpoints import register_checkpoint
from .ml.corpus import FileCorpus,PairPool
from .ml.dataset import DatasetManager,WeightedMixer
from .ml.model import ModelConfig,parameter_count
from .ml.tokenizer import Tokenizer
from .ml.trainer import Trainer
from .storage import Store

PROFILE=json.loads((Path(__file__).resolve().parents[1]/'configs/zerogpu.json').read_text())


def session_path(root,session):
    if not isinstance(session,str) or not re.fullmatch('[a-f0-9]{32}',session):raise ValueError('Create a model/session first.')
    root=Path(root).resolve();path=root/session
    if not path.is_dir() or path.is_symlink():raise ValueError('Session expired. Create or import it again.')
    return path


@contextmanager
def session_lock(path):
    with (path/'.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('This session already has a running request. Wait for it to finish.') from exc
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def atomic_json(path,payload):
    pending=path.with_suffix(path.suffix+'.pending');pending.write_text(json.dumps(payload),encoding='utf-8');pending.replace(path)


def validate_model(config):
    if parameter_count(config)>PROFILE['max_parameters']:raise ValueError('ZeroGPU demo limit is 10M total parameters. Use Colab/dedicated GPU for larger training.')
    if config.n_layer>8 or config.n_embd>256 or config.block_size>256:raise ValueError('ZeroGPU limits: 8 layers, width 256, context 256.')
    if config.architecture=='moe' and (config.n_experts>8 or config.experts_per_token>2):raise ValueError('ZeroGPU limits: at most 8 experts and top-k 2.')


def prepare_session(root,options,dataset=None,checkpoint=None):
    """CPU-only callback. GPU worker mutations persist only through saved files."""
    config=NewRun.model_validate(options).model_dump(exclude_none=True)
    if config['multi_gpu']:raise ValueError('ZeroGPU profile uses a single allocated device.')
    if config['watchdog_enabled']:raise ValueError('The asynchronous watchdog needs the persistent Studio/CLI trainer, not ZeroGPU leases.')
    if config['batch_size']>8 or config['gradient_accumulation']>4:raise ValueError('ZeroGPU limits: batch 8, accumulation 4.')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    session=uuid.uuid4().hex;path=root/session;path.mkdir()
    store=None;trainer=None;failed=False
    try:
        manager=DatasetManager(path/'data')
        name='starter-sample.txt'
        if dataset:
            source=Path(dataset)
            if source.suffix.lower() not in ('.txt','.jsonl') or source.stat().st_size>PROFILE['max_dataset_bytes']:
                raise ValueError('Upload a UTF-8 .txt/.jsonl dataset up to 20 MiB.')
            text=source.read_bytes().decode('utf-8-sig')
            name='dataset'+source.suffix.lower()
            if config['mode']=='dpo' and source.suffix.lower()=='.jsonl':
                # Character vocabulary scanning reads prompt/completion records.
                # Preserve preference fields and expose the chosen branch to it.
                rows=[]
                for line in text.splitlines():
                    if not line.strip():continue
                    row=json.loads(line)
                    if isinstance(row,dict) and isinstance(row.get('chosen'),str):row.setdefault('completion',row['chosen'])
                    rows.append(json.dumps(row,ensure_ascii=False))
                text='\n'.join(rows)+'\n'
            (manager.directory/name).write_text(text,encoding='utf-8')
        config['datasets']=[{'name':name,'weight':100}]
        store=Store(path/'nanoforge.sqlite3');trainer=Trainer(manager,store,path/'checkpoints','cpu')
        if checkpoint:
            if Path(checkpoint).stat().st_size>PROFILE['max_checkpoint_bytes']:raise ValueError('Checkpoint upload limit is 128 MiB.')
            metadata=register_checkpoint(checkpoint,trainer.directory)
            model_config=ModelConfig(**metadata['model_config']);validate_model(model_config)
            trainer.load(metadata['name'])
            if trainer.tokenizer.kind!=config['tokenizer']:raise ValueError('Keep the imported checkpoint tokenizer selected.')
        else:
            # Size preview validates the total cap before allocating CPU model weights.
            preview=trainer.preview_model(config);validate_model(ModelConfig(**preview['model_config']))
            trainer.create({**config,'precision':'float32'})
            trainer.save()
        config['model']={key:value for key,value in asdict(trainer.model.config).items() if key!='vocab_size'}
        if config['mode']=='pretrain' or (config['mode']=='sft' and config.get('base_mix',0)>0):
            WeightedMixer([(FileCorpus(manager.path(name)),100)],trainer.tokenizer,trainer.model.config.block_size,'cpu')
        if config['mode']!='pretrain' and not PairPool([], [FileCorpus(manager.path(name))], config['mode']):
            raise ValueError('This mode needs normalized prompt/completion or chosen/rejected JSONL examples.')
        atomic_json(path/'session.json',{'config':config,'checkpoint':trainer.checkpoint,'step':trainer.step})
        return session,f"Ready: {parameter_count(trainer.model.config):,} total parameters; step {trainer.step}. GPU not requested yet.",str(trainer.checkpoint_path(trainer.checkpoint))
    except Exception:
        failed=True
        raise
    finally:
        if trainer:trainer.shutdown()
        if store:store.close()
        if failed:shutil.rmtree(path)


def prune_checkpoints(directory,keep):
    # Only this temporary session's generated versions. Keep the previous committed
    # checkpoint too; never remove it before the new manifest is safely replaced.
    for path in directory.iterdir():
        if path.suffix=='.pt' and path.name not in keep:
            path.unlink()
            path.with_suffix('.json').unlink(missing_ok=True)


def train_chunk(root,session,steps,device='cuda'):
    if not isinstance(steps,int) or not 1<=steps<=PROFILE['max_steps_per_call']:raise ValueError('Choose 1–25 updates per call.')
    path=session_path(root,session)
    with session_lock(path):
        began=time.monotonic();plan=json.loads((path/'session.json').read_text())
        store=Store(path/'nanoforge.sqlite3');trainer=None
        try:
            trainer=Trainer(DatasetManager(path/'data'),store,path/'checkpoints',device)
            trainer.load(plan['checkpoint'])
            # CPU fixtures validate the same loop; real ZeroGPU calls explicitly use CUDA.
            options={**plan['config'],'max_steps':steps,'eval_interval':1000,'precision':plan['config']['precision'] if device=='cuda' else 'float32'}
            status=trainer.start(options,background=False,evaluate=False,
                                 time_budget_seconds=max(0,PROFILE['work_seconds']-(time.monotonic()-began)))
            if status['state']=='error':raise RuntimeError(status['error'])
            atomic_json(path/'session.json',{**plan,'checkpoint':trainer.checkpoint,'step':trainer.step})
            prune_checkpoints(trainer.directory,{plan['checkpoint'],trainer.checkpoint})
            metrics={'step':trainer.step,'state':trainer.state,'last_loss':trainer.metrics[-1]['train_loss'] if trainer.metrics else None,
                     'precision':trainer.precision,'seconds':round(time.monotonic()-began,2)}
            return metrics,str(trainer.checkpoint_path(trainer.checkpoint))
        finally:
            if trainer:trainer.shutdown()
            store.close()


def generate(root,session,prompt,tokens=32,device='cuda'):
    if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>2000:raise ValueError('Use a nonempty prompt of at most 2,000 characters.')
    if not isinstance(tokens,int) or not 1<=tokens<=64:raise ValueError('Choose 1–64 generated tokens.')
    path=session_path(root,session)
    with session_lock(path):
        plan=json.loads((path/'session.json').read_text());store=Store(path/'nanoforge.sqlite3');trainer=None
        try:
            trainer=Trainer(DatasetManager(path/'data'),store,path/'checkpoints',device);trainer.load(plan['checkpoint'])
            return trainer.generate(prompt=prompt,max_new_tokens=tokens,telemetry=False)['text']
        finally:
            if trainer:trainer.shutdown()
            store.close()
