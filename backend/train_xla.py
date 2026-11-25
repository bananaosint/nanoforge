"""Kaggle/PJRT TPU data parallel training, isolated from the local Studio process.

Run in a fresh process: python -m backend.train_xla --config configs/kaggle-tpu.json
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import uuid
import torch
from .api.schemas import NewRun
from .ml.corpus import FileCorpus, PairPool, vocabulary_text
from .ml.dataset import DatasetManager, WeightedMixer
from .ml.model import ModelConfig, resolve_model
from .ml.tokenizer import Tokenizer, format_prompt, END
from .ml.xla_training import run_replica
from .storage import Store


def prepare(config,home,resume=None,seed=42,expected_replicas=0,log_every=10,keep_last=1):
    """Build caches once before spawning, avoiding concurrent .partial writes."""
    config=NewRun.model_validate(config).model_dump(exclude_none=True)
    if config['watchdog_enabled']:raise ValueError('Gemini watchdog is supported by the persistent Studio/CLI trainer, not TPU/XLA workers.')
    if config['precision']=='float16':
        raise ValueError('TPUs use bfloat16, not CUDA float16. Set precision to bfloat16 or float32.')
    if config['precision']=='auto':config['precision']='bfloat16'
    home=Path(home).resolve();home.mkdir(parents=True,exist_ok=True)
    manager=DatasetManager(home/'data')
    feedback=[]
    if (home/'nanoforge.sqlite3').exists():
        store=Store(home/'nanoforge.sqlite3')
        try:feedback=store.feedback('preference' if config['mode']=='dpo' else 'sft')
        finally:store.close()
    sources=[];inline=[]
    for source in config['datasets']:
        if source['weight']<=0:continue
        if source['name']=='hitl':
            if config['mode']=='dpo':
                text='\n'.join(format_prompt(pair['prompt'])+pair['response_a']+END+pair['response_b']+END for pair in feedback)
            else:
                text='\n'.join(format_prompt(pair['prompt'])+pair['corrected_output']+END
                               for pair in feedback if pair.get('corrected_output'))
            if text:inline.append((text,source['weight']))
        else:sources.append((str(manager.path(source['name']).resolve()),source['weight']))
    texts=[(FileCorpus(Path(path)),weight) for path,weight in sources]+inline
    if not texts:raise ValueError('Select at least one nonempty dataset with a positive weight.')
    if resume:
        resume=str(Path(resume).resolve())
        payload=torch.load(resume,map_location='cpu',weights_only=True,mmap=True)
        tokenizer=Tokenizer(state=payload['tokenizer_state'])
        model=ModelConfig(**payload['model_config'])
        if tokenizer.kind!=config['tokenizer'] or model.vocab_size!=tokenizer.vocab_size:
            raise ValueError('Resume requires the checkpoint tokenizer and matching vocabulary.')
        config['parent_checkpoint']=Path(resume).name
        config['model']={key:value for key,value in asdict(model).items() if key!='vocab_size'}
        del payload
    else:
        tokenizer=Tokenizer(config['tokenizer'],vocabulary_text(texts) if config['tokenizer']=='char' else '')
        model=resolve_model(config['preset'],config.get('model',{}),tokenizer.vocab_size)
    if model.architecture=='moe':raise ValueError('MoE is not supported by the TPU/XLA trainer. Use the CPU/CUDA Studio.')
    # Prepare shared uint16/uint32 token arrays and pair indexes in the parent.
    if config['mode']=='pretrain' or (config['mode']=='sft' and config['base_mix']>0):
        WeightedMixer(texts,tokenizer,model.block_size,'cpu')
    if config['mode']!='pretrain':
        pairs=PairPool(feedback,[FileCorpus(Path(path)) for path,_ in sources],
                       'sft' if config['mode']=='sft' else 'preference')
        if not pairs:raise ValueError('No usable SFT/preference pairs in selected data or feedback.')
    # Always store resolved dimensions for exact reconstruction in Studio.
    config['model']={key:value for key,value in asdict(model).items() if key!='vocab_size'}
    return {'home':str(home),'config':config,'model_config':asdict(model),'tokenizer_state':tokenizer.state(),
            'sources':sources,'inline_sources':inline,'feedback':feedback,'resume':resume,
            'run_id':uuid.uuid4().hex[:12],'seed':seed,'expected_replicas':expected_replicas,
            'log_every':log_every,'keep_last':keep_last}


class XlaRuntime:
    def __init__(self):
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        self.xla,self.xm=torch_xla,xm
        if xr.device_type()!='TPU':raise ValueError('The XLA backend requires a TPU runtime, not CPU/GPU.')
        self.device=torch_xla.device()
        self.rank,self.world_size=xr.global_ordinal(),xr.world_size()

    def optimizer(self,parameters,learning_rate):
        from torch_xla.amp.syncfree import AdamW
        return AdamW(parameters,lr=learning_rate,weight_decay=0.1,foreach=False,fused=False)

    def autocast(self,precision):
        return torch.autocast('xla',dtype=torch.bfloat16) if precision=='bfloat16' else nullcontext()

    def broadcast(self,model):self.xm.broadcast_master_param(model)
    def seed(self,seed):self.xm.set_rng_state(seed,self.device)
    def reduce_gradients(self,optimizer):self.xm.reduce_gradients(optimizer)
    def mean(self,value):
        return self.xm.all_reduce(self.xm.REDUCE_SUM,value,scale=1/self.world_size) if self.world_size>1 else value
    def step_optimizer(self,optimizer,found_inf):optimizer.step(found_inf=found_inf)
    def sync(self):self.xla.sync(wait=True)
    def barrier(self,tag,error=''):
        errors=self.xm.rendezvous(tag,payload=error.encode('utf-8'))
        for failure in errors:
            if failure:raise ValueError('Distributed checkpoint failed: '+failure.decode('utf-8',errors='replace'))


def worker(index,plan):
    runtime=XlaRuntime()
    # Tiny CPU-side preprocessing should not start dozens of threads per replica.
    run_replica(plan,runtime)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--home',type=Path,default=Path('/kaggle/working/nanoforge-run'))
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--expected-replicas',type=int,default=0,help='Fail if the actual replica count differs (8 for Kaggle TPU x8).')
    parser.add_argument('--single-core',action='store_true',help='Troubleshooting only; default launches every available TPU device.')
    parser.add_argument('--log-every',type=int,default=10)
    parser.add_argument('--keep-last',type=int,default=1,help='Keep this many completed checkpoints from this invocation; 0 keeps all.')
    args=parser.parse_args()
    if args.log_every<1 or args.expected_replicas<0 or args.keep_last<0:
        parser.error('log-every must be positive; expected-replicas and keep-last must be nonnegative.')
    os.environ.setdefault('PJRT_DEVICE','TPU')
    os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('MKL_NUM_THREADS','1')
    if any(os.environ.get(key,'').lower() in ('1','true') for key in ('XLA_USE_BF16','XLA_DOWNCAST_BF16')):
        parser.error('Unset legacy XLA_USE_BF16/XLA_DOWNCAST_BF16 flags. This trainer uses autocast with FP32 master parameters.')
    # Avoid importing/initializing an XLA device in the notebook/launcher before spawn.
    try:
        version=importlib.metadata.version('torch_xla')
        if tuple(version.split('.')[:2])!=tuple(torch.__version__.split('.')[:2]):
            raise ValueError(f'torch {torch.__version__} and torch_xla {version} must have matching major/minor versions.')
        if tuple(int(part) for part in version.split('.')[:2])<(2,8):
            raise ValueError('This training path requires PyTorch/XLA 2.8 or newer.')
        import torch_xla
    except (ImportError,importlib.metadata.PackageNotFoundError,OSError) as exc:
        raise SystemExit('A matching torch / torch_xla[tpu] installation is required. Follow docs/kaggle-tpu.md. '+str(exc)) from exc
    torch.set_num_threads(1)
    plan=prepare(json.loads(args.config.read_text()),args.home,args.resume,args.seed,args.expected_replicas,args.log_every,args.keep_last)
    torch_xla.launch(worker,args=(plan,),debug_single_process=args.single_core)
    manifest=json.loads((args.home/'latest-checkpoint.json').read_text())
    print(json.dumps({'state':'completed',**manifest}),flush=True)


if __name__=='__main__':main()
