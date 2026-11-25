"""Fixed-shape TPU objectives and portable exports; importing needs no torch_xla."""
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import threading
import time
import torch
from torch.nn import functional as F
from .model import GPT, ModelConfig, parameter_count, export_config
from .tokenizer import Tokenizer
from .dataset import WeightedMixer, completion_batch
from .corpus import FileCorpus, PairPool
from ..storage import now

_MODEL_INIT_LOCK = threading.Lock()


def initialize_model(config,seed,device):
    # TPU v2/v3 can run two device threads per process. Protect the shared CPU RNG.
    with _MODEL_INIT_LOCK,torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model=GPT(ModelConfig(**config))
    return model.to(device)


def fixed_pretrain_batch(mixer, batch_size, rng, validation=False):
    # Include every source and split when picking T, so shape never depends on RNG.
    length=min(mixer.block_size,min(min(len(train),len(val))-1 for train,val,_ in mixer.sources))
    xs,ys=[],[]
    for source in rng.choices(mixer.sources,weights=[s[2] for s in mixer.sources],k=batch_size):
        ids=source[int(validation)]
        start=rng.randrange(len(ids)-length)
        xs.append(torch.tensor(ids[start:start+length].tolist(),dtype=torch.long))
        ys.append(torch.tensor(ids[start+1:start+length+1].tolist(),dtype=torch.long))
    return torch.stack(xs),torch.stack(ys)


def fixed_completion_batch(tokenizer,pairs,block_size,completion_key='corrected_output'):
    xs=torch.zeros((len(pairs),block_size),dtype=torch.long)
    ys=torch.full_like(xs,-100)
    for index,pair in enumerate(pairs):
        x,y=completion_batch(tokenizer,pair['prompt'],pair[completion_key],block_size,'cpu')
        xs[index,:x.shape[1]]=x[0]
        ys[index,:y.shape[1]]=y[0]
    return xs,ys


def log_probabilities(model,x,y):
    mask=y!=-100
    logits=model(x)['logits'].float().log_softmax(-1)
    return (logits.gather(-1,y.clamp_min(0).unsqueeze(-1)).squeeze(-1)*mask).sum(-1)


def objective(model,tokenizer,mode,mixer,pairs,reference,options,rng,device,use_base=False):
    batch_size=options['batch_size']
    if mode=='pretrain' or use_base:
        x,y=fixed_pretrain_batch(mixer,batch_size,rng)
        x,y=x.to(device),y.to(device)
        return model(x,y)['loss'],x.numel()
    selected=rng.choices(pairs,k=batch_size)
    if mode=='sft':
        x,y=fixed_completion_batch(tokenizer,selected,model.config.block_size)
        x,y=x.to(device),y.to(device)
        return model(x,y)['loss'],x.numel()
    chosen,rejected=[],[]
    for pair in selected:
        a,b=pair['response_a'],pair['response_b']
        win,lose=(a,b) if pair['winner']=='a' else (b,a)
        chosen.append({'prompt':pair['prompt'],'corrected_output':win})
        rejected.append({'prompt':pair['prompt'],'corrected_output':lose})
    win=tuple(t.to(device) for t in fixed_completion_batch(tokenizer,chosen,model.config.block_size))
    lose=tuple(t.to(device) for t in fixed_completion_batch(tokenizer,rejected,model.config.block_size))
    model.eval()  # No dropout in either policy or fixed reference for DPO.
    policy=log_probabilities(model,*win)-log_probabilities(model,*lose)
    with torch.no_grad():
        baseline=log_probabilities(reference,*win)-log_probabilities(reference,*lose)
    return -F.logsigmoid(options['beta']*(policy-baseline)).mean(),win[0].numel()+lose[0].numel()


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def export_checkpoint(model,tokenizer,optimizer,plan,step,val_loss,world_size):
    directory=Path(plan['home'])/'checkpoints'
    directory.mkdir(parents=True,exist_ok=True)
    tag=f"{plan['config']['mode']}-{plan['run_id']}-s{step}"
    # XLA tensors have no usable data_ptr; tie the two known embedding names explicitly.
    state=model.state_dict()
    embedding=state['token_embedding.weight'].detach().float().cpu()
    weights={name:embedding if name in ('token_embedding.weight','lm_head.weight') else tensor.detach().float().cpu()
             for name,tensor in state.items()}
    payload={'model_config':export_config(model.config),'tokenizer_state':tokenizer.state(),
             'model_state_dict':weights,'optimizer_state_dict':cpu_tree(optimizer.state_dict()),
             'step':step,'val_loss':val_loss,'version_tag':tag,'training_config':plan['config'],
             'mode':plan['config']['mode'],'run_id':plan['run_id'],'precision':plan['config']['precision'],
             'scaler_state_dict':{},'torch_version':str(torch.__version__),'checkpoint_format_version':1,
             'accelerator':'xla:TPU','world_size':world_size}
    path=directory/(tag+'.pt')
    temporary=path.with_suffix('.pt.pending')
    torch.save(payload,temporary);temporary.replace(path)
    metadata={key:payload[key] for key in ('model_config','step','val_loss','version_tag','mode','run_id','precision')}
    metadata.update(name=path.name,created_at=now(),parameters=parameter_count(model.config),world_size=world_size)
    sidecar=path.with_suffix('.json')
    temporary=sidecar.with_suffix('.json.pending')
    temporary.write_text(json.dumps(metadata),encoding='utf-8');temporary.replace(sidecar)
    manifest=Path(plan['home'])/'latest-checkpoint.json'
    pending=manifest.with_suffix('.json.pending')
    pending.write_text(json.dumps({'checkpoint':str(path),'metadata':str(sidecar)}),encoding='utf-8')
    pending.replace(manifest)
    keep=plan.get('keep_last',1)
    if keep:
        owned=sorted(directory.glob(f"{plan['config']['mode']}-{plan['run_id']}-s*.pt"),key=lambda p:p.stat().st_mtime_ns,reverse=True)
        for older in owned[keep:]:
            older.unlink();older.with_suffix('.json').unlink(missing_ok=True)
    return path


def run_replica(plan,runtime):
    """Runtime owns XLA collectives; CPU tests exercise the exact objectives/export."""
    options=plan['config']
    device=runtime.device
    rank,world=runtime.rank,runtime.world_size
    if plan.get('expected_replicas',0) and world!=plan['expected_replicas']:
        raise ValueError(f"Expected {plan['expected_replicas']} TPU replicas, found {world}. Check the Kaggle accelerator setting.")
    tokenizer=Tokenizer(state=plan['tokenizer_state'])
    model=initialize_model(plan['model_config'],plan['seed'],device)
    optimizer=runtime.optimizer(model.parameters(),options['learning_rate'])
    initial_step=0
    if plan.get('resume'):
        payload=torch.load(plan['resume'],map_location='cpu',weights_only=True,mmap=True)
        model.load_state_dict(payload['model_state_dict'])
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        for group in optimizer.param_groups:
            group.update(lr=options['learning_rate'],fused=False,foreach=False,capturable=False)
        for state in optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value):state[key]=value.to(device)
        initial_step=payload['step']
        del payload
    runtime.broadcast(model)
    runtime.seed(plan['seed']+rank)
    reference=copy.deepcopy(model).eval().requires_grad_(False) if options['mode']=='dpo' else None
    corpora=[FileCorpus(Path(path)) for path,weight in plan['sources'] if weight>0]
    pairs=PairPool(plan['feedback'],corpora,'sft' if options['mode']=='sft' else 'preference') if options['mode']!='pretrain' else []
    texts=[(FileCorpus(Path(path)),weight) for path,weight in plan['sources']]
    texts.extend((text,weight) for text,weight in plan.get('inline_sources',[]))
    needs_mixer=options['mode']=='pretrain' or (options['mode']=='sft' and options['base_mix']>0)
    mixer=WeightedMixer(texts,tokenizer,model.config.block_size,'cpu') if needs_mixer else None
    rng=random.Random(plan['seed']+rank*1_000_003+initial_step)
    val_rng=random.Random(plan['seed']+rank*1_000_003+91)
    accumulation=options['gradient_accumulation']
    target=initial_step+options['max_steps']
    val_loss=None
    last_checkpoint=None
    if rank==0:
        print(json.dumps({'device':'xla:TPU','replicas':world,'precision':options['precision'],
                          'parameters':parameter_count(model.config),'batch_per_core':options['batch_size'],
                          'effective_batch':options['batch_size']*accumulation*world}),flush=True)
    for step in range(initial_step+1,target+1):
        began=time.perf_counter()
        model.train();optimizer.zero_grad(set_to_none=True)
        total=torch.zeros((),device=device)
        tokens=0
        for micro in range(accumulation):
            # All ranks follow the same SFT base-data branch; samples remain independent.
            use_base=(options['mode']=='sft' and mixer is not None and
                      random.Random(plan['seed']+step*accumulation+micro).random()<options['base_mix'])
            with runtime.autocast(options['precision']):
                loss,count=objective(model,tokenizer,options['mode'],mixer,pairs,reference,options,rng,device,use_base)
            (loss/accumulation).backward()
            total=total+loss.detach().float()/accumulation
            tokens+=count
        runtime.reduce_gradients(optimizer)  # Average first, then clip the shared gradient.
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,foreach=False)
        mean_loss=runtime.mean(total)
        found_inf=(~(torch.isfinite(norm)&torch.isfinite(mean_loss))).float()
        runtime.step_optimizer(optimizer,found_inf)
        runtime.sync()
        if found_inf.cpu().item():
            raise ValueError('Non-finite TPU loss or gradient; update skipped. Lower learning rate or inspect data.')
        elapsed=max(time.perf_counter()-began,1e-9)
        evaluate=step%options['eval_interval']==0 or step==target
        if evaluate and options['mode']=='pretrain':
            model.eval()
            with torch.no_grad(),runtime.autocast(options['precision']):
                losses=[]
                for _ in range(3):
                    x,y=fixed_pretrain_batch(mixer,options['batch_size'],val_rng,validation=True)
                    losses.append(model(x.to(device),y.to(device))['loss'].float())
                validation=runtime.mean(torch.stack(losses).mean())
            runtime.sync()
            val_loss=validation.cpu().item()
            if not math.isfinite(val_loss):raise ValueError('Non-finite TPU validation loss.')
        if step%plan.get('log_every',10)==0 or evaluate or step==initial_step+1:
            value=mean_loss.cpu().item()
            if not math.isfinite(value):raise ValueError('Non-finite TPU training loss.')
            if rank==0:
                metric={'step':step,'target_step':target,'train_loss':value,'val_loss':val_loss,
                        'tokens_per_sec':tokens*world/elapsed,'mode':options['mode'],'replicas':world}
                print(json.dumps(metric),flush=True)
                with (Path(plan['home'])/'tpu-metrics.jsonl').open('a',encoding='utf-8') as output:
                    output.write(json.dumps(metric)+'\n')
        if evaluate:
            runtime.barrier('before-save-'+str(step))
            error=''
            if rank==0:
                try:last_checkpoint=export_checkpoint(model,tokenizer,optimizer,plan,step,val_loss,world)
                except Exception as exc:error=type(exc).__name__+': '+str(exc)
            # Share save errors before raising, so other ranks do not wait forever.
            runtime.barrier('after-save-'+str(step),error)
    return last_checkpoint
