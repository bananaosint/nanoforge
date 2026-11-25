"""Single-owner background training with explicit CE, SFT, and DPO objectives."""
import copy
from contextlib import nullcontext
import json
import math
import random
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import GPT, ModelConfig, resolve_model, parameter_count, MAX_PARAMETERS, detect_device, device_name
from .tokenizer import Tokenizer, format_prompt, END
from .dataset import WeightedMixer, completion_batch, completion_log_probability
from .corpus import FileCorpus, vocabulary_text, PairPool
from .parallel import ReplicaLoss, reduced_loss
from .xla_training import fixed_completion_batch
from ..storage import now

DEFAULT_PROMPTS = ['First Citizen:', 'To be, or not to be', 'The meaning of life is']


class Trainer:
    def __init__(self, datasets, store, checkpoint_dir, device=None):
        self.datasets, self.store = datasets, store
        self.directory = Path(checkpoint_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device = device or detect_device()
        # Tiny CPU models often run slower with dozens of OpenMP threads.
        if self.device == 'cpu':
            torch.set_num_threads(min(4, torch.get_num_threads()))
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.thread = None
        self.model = self.tokenizer = self.optimizer = None
        self.amp_enabled = False
        self.precision = 'float32'
        self.allow_tf32 = False
        self.parallel = None
        self.training_gpus = 1
        self.scaler = None
        self.overflow_retries = 0
        self.step = 0
        self.target_step = None
        self.run_id = None
        self.checkpoint = None
        self._branch_pending = False
        self.mode = 'pretrain'
        self.metrics = []
        self.config = {}
        self.state = 'idle'
        self.error = None
        self.val_loss = None
        self.prompts = store.setting('prompts', DEFAULT_PROMPTS)

    def _configure_precision(self, config, scaler_state=None):
        requested = config.get('precision','auto')
        if requested == 'bfloat16':
            raise ValueError('TPU BF16 training uses python -m backend.train_xla. Use auto or float32 in the local studio.')
        if requested not in ('auto','float32','float16'):
            raise ValueError('Precision must be auto, float32, or float16.')
        cuda = str(self.device).split(':')[0] == 'cuda'
        if requested == 'float16' and not cuda:
            raise ValueError('FP16 training requires CUDA. Use auto or float32 on CPU/MPS/XPU.')
        self.allow_tf32 = bool(cuda and config.get('allow_tf32',False)
                               and torch.cuda.get_device_capability(self.device)[0] >= 8)
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32
        enabled = cuda and requested != 'float32'
        if self.scaler is None or enabled != self.amp_enabled:
            self.scaler = torch.amp.GradScaler('cuda',enabled=enabled,init_scale=1024.0)
        self.amp_enabled = enabled
        self.precision = 'float16' if enabled else 'float32'
        if enabled and scaler_state:
            self.scaler.load_state_dict(scaler_state)

    def _configure_parallel(self,options,reference=None,validate_only=False):
        requested=options.get('multi_gpu',False)
        if requested and (not str(self.device).startswith('cuda') or torch.cuda.device_count()<2):
            raise ValueError('Two-GPU training requires at least two CUDA GPUs. Disable Use both GPUs on this runtime.')
        self.parallel=None
        self.training_gpus=2 if requested else 1
        if requested:
            if options.get('batch_size',8)<2:
                raise ValueError('Use a batch size of at least 2 to train on both GPUs.')
            if validate_only:return
            replica=ReplicaLoss(self.model,self.precision,reference,options.get('beta',0.1))
            self.parallel=torch.nn.DataParallel(replica,device_ids=[0,1],output_device=0)

    def _parallel_completion_loss(self,pairs,mode,options):
        selected=random.choices(pairs,k=options.get('batch_size',8))
        if mode=='sft':
            x,y=fixed_completion_batch(self.tokenizer,selected,self.model.config.block_size)
            return reduced_loss(self.parallel(x.to(self.device),y.to(self.device))),x.numel()
        self.model.eval()
        chosen,rejected=[],[]
        for pair in selected:
            a,b=pair['response_a'],pair['response_b']
            win,lose=(a,b) if pair['winner']=='a' else (b,a)
            chosen.append({'prompt':pair['prompt'],'corrected_output':win})
            rejected.append({'prompt':pair['prompt'],'corrected_output':lose})
        win=fixed_completion_batch(self.tokenizer,chosen,self.model.config.block_size)
        lose=fixed_completion_batch(self.tokenizer,rejected,self.model.config.block_size)
        tensors=[tensor.to(self.device) for tensor in (*win,*lose)]
        return reduced_loss(self.parallel(*tensors)),win[0].numel()+lose[0].numel()

    def _autocast(self):
        return torch.autocast(device_type='cuda',dtype=torch.float16) if self.amp_enabled else nullcontext()

    @staticmethod
    def _cpu_tree(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value,dict):
            return {key:Trainer._cpu_tree(item) for key,item in value.items()}
        if isinstance(value,list):
            return [Trainer._cpu_tree(item) for item in value]
        if isinstance(value,tuple):
            return tuple(Trainer._cpu_tree(item) for item in value)
        return value

    def _model_state_for_checkpoint(self):
        # Preserve tied embedding storage while copying CUDA parameters to CPU.
        copies={}
        result={}
        for name,tensor in self.model.state_dict().items():
            key=(str(tensor.device),tensor.data_ptr(),tuple(tensor.shape),tuple(tensor.stride()),str(tensor.dtype))
            if key not in copies:
                copies[key]=tensor.detach().float().cpu() if tensor.is_floating_point() else tensor.detach().cpu()
            result[name]=copies[key]
        return result

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def require_idle(self):
        if self.busy():
            raise ValueError('Stop the active run before changing models or checkpoints.')

    def create(self, config):
        with self.lock:
            self.require_idle()
            tokenizer = self._new_tokenizer(config)
            model_config = resolve_model(config.get('preset', 'Nano'), config.get('model', {}), tokenizer.vocab_size)
            self._configure_precision(config)
            self.parallel = None
            self.training_gpus = 1
            candidate = GPT(model_config).to(self.device)
            self.model, self.tokenizer = candidate, tokenizer
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.get('learning_rate', 0.0003), weight_decay=0.1, fused=str(self.device).startswith('cuda'), foreach=False if str(self.device).split(':')[0] in ('xpu','cuda') else None)
            self.config = config
            self.step, self.val_loss, self.checkpoint = 0, None, None
            self.target_step = None
            self._branch_pending = False
            self.mode, self.state, self.error = 'pretrain', 'ready', None
            self._new_run(config)
            return self.status()

    def _new_tokenizer(self, config):
        texts = self._texts(config.get('datasets', []))
        kind = config.get('tokenizer', 'char')
        return Tokenizer(kind, vocabulary_text(texts) if kind == 'char' else '')

    def preview_model(self, config):
        """Count/size a new model without allocating tensors or changing the run."""
        tokenizer = self._new_tokenizer(config)
        model = resolve_model(config.get('preset', 'Nano'), config.get('model', {}),
                              tokenizer.vocab_size, config.get('target_parameters'))
        count = parameter_count(model)
        return {'model_config': asdict(model), 'parameters': count, 'max_parameters': MAX_PARAMETERS,
                'weights_bytes': count * 4, 'training_state_bytes': count * 16}

    def _new_run(self, config):
        self.run_id = uuid.uuid4().hex[:12]
        self.metrics = []
        self.store.execute('INSERT INTO runs VALUES(?,?,?)', (self.run_id, json.dumps(config), now()))

    def _texts(self, sources):
        texts = []
        for source in sources:
            if source['weight'] <= 0:
                continue
            if source['name'] == 'hitl':
                text = '\n'.join(format_prompt(p['prompt']) + p['corrected_output'] + END for p in self.store.feedback('sft'))
            else:
                text = FileCorpus(self.datasets.path(source['name']))
            if (isinstance(text, str) and not text.strip()) or (isinstance(text, FileCorpus) and text.path.stat().st_size == 0):
                raise ValueError(f"Dataset {source['name']} is empty.")
            texts.append((text, source['weight']))
        if not texts:
            raise ValueError('Select at least one dataset with a positive weight.')
        return texts

    def start(self, config):
        with self.lock:
            if self.model is None:
                raise ValueError('Create a model run or select a checkpoint first.')
            if self.busy():
                raise ValueError('A run is already active. Use Resume for a paused run.')
            options = {**self.config, **config}
            mode = options.get('mode', 'pretrain')
            self._configure_precision(options)
            self._configure_parallel(options,validate_only=True)
            mixer = None
            pairs = []
            if mode == 'pretrain':
                mixer = WeightedMixer(self._texts(options.get('datasets', [])), self.tokenizer, self.model.config.block_size, self.device)
            else:
                corpora = [FileCorpus(self.datasets.path(s['name'])) for s in options.get('datasets', []) if s['weight'] > 0 and s['name'] != 'hitl']
                pairs = PairPool(self.store.feedback('sft' if mode == 'sft' else 'preference'), corpora, mode)
                if not pairs:
                    raise ValueError('Collect feedback pairs before starting fine-tuning.')
                if mode == 'sft' and options.get('base_mix', 0) > 0:
                    mixer = WeightedMixer(self._texts(options.get('datasets', [])), self.tokenizer, self.model.config.block_size, self.device)
            if self._branch_pending:
                options['parent_checkpoint'] = self.checkpoint
            if self._branch_pending or mode != self.mode or self.state in ('completed', 'stopped', 'error'):
                self._new_run(options)
            self._branch_pending = False
            # A DPO reference is fixed for this entire fine-tuning pass.
            reference = copy.deepcopy(self.model).eval().requires_grad_(False) if mode == 'dpo' else None
            self._configure_parallel(options,reference)
            if mode != 'pretrain':
                self.val_loss = None
            self.mode = mode
            self.target_step = self.step + options.get('max_steps', 250)
            self.config = options
            for group in self.optimizer.param_groups:
                group['lr'] = options.get('learning_rate', 0.0003)
            self.stop_event.clear()
            self.pause_event.clear()
            self.error, self.state = None, 'running'
            self.thread = threading.Thread(target=self._train, args=(options, mixer, pairs, reference), daemon=False, name='nanoforge-trainer')
            self.thread.start()
            return self.status()

    def _train(self, options, mixer, pairs, reference):
        try:
            if self.mode == 'pretrain':
                with self.lock:
                    self._evaluate(mixer)
            target = self.target_step
            interval = options.get('eval_interval', 50)
            accumulation = options.get('gradient_accumulation', 1)
            while self.step < target and not self.stop_event.is_set():
                if self.pause_event.is_set():
                    self.stop_event.wait(0.1)
                    continue
                began = time.perf_counter()
                with self.lock:
                    self.model.train()
                    self.optimizer.zero_grad(set_to_none=True)
                    total, tokens = 0.0, 0
                    for _ in range(accumulation):
                        with self._autocast():
                            if self.mode == 'pretrain' or (self.mode == 'sft' and mixer and random.random() < options.get('base_mix', 0.1)):
                                x, y = mixer.batch(options.get('batch_size', 8))
                                loss = reduced_loss(self.parallel(x,y)) if self.parallel is not None else self.model(x, y)['loss']
                                tokens += x.numel()
                            elif self.parallel is not None:
                                loss,count=self._parallel_completion_loss(pairs,self.mode,options)
                                tokens+=count
                            elif self.mode == 'sft':
                                losses = []
                                for pair in random.choices(pairs, k=options.get('batch_size', 8)):
                                    x, y = completion_batch(self.tokenizer, pair['prompt'], pair['corrected_output'], self.model.config.block_size, self.device)
                                    losses.append(self.model(x, y)['loss'])
                                    tokens += x.numel()
                                loss = torch.stack(losses).mean()
                            else:
                                losses = []
                                # Disable dropout for policy and reference log-probability comparison.
                                self.model.eval()
                                for pair in random.choices(pairs, k=options.get('batch_size', 8)):
                                    winner = pair['winner']
                                    chosen, rejected = (pair['response_a'], pair['response_b']) if winner == 'a' else (pair['response_b'], pair['response_a'])
                                    win = completion_log_probability(self.model, self.tokenizer, pair['prompt'], chosen, self.device)
                                    lose = completion_log_probability(self.model, self.tokenizer, pair['prompt'], rejected, self.device)
                                    with torch.no_grad():
                                        ref_win = completion_log_probability(reference, self.tokenizer, pair['prompt'], chosen, self.device)
                                        ref_lose = completion_log_probability(reference, self.tokenizer, pair['prompt'], rejected, self.device)
                                    loss_dpo = -F.logsigmoid(options.get('beta', 0.1) * ((win - ref_win) - (lose - ref_lose)))
                                    losses.append(loss_dpo.mean())
                                    tokens += min(self.model.config.block_size, len(self.tokenizer.encode(chosen))) + min(self.model.config.block_size, len(self.tokenizer.encode(rejected)))
                                loss = torch.stack(losses).mean()
                        loss_value=loss.detach().float().item()
                        if not math.isfinite(loss_value):
                            raise ValueError('Non-finite loss detected. Try a smaller learning rate or inspect the dataset.')
                        self.scaler.scale(loss / accumulation).backward()
                        total += loss_value / accumulation
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0,error_if_nonfinite=not self.amp_enabled)
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.amp_enabled and self.scaler.get_scale() < scale_before:
                        self.overflow_retries += 1
                        if self.overflow_retries > 20:
                            raise ValueError('Repeated FP16 gradient overflow. Lower the learning rate or use float32.')
                        continue  # Overflowed updates do not count as training steps.
                    self.overflow_retries = 0
                    if str(self.device).startswith('cuda'):
                        torch.cuda.synchronize()
                    self.step += 1
                    elapsed = max(time.perf_counter() - began, 1e-9)
                    metric = {'step': self.step, 'train_loss': total, 'val_loss': None, 'steps_per_sec': 1 / elapsed,
                              'tokens_per_sec': tokens / elapsed, 'mode': self.mode}
                    if self.mode == 'pretrain' and (self.step % interval == 0 or self.step == target):
                        self._evaluate(mixer)
                        metric['val_loss'] = self.val_loss
                    self.metrics.append(metric)
                    self.metrics = self.metrics[-3000:]
                    self.store.execute('INSERT INTO metrics(run_id,payload) VALUES(?,?)', (self.run_id, json.dumps(metric)))
                    if self.step % interval == 0:
                        self._save()
            with self.lock:
                self._save()
                self.state = 'stopped' if self.stop_event.is_set() else 'completed'
        except Exception as exc:
            with self.lock:
                self.state, self.error = 'error', str(exc)
        finally:
            if self.model:
                self.model.eval()

    def _evaluate(self, mixer):
        self.model.eval()
        with torch.no_grad(), self._autocast():
            losses = [self.model(*mixer.batch(self.config.get('batch_size', 8), validation=True))['loss'].item() for _ in range(3)]
            self.val_loss = sum(losses) / len(losses)
            if not math.isfinite(self.val_loss):
                raise ValueError('Non-finite validation loss.')
            for prompt in self.prompts:
                result = self._generate(prompt, max_new_tokens=40, telemetry=False, seed=7)
                self.store.execute('INSERT INTO timeline(run_id,step,prompt,output) VALUES(?,?,?,?)', (self.run_id, self.step, prompt, result['text']))

    def pause(self):
        if self.state != 'running':
            raise ValueError('Only a running model can be paused.')
        self.pause_event.set()
        self.state = 'paused'
        return self.status()

    def resume(self):
        if self.state != 'paused':
            raise ValueError('No paused run to resume.')
        self.pause_event.clear()
        self.state = 'running'
        return self.status()

    def stop(self):
        self.stop_event.set()
        self.pause_event.clear()
        if self.busy():
            self.state = 'stopping'
        return self.status()

    def shutdown(self):
        self.stop()
        if self.thread:
            self.thread.join()  # Finish the current atomic step and checkpoint before closing SQLite.

    def _save(self, version_tag=None):
        version_tag = version_tag or f'{self.mode}-{self.run_id}-s{self.step}'
        name = version_tag + '.pt'
        path = self.directory / name
        payload = {'model_config': asdict(self.model.config), 'tokenizer_state': self.tokenizer.state(),
                   'model_state_dict': self._model_state_for_checkpoint(), 'optimizer_state_dict': self._cpu_tree(self.optimizer.state_dict()),
                   'step': self.step, 'val_loss': self.val_loss, 'version_tag': version_tag,
                   'training_config': self.config, 'mode': self.mode, 'run_id': self.run_id,
                   'precision':self.precision,'scaler_state_dict':self.scaler.state_dict() if self.scaler else {},
                   'torch_version':str(torch.__version__),'checkpoint_format_version':1}
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
        metadata = {k: payload[k] for k in ('model_config', 'step', 'val_loss', 'version_tag', 'mode', 'run_id')}
        metadata.update(name=name, created_at=now(), parameters=sum(p.numel() for p in self.model.parameters()))
        sidecar = path.with_suffix('.json')
        temp_sidecar = sidecar.with_suffix('.json.tmp')
        temp_sidecar.write_text(json.dumps(metadata))
        temp_sidecar.replace(sidecar)
        self.checkpoint = name
        return metadata

    def save(self):
        with self.lock:
            if self.model is None:
                raise ValueError('No model to save.')
            return self._save()

    def checkpoints(self):
        return sorted([json.loads(p.read_text()) for p in self.directory.glob('*.json')], key=lambda r: r['created_at'], reverse=True)

    def checkpoint_path(self, name):
        if Path(name).name != name or not name.endswith('.pt') or not (self.directory / name).is_file():
            raise ValueError('Checkpoint not found.')
        return self.directory / name

    def load(self, name):
        with self.lock:
            self.require_idle()
            # Files are produced locally by NanoForge. weights_only avoids arbitrary pickle code.
            payload = torch.load(self.checkpoint_path(name), map_location=self.device, weights_only=True)
            model = GPT(ModelConfig(**payload['model_config'])).to(self.device)
            model.load_state_dict(payload['model_state_dict'])
            self.parallel = None
            self.training_gpus = 1
            self.model = model.eval()
            self.tokenizer = Tokenizer(state=payload['tokenizer_state'])
            self.optimizer = torch.optim.AdamW(model.parameters(), fused=str(self.device).startswith('cuda'), foreach=False if str(self.device).split(':')[0] in ('xpu','cuda') else None)
            self.optimizer.load_state_dict(payload['optimizer_state_dict'])
            for group in self.optimizer.param_groups:
                group['fused'] = str(self.device).startswith('cuda')
                if str(self.device).split(':')[0] in ('xpu','cuda'):
                    group['foreach'] = False
            self.step, self.val_loss, self.config = payload['step'], payload['val_loss'], payload.get('training_config', {})
            self.run_id, self.mode = payload.get('run_id'), payload.get('mode', 'pretrain')
            self._configure_precision({**self.config,'precision':'auto'},payload.get('scaler_state_dict'))
            self.metrics = [json.loads(row['payload']) for row in self.store.rows('SELECT payload FROM metrics WHERE run_id=? ORDER BY id DESC LIMIT 3000', (self.run_id,))][::-1]
            self.checkpoint, self.state, self.error = name, 'ready', None
            self.target_step = None
            self._branch_pending = True
            return self.status()

    def _generate(self, prompt, max_new_tokens=64, temperature=0.8, top_k=40, telemetry=True, seed=42, model=None, tokenizer=None, mode=None):
        model, tokenizer = model or self.model, tokenizer or self.tokenizer
        if model is None:
            raise ValueError('Create a model or select a checkpoint before generating.')
        if not prompt.strip():
            raise ValueError('Enter a nonempty prompt.')
        prefix = format_prompt(prompt) if (mode or self.mode) in ('sft', 'dpo') else prompt
        ids = torch.tensor([tokenizer.encode(prefix)], device=self.device)
        with self._autocast():
            result = model.generate(ids, max_new_tokens, temperature, top_k, telemetry, seed)
        generated = result['ids'][0, ids.size(1):].tolist()
        for item in result['steps']:
            item['token'] = tokenizer.decode_token(item['token_id'])
            for candidate in item['top5']:
                candidate['token'] = tokenizer.decode_token(candidate['id'])
        return {'text': tokenizer.decode(generated), 'tokens': [tokenizer.decode_token(i) for i in generated],
                'steps': result['steps'], 'attention': [a[0].cpu().tolist() for a in result['attention']],
                'context_tokens': [tokenizer.decode_token(i) for i in result['context_ids']],
                'checkpoint': self.checkpoint, 'seed': seed}

    def generate(self, **kwargs):
        with self.lock:
            return self._generate(**kwargs)

    def generate_from(self, checkpoint=None, **kwargs):
        with self.lock:
            if not checkpoint or checkpoint == self.checkpoint:
                result = self._generate(**kwargs)
            else:
                payload = torch.load(self.checkpoint_path(checkpoint), map_location=self.device, weights_only=True)
                model = GPT(ModelConfig(**payload['model_config'])).to(self.device).eval()
                model.load_state_dict(payload['model_state_dict'])
                result = self._generate(**kwargs, model=model, tokenizer=Tokenizer(state=payload['tokenizer_state']), mode=payload.get('mode', 'pretrain'))
            result['checkpoint'] = checkpoint or self.checkpoint
            return result

    def imported_pairs(self):
        manifest = self.datasets.directory / '.imports/jobs.json'
        if not manifest.exists():
            return {}
        try:
            jobs = json.loads(manifest.read_text())
            result={}
            for key,job in jobs.items():
                result.update(job.get('previous_outputs',{}))
                if job.get('state') in ('completed','sampled') and job.get('output','').endswith('.jsonl'):
                    result[job['output']]={'sft':job.get('sft_pairs',job.get('rows') or 0),'dpo':job.get('dpo_pairs',(job.get('rows') or 0) if key=='hh-rlhf' else 0)}
            return result
        except (ValueError, OSError):
            return {}

    def status(self):
        return {'imported_pairs': self.imported_pairs(), 'state': self.state, 'device': self.device, 'device_name': device_name(self.device), 'precision':self.precision, 'allow_tf32':self.allow_tf32, 'cuda_device_count':torch.cuda.device_count() if str(self.device).startswith('cuda') else 0, 'training_gpus':self.training_gpus, 'step': self.step, 'target_step': self.target_step, 'mode': self.mode,
                'run_id': self.run_id, 'checkpoint': self.checkpoint, 'error': self.error, 'val_loss': self.val_loss,
                'parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
                'model_config': asdict(self.model.config) if self.model else None, 'config': self.config,
                'metrics': self.metrics[-3000:], 'prompts': self.prompts}
