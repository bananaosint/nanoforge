"""Single-owner background training with explicit CE, SFT, and DPO objectives."""
import copy
from contextlib import ExitStack, nullcontext
from functools import lru_cache, partial
import json
import math
import random
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
import chess.engine
import torch
from torch.nn import functional as F
from .model import GPT, ModelConfig, resolve_model, parameter_count, active_parameter_count, export_config, MAX_PARAMETERS, detect_device, device_name
from .tokenizer import Tokenizer, format_prompt, END
from .dataset import WeightedMixer, completion_batch, completion_log_probability
from .streaming import StreamingMixer, StreamingError, is_stream, stream_definition
from .corpus import FileCorpus, vocabulary_text, PairPool
from .parallel import ReplicaLoss, reduced_loss
from .xla_training import fixed_completion_batch
from ..storage import now
from ..chess_eval import ChessReports, MAX_NEW_TOKENS, evaluate_checkpoint
from ..chess_pipeline import ChessDatasets, ChessExampleStream, STRUCTURAL_TOKENS, format_example, open_stockfish
from ..gemini_client import GeminiService
from ..watchdog import CheckpointWatchdog

CHESS_CONTEXT = 512

DEFAULT_PROMPTS = ['First Citizen:', 'To be, or not to be', 'The meaning of life is']


class Trainer:
    def __init__(self, datasets, store, checkpoint_dir, device=None, gemini=None):
        self.datasets, self.store = datasets, store
        self.directory = Path(checkpoint_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.chess_datasets = ChessDatasets(datasets.directory / '.chess', store)
        self.chess_reports = ChessReports(self.directory)
        self.chess_evaluating_step = None
        self.device = device or detect_device()
        # Tiny CPU models often run slower with dozens of OpenMP threads.
        if self.device == 'cpu':
            torch.set_num_threads(min(4, torch.get_num_threads()))
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.thread = None
        self._synchronous_running = False
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
        self.gemini=gemini or GeminiService(store)
        self.watchdog=CheckpointWatchdog(self,self.gemini)
        self.stream_state = None
        self.stream_mixer = None

    def _configure_precision(self, config, scaler_state=None):
        requested = config.get('precision','auto')
        if requested not in ('auto','float32','float16','bfloat16'):
            raise ValueError('Unknown compute precision.')
        cuda = str(self.device).split(':')[0] == 'cuda'
        if requested == 'float16' and not cuda:
            raise ValueError('FP16 training requires CUDA. Use auto or float32 on CPU/MPS/XPU.')
        if requested=='bfloat16' and not (cuda and torch.cuda.get_device_capability(self.device)[0]>=8 and torch.cuda.is_bf16_supported()):
            raise ValueError('CUDA BF16 needs native BF16 GPU support. For TPU use python -m backend.train_xla; use auto on CPU.')
        self.allow_tf32 = bool(cuda and config.get('allow_tf32',False)
                               and torch.cuda.get_device_capability(self.device)[0] >= 8)
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32
        enabled = cuda and requested not in ('float32','bfloat16')
        if self.scaler is None or enabled != self.amp_enabled:
            self.scaler = torch.amp.GradScaler('cuda',enabled=enabled,init_scale=1024.0)
        self.amp_enabled = enabled
        self.precision = 'bfloat16' if requested=='bfloat16' else ('float16' if enabled else 'float32')
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
            replica=ReplicaLoss(self.model,self.precision,reference,options.get('beta',0.1),
                                telemetry=self.model.config.architecture=='moe' and options.get('mode','pretrain')=='pretrain')
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
        return torch.autocast(device_type='cuda',dtype=torch.bfloat16 if self.precision=='bfloat16' else torch.float16) if str(self.device).startswith('cuda') and self.precision!='float32' else nullcontext()

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
        return self._synchronous_running or (self.thread is not None and self.thread.is_alive())

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
            self.mode = 'chess' if config.get('mode') == 'chess' else 'pretrain'
            self.state, self.error = 'ready', None
            self.watchdog.reset()
            self.stream_state = None
            self._new_run(config)
            return self.status()

    def _new_tokenizer(self, config):
        if config.get('mode') == 'chess':
            return Tokenizer('magic8k', structural_tokens=(*STRUCTURAL_TOKENS, '<|end|>'))
        local, remote = self._partition_sources(config.get('datasets', []))
        kind = config.get('tokenizer', 'char')
        if remote and kind == 'char':
            raise ValueError('Streaming requires Magic 8k or GPT-2 BPE. Character tokenization scans the entire corpus.')
        texts = self._texts(local) if local or not remote else []
        return Tokenizer(kind, vocabulary_text(texts) if kind == 'char' else '')

    @staticmethod
    def _partition_sources(sources):
        local, remote = [], []
        for source in sources:
            if source['weight'] <= 0:
                continue
            if is_stream(source['name']):
                stream_definition(source['name'])
                remote.append(source)
            else:
                local.append(source)
        return local, remote

    def _mixer(self, local, remote):
        block_size = self.model.config.block_size
        if not remote:
            return WeightedMixer(self._texts(local), self.tokenizer, block_size, self.device)
        if self.tokenizer.kind == 'char':
            raise ValueError('Streaming requires a model with Magic 8k or GPT-2 BPE.')
        if block_size > 8192:
            raise ValueError('Streaming supports context windows up to 8192 tokens.')
        return StreamingMixer(remote, self._texts(local) if local else [],
                              self.tokenizer, block_size, self.device, self.stop_event, self.stream_state)

    def preview_model(self, config):
        """Count/size a new model without allocating tensors or changing the run."""
        tokenizer = self._new_tokenizer(config)
        model = resolve_model(config.get('preset', 'Nano'), config.get('model', {}),
                              tokenizer.vocab_size, config.get('target_parameters'))
        count = parameter_count(model)
        return {'model_config': asdict(model), 'parameters': count, 'max_parameters': MAX_PARAMETERS,
                'active_parameters':active_parameter_count(model), 'weights_bytes': count * 4, 'training_state_bytes': count * 16}

    def _new_run(self, config):
        self.run_id = uuid.uuid4().hex[:12]
        self.metrics = []
        self.store.execute('INSERT INTO runs VALUES(?,?,?)', (self.run_id, json.dumps(config), now()))

    def is_chess_run(self, run_id):
        rows = self.store.rows('SELECT config FROM runs WHERE id=?', (run_id,))
        return bool(rows and json.loads(rows[0]['config']).get('mode') == 'chess')

    def run_history(self, mode):
        rows = self.store.rows('SELECT * FROM runs ORDER BY created_at DESC')
        runs = [{**row, 'config': json.loads(row['config'])} for row in rows]
        return [{**row, 'mode': mode} for row in runs if (row['config'].get('mode') == 'chess') == (mode == 'chess')]

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

    def start_chess(self, config, checkpoint=None, engine_path=None):
        examples = self.chess_datasets.examples(config['dataset'])
        with self.lock:
            self.require_idle()
            options = {**config, 'mode': 'chess', 'tokenizer': 'magic8k',
                       'model': {'block_size': CHESS_CONTEXT}, 'watchdog_enabled': False,
                       'datasets': [], 'base_mix': 0}
            self._configure_precision(options)
            if checkpoint:
                self.load(checkpoint, expected_mode='chess')
                if self.model.config.block_size < CHESS_CONTEXT:
                    raise ValueError('Chess training requires a context of at least 512 tokens.')
                options['preset'] = self.config.get('preset', options['preset'])
                options['model'] = self.config.get('model', {'block_size': self.model.config.block_size})
            else:
                self.create(options)
            return self.start(options, chess_examples=examples, chess_engine_path=engine_path)

    def require_text(self):
        if self.mode == 'chess':
            raise ValueError('Create a text run or select a text checkpoint first.')

    def start(self, config, *, background=True, evaluate=True, time_budget_seconds=None, chess_examples=None, chess_engine_path=None):
        with self.lock:
            if self.model is None:
                raise ValueError('Create a model run or select a checkpoint first.')
            if self.busy():
                raise ValueError('A run is already active. Use Resume for a paused run.')
            options = {**self.config, **config}
            mode = options.get('mode', 'pretrain')
            if (mode == 'chess') != (self.mode == 'chess') or (mode == 'chess' and chess_examples is None):
                raise ValueError('Start chess training through the Chess Mode run endpoint.')
            self._configure_precision(options)
            self._configure_parallel(options,validate_only=True)
            mixer = None
            pairs = []
            local, remote = self._partition_sources(options.get('datasets', []))
            if mode == 'chess':
                pairs = chess_examples
            elif mode == 'pretrain':
                mixer = self._mixer(local, remote)
            else:
                corpora = [FileCorpus(self.datasets.path(s['name'])) for s in local if s['name'] != 'hitl']
                pairs = PairPool(self.store.feedback('sft' if mode == 'sft' else 'preference'), corpora, mode)
                if not pairs:
                    raise ValueError('Collect feedback or import a Local cache of a paired dataset before fine-tuning.')
                if mode == 'sft' and options.get('base_mix', 0) > 0:
                    mixer = self._mixer(local, remote)
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
            self.watchdog.reset(self.watchdog.state())
            self.stream_mixer = mixer if isinstance(mixer, StreamingMixer) else None
            if self.stream_mixer:
                self.stream_state = self.stream_mixer.state()
            if background:
                self.thread = threading.Thread(target=self._train, args=(options, mixer, pairs, reference, evaluate, time_budget_seconds, chess_engine_path), daemon=False, name='nanoforge-trainer')
                self.thread.start()
                return self.status()
            self.thread = None
            self._synchronous_running = True
        self._train(options, mixer, pairs, reference, evaluate, time_budget_seconds, chess_engine_path)
        return self.status()

    def _train(self, options, mixer, pairs, reference, evaluate=True, time_budget_seconds=None, chess_engine_path=None):
        deadline=None if time_budget_seconds is None else time.monotonic()+max(0,time_budget_seconds)
        resources = ExitStack()
        chess_stream = None
        try:
            if self.mode == 'chess':
                dataset = options['dataset']
                if dataset.startswith('hf:'):
                    previous = self.stream_state or {}
                    state = previous if previous.get('dataset') == dataset else None
                    self.stream_state = state
                    chess_stream = ChessExampleStream(dataset.removeprefix('hf:'), self.stop_event.is_set, state)
                    resources.callback(chess_stream.close)
                else:
                    self.stream_state = None
                engine_resources = resources.enter_context(ExitStack())
                engine = engine_resources.enter_context(open_stockfish(chess_engine_path))

                @lru_cache(maxsize=2048)
                def chess_pair(example):
                    nonlocal engine
                    try:
                        context = example.context(engine)
                    except (OSError, chess.engine.EngineError, TimeoutError):
                        if engine is None:
                            raise
                        engine_resources.close()
                        engine = None
                        context = example.context()
                    prompt, completion = format_example(context, example.explanation, example.themes)
                    return {'prompt': prompt, 'corrected_output': completion}

            if evaluate and self.mode == 'pretrain':
                with self.lock:
                    self._evaluate(mixer)
                    if self.stream_mixer:
                        self.stream_state = self.stream_mixer.state()
                    if self.watchdog.enabled():self._save()
            target = self.target_step
            interval = options.get('eval_interval', 50)
            accumulation = options.get('gradient_accumulation', 1)
            while self.step < target and not self.stop_event.is_set():
                if deadline is not None and time.monotonic()>=deadline:
                    self.stop_event.set();break
                if self.pause_event.is_set():
                    self.stop_event.wait(0.1)
                    continue
                began = time.perf_counter()
                with self.lock:
                    self.model.train()
                    self.optimizer.zero_grad(set_to_none=True)
                    total, tokens = 0.0, 0
                    balances=[];zs=[];loads=[];routing_weights=[]
                    for _ in range(accumulation):
                        with self._autocast():
                            if self.mode == 'pretrain' or (self.mode == 'sft' and mixer and random.random() < options.get('base_mix', 0.1)):
                                x, y = mixer.batch(options.get('batch_size', 8))
                                if self.parallel is not None:
                                    output=self.parallel(x,y);loss=reduced_loss(output)
                                    if len(output)>2:
                                        weight=output[5].sum()
                                        balances.append(output[2].sum()/weight);zs.append(output[3].sum()/weight)
                                        loads.append(output[4].sum(0)/weight)
                                        routing_weights.append(x.numel())
                                else:
                                    output=self.model(x,y);loss=output['loss']
                                    balances.append(output['balance_loss']);zs.append(output['z_loss'])
                                    if output['expert_load_pct'] is not None:
                                        loads.append(output['expert_load_pct']);routing_weights.append(x.numel())
                                del output
                                tokens += x.numel()
                            elif self.parallel is not None:
                                loss,count=self._parallel_completion_loss(pairs,self.mode,options)
                                tokens+=count
                            elif self.mode == 'chess':
                                batch_size = options.get('batch_size', 8)
                                if chess_stream is None:
                                    examples = random.choices(pairs, k=batch_size)
                                else:
                                    try:
                                        examples = [next(chess_stream)]
                                        for _ in range(batch_size - 1):
                                            examples.append(random.choice(pairs) if pairs and random.random() < 0.1 else next(chess_stream))
                                    except InterruptedError:
                                        raise
                                    except (OSError, ValueError) as exc:
                                        raise StreamingError(f'{dataset}: {exc}') from exc
                                selected = [chess_pair(example) for example in examples]
                                x, y = fixed_completion_batch(self.tokenizer, selected, self.model.config.block_size,
                                                              max_target_tokens=128, pad_to_block=False)
                                loss = self.model(x.to(self.device), y.to(self.device))['loss']
                                tokens += int((y != -100).sum())
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
                                    win,win_aux = completion_log_probability(self.model, self.tokenizer, pair['prompt'], chosen, self.device,return_aux=True)
                                    lose,lose_aux = completion_log_probability(self.model, self.tokenizer, pair['prompt'], rejected, self.device,return_aux=True)
                                    with torch.no_grad():
                                        ref_win = completion_log_probability(reference, self.tokenizer, pair['prompt'], chosen, self.device)
                                        ref_lose = completion_log_probability(reference, self.tokenizer, pair['prompt'], rejected, self.device)
                                    loss_dpo = -F.logsigmoid(options.get('beta', 0.1) * ((win - ref_win) - (lose - ref_lose)))
                                    losses.append(loss_dpo.mean()+0.5*(win_aux+lose_aux))
                                    tokens += min(self.model.config.block_size, len(self.tokenizer.encode(chosen))) + min(self.model.config.block_size, len(self.tokenizer.encode(rejected)))
                                loss = torch.stack(losses).mean()
                        loss_value=loss.detach().float().item()
                        if not math.isfinite(loss_value):
                            raise ValueError('Non-finite loss detected. Try a smaller learning rate or inspect the dataset.')
                        self.scaler.scale(loss / accumulation).backward()
                        total += loss_value / accumulation
                    self.scaler.unscale_(self.optimizer)
                    grad_norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0,error_if_nonfinite=not self.amp_enabled)
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
                    if self.stream_mixer:
                        self.stream_state = self.stream_mixer.state()
                    if chess_stream is not None:
                        self.stream_state = chess_stream.state()
                    elapsed = max(time.perf_counter() - began, 1e-9)
                    metric = {'step': self.step, 'train_loss': total, 'val_loss': None, 'steps_per_sec': 1 / elapsed,
                              'tokens_per_sec': tokens / elapsed, 'mode': self.mode}
                    norm=float(grad_norm.detach().float())
                    metric.update(grad_norm=norm if math.isfinite(norm) else None,
                                  learning_rate=self.optimizer.param_groups[0]['lr'],
                                  balance_loss=float(torch.stack(balances).mean()) if balances else 0.0,
                                  z_loss=float(torch.stack(zs).mean()) if zs else 0.0,
                                  expert_load_pct=(sum(load*weight for load,weight in zip(loads,routing_weights))/sum(routing_weights)).cpu().tolist() if loads else [],
                                  routing_tokens=sum(routing_weights))
                    if evaluate and self.mode == 'pretrain' and (self.step % interval == 0 or self.step == target):
                        self._evaluate(mixer)
                        metric['val_loss'] = self.val_loss
                    self.metrics.append(metric)
                    self.metrics = self.metrics[-3000:]
                    self.store.execute('INSERT INTO metrics(run_id,payload) VALUES(?,?)', (self.run_id, json.dumps(metric)))
                    chess_evaluation = self.mode == 'chess' and (self.step % interval == 0 or self.step == target)
                    checkpoint = self._save() if chess_evaluation or self.step % interval == 0 else None
                    self.watchdog.boundary()
                if chess_evaluation:
                    self._evaluate_chess(checkpoint, options['benchmark_size'], chess_engine_path)
            with self.lock:
                if self.mode != 'chess' or not self.checkpoint or self.checkpoint != f'chess-{self.run_id}-s{self.step}.pt':
                    self._save()
                self.state = 'stopped' if self.stop_event.is_set() else 'completed'
        except (StreamingError, InterruptedError) as exc:
            if not self.stream_mixer and chess_stream is None:
                self.state, self.error = 'error', str(exc)
            else:
                with self.lock:
                    try:
                        self._save()
                        self.state = 'stopped' if self.stop_event.is_set() else 'error'
                        self.error = None if self.state == 'stopped' else str(exc)
                    except Exception as save_error:
                        self.state, self.error = 'error', str(save_error)
        except Exception as exc:
            with self.lock:
                self.state, self.error = 'error', str(exc)
        finally:
            resources.close()
            if self.stream_mixer:
                self.stream_mixer.close()
                self.stream_mixer = None
            self._synchronous_running = False
            if self.model:
                self.model.eval()

    def _evaluate(self, mixer):
        if isinstance(mixer, StreamingMixer):
            mixer.reset_validation()
        self.model.eval()
        with torch.no_grad(), self._autocast():
            losses = [self.model(*mixer.batch(self.config.get('batch_size', 8), validation=True))['loss'].item() for _ in range(3)]
            self.val_loss = sum(losses) / len(losses)
            if not math.isfinite(self.val_loss):
                raise ValueError('Non-finite validation loss.')
            self.watchdog.evaluated()
            for prompt in self.prompts:
                result = self._generate(prompt, max_new_tokens=40, telemetry=False, seed=7)
                self.store.execute('INSERT INTO timeline(run_id,step,prompt,output) VALUES(?,?,?,?)', (self.run_id, self.step, prompt, result['text']))

    def _evaluate_chess(self, checkpoint, size, engine_path):
        self.chess_evaluating_step = self.step
        try:
            generate = partial(self._generate_chess, self.model, self.tokenizer)
            generate_batch = partial(self._generate_chess_batch, self.model, self.tokenizer)
            report = evaluate_checkpoint(generate, checkpoint['name'], self.run_id, self.step, size, engine_path,
                                         generate_batch=generate_batch)
            self.chess_reports.save(report, 'automatic')
            with self.lock:
                metric = self.metrics[-1]
                metric['chess_accuracy'] = report['metrics']
                self.store.execute('UPDATE metrics SET payload=? WHERE id=(SELECT MAX(id) FROM metrics WHERE run_id=?)',
                                   (json.dumps(metric), self.run_id))
        finally:
            self.chess_evaluating_step = None

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
        payload = {'model_config': export_config(self.model.config), 'tokenizer_state': self.tokenizer.state(),
                   'model_state_dict': self._model_state_for_checkpoint(), 'optimizer_state_dict': self._cpu_tree(self.optimizer.state_dict()),
                   'step': self.step, 'val_loss': self.val_loss, 'version_tag': version_tag,
                   'training_config': self.config, 'mode': self.mode, 'run_id': self.run_id,
                   'precision':self.precision,'scaler_state_dict':self.scaler.state_dict() if self.scaler else {},
                   'torch_version':str(torch.__version__),'checkpoint_format_version':1}
        payload['watchdog_state']=self.watchdog.checkpoint_state()
        if self.stream_state is not None:
            payload['stream_state'] = self.stream_state
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
        try:self.watchdog.checkpoint(path)
        except Exception:self.watchdog.event(action='ERROR',reason='Watchdog recovery checkpoint unavailable; training continues.')
        return metadata

    def save(self):
        with self.lock:
            if self.model is None:
                raise ValueError('No model to save.')
            return self._save()

    def checkpoints(self, expected_mode=None):
        rows = [json.loads(p.read_text()) for p in self.directory.glob('*.json')]
        if expected_mode is not None:
            rows = [row for row in rows if (row.get('mode') == 'chess') == (expected_mode == 'chess')]
        return sorted(rows, key=lambda row: row['created_at'], reverse=True)

    def checkpoint_path(self, name):
        if Path(name).name != name or not name.endswith('.pt') or not (self.directory / name).is_file():
            raise ValueError('Checkpoint not found.')
        return self.directory / name

    def checkpoint_payload(self, name, expected_mode):
        payload = torch.load(self.checkpoint_path(name), map_location='cpu', weights_only=True)
        if (payload.get('mode') == 'chess') != (expected_mode == 'chess'):
            raise ValueError('Choose a checkpoint from the current studio mode.')
        return payload

    def load(self, name, expected_mode='text'):
        with self.lock:
            self.require_idle()
            # Files are produced locally by NanoForge. weights_only avoids arbitrary pickle code.
            payload = self.checkpoint_payload(name, expected_mode)
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
            restored='auto'
            if self.config.get('precision')=='bfloat16':
                native=bool(str(self.device).startswith('cuda') and torch.cuda.get_device_capability(self.device)[0]>=8 and torch.cuda.is_bf16_supported())
                restored='bfloat16' if native else 'float32'
            self._configure_precision({**self.config,'precision':restored},payload.get('scaler_state_dict'))
            self.metrics = [json.loads(row['payload']) for row in self.store.rows('SELECT payload FROM metrics WHERE run_id=? ORDER BY id DESC LIMIT 3000', (self.run_id,))][::-1]
            self.checkpoint, self.state, self.error = name, 'ready', None
            self.target_step = None
            self._branch_pending = True
            self.watchdog.reset(payload.get('watchdog_state'))
            self.stream_state = payload.get('stream_state')
            return self.status()

    def _watchdog_rollback(self,name):
        """Prepare weights/optimizer/scaler completely before swapping live state."""
        payload=torch.load(self.directory/name,map_location=self.device,weights_only=True)
        config=ModelConfig(**payload['model_config'])
        current=self.model.config
        keys=('vocab_size','n_layer','n_head','n_embd','block_size','architecture')
        if config.architecture=='moe':keys+=('n_experts','experts_per_token')
        if any(getattr(config,key)!=getattr(current,key) for key in keys):
            raise ValueError('Recovery checkpoint architecture changed.')
        config.router_aux_loss=current.router_aux_loss;config.router_z_loss=current.router_z_loss
        candidate=GPT(config).to(self.device);candidate.load_state_dict(payload['model_state_dict'])
        optimizer=torch.optim.AdamW(candidate.parameters(),fused=str(self.device).startswith('cuda'))
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        learning_rate=max(1e-8,self.optimizer.param_groups[0]['lr']*self.config.get('watchdog_lr_factor',0.5))
        for group in optimizer.param_groups:
            group['lr']=learning_rate;group['fused']=str(self.device).startswith('cuda')
            if str(self.device).split(':')[0] in ('cuda','xpu'):group['foreach']=False
        scaler=torch.amp.GradScaler('cuda',enabled=self.amp_enabled,init_scale=1024.0)
        if self.amp_enabled and payload.get('scaler_state_dict'):scaler.load_state_dict(payload['scaler_state_dict'])
        parallel=None;training_gpus=1
        if self.config.get('multi_gpu',False):
            if not str(self.device).startswith('cuda') or torch.cuda.device_count()<2:raise ValueError('Recovery requires two available CUDA GPUs.')
            replica=ReplicaLoss(candidate,self.precision,telemetry=config.architecture=='moe')
            parallel=torch.nn.DataParallel(replica,device_ids=[0,1],output_device=0);training_gpus=2
        source=payload.get('version_tag','')+'.pt'
        source=source if Path(source).name==source and (self.directory/source).is_file() else None
        stream_state = payload.get('stream_state')
        if self.stream_mixer:
            self.stream_mixer.restore(stream_state)
        self.stream_state = stream_state
        self.model,self.optimizer,self.scaler=candidate,optimizer,scaler
        self.parallel,self.training_gpus=parallel,training_gpus
        self.step,self.val_loss=payload['step'],payload['val_loss']
        if source:self.checkpoint=source
        self.config['learning_rate']=learning_rate
        self.metrics=[]

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

    @staticmethod
    def _generate_chess(model, tokenizer, prompt):
        ids = torch.tensor([tokenizer.encode(format_prompt(prompt))], device=model.token_embedding.weight.device)
        end = tokenizer.encode('<|end|>')
        result = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, greedy=True,
                                stop_token_id=end[0] if len(end) == 1 else None)
        return tokenizer.decode(result['ids'][0, ids.size(1):].tolist()).split('<|end|>', 1)[0].strip()

    @staticmethod
    @torch.inference_mode()
    def _generate_chess_batch(model, tokenizer, prompts):
        prefixes = [tokenizer.encode(format_prompt(prompt)) for prompt in prompts]
        width = max(map(len, prefixes))
        stop = tokenizer.encode('<|end|>')
        if len(stop) != 1 or width + MAX_NEW_TOKENS > model.config.block_size:
            return [Trainer._generate_chess(model, tokenizer, prompt) for prompt in prompts]

        device = model.token_embedding.weight.device
        positions = torch.tensor([len(prefix) for prefix in prefixes], device=device)
        rows = torch.arange(len(prefixes), device=device)
        ids = torch.zeros((len(prefixes), width + MAX_NEW_TOKENS), dtype=torch.long, device=device)
        for row, prefix in enumerate(prefixes):
            ids[row, :len(prefix)] = torch.tensor(prefix, device=device)
        stopped = torch.zeros(len(prefixes), dtype=torch.bool, device=device)
        tokens = []
        was_training = model.training
        model.eval()
        try:
            for offset in range(MAX_NEW_TOKENS):
                # Pads are to the right of each real sequence. Causal attention cannot see them.
                output = model(ids[:, :width + offset], logits_positions=positions - 1)
                token = output['logits'][:, 0].argmax(dim=-1)
                token = torch.where(stopped, stop[0], token)
                ids[rows, positions] = token
                positions += 1
                tokens.append(token)
                stopped |= token == stop[0]
                if bool(stopped.all()):
                    break
        finally:
            model.train(was_training)

        sequences = torch.stack(tokens, dim=1).tolist()
        return [tokenizer.decode(row[:row.index(stop[0])] if stop[0] in row else row).strip() for row in sequences]

    def chess_generator(self, checkpoint):
        with self.lock:
            payload = self.checkpoint_payload(checkpoint, 'chess')
        with torch.device('meta'):
            model = GPT(ModelConfig(**payload['model_config']))
        model.load_state_dict(payload['model_state_dict'], assign=True)
        model = model.to(self.device).eval()
        tokenizer = Tokenizer(state=payload['tokenizer_state'])
        meta = {'run_id': payload.get('run_id'), 'step': payload['step'], 'config': payload.get('training_config', {})}
        return partial(self._generate_chess, model, tokenizer), partial(self._generate_chess_batch, model, tokenizer), meta

    def generate(self, **kwargs):
        with self.lock:
            self.require_text()
            return self._generate(**kwargs)

    def generate_from(self, checkpoint=None, **kwargs):
        with self.lock:
            if not checkpoint or checkpoint == self.checkpoint:
                self.require_text()
                result = self._generate(**kwargs)
            else:
                payload = self.checkpoint_payload(checkpoint, 'text')
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
        return {'imported_pairs': self.imported_pairs(), 'state': self.state, 'device': self.device, 'device_name': device_name(self.device), 'precision':self.precision, 'bf16_supported':bool(str(self.device).startswith('cuda') and torch.cuda.get_device_capability(self.device)[0]>=8 and torch.cuda.is_bf16_supported()), 'allow_tf32':self.allow_tf32, 'cuda_device_count':torch.cuda.device_count() if str(self.device).startswith('cuda') else 0, 'training_gpus':self.training_gpus, 'step': self.step, 'target_step': self.target_step, 'mode': self.mode,
                'run_id': self.run_id, 'checkpoint': self.checkpoint, 'error': self.error, 'val_loss': self.val_loss,
                'parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
                'model_config': asdict(self.model.config) if self.model else None, 'config': self.config,
                'metrics': self.metrics[-3000:], 'chess_evaluating_step': self.chess_evaluating_step, 'prompts': self.prompts,'watchdog':self.watchdog.status()}
