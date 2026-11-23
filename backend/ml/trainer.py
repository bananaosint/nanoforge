"""Single-owner background training with explicit CE, SFT, and DPO objectives."""
import copy
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
from .model import GPT, ModelConfig, PRESETS, detect_device, device_name
from .tokenizer import Tokenizer, format_prompt, END
from .dataset import WeightedMixer, completion_batch, completion_log_probability
from .corpus import FileCorpus, vocabulary_text, PairPool
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

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def require_idle(self):
        if self.busy():
            raise ValueError('Stop the active run before changing models or checkpoints.')

    def create(self, config):
        with self.lock:
            self.require_idle()
            texts = self._texts(config.get('datasets', []))
            tokenizer = Tokenizer(config.get('tokenizer', 'char'), vocabulary_text(texts) if config.get('tokenizer', 'char') == 'char' else '')
            model_config = {**PRESETS[config.get('preset', 'Nano')], **config.get('model', {}), 'vocab_size': tokenizer.vocab_size}
            candidate = GPT(ModelConfig(**model_config)).to(self.device)
            self.model, self.tokenizer = candidate, tokenizer
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.get('learning_rate', 0.0003), weight_decay=0.1, foreach=False if self.device == 'xpu' else None)
            self.config = config
            self.step, self.val_loss, self.checkpoint = 0, None, None
            self.target_step = None
            self._branch_pending = False
            self.mode, self.state, self.error = 'pretrain', 'ready', None
            self._new_run(config)
            return self.status()

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
                        if self.mode == 'pretrain' or (self.mode == 'sft' and mixer and random.random() < options.get('base_mix', 0.1)):
                            x, y = mixer.batch(options.get('batch_size', 8))
                            loss = self.model(x, y)['loss']
                            tokens += x.numel()
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
                        if not torch.isfinite(loss):
                            raise ValueError('Non-finite loss detected. Try a smaller learning rate or inspect the dataset.')
                        (loss / accumulation).backward()
                        total += loss.item() / accumulation
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
                    self.optimizer.step()
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
        with torch.no_grad():
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
                   'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
                   'step': self.step, 'val_loss': self.val_loss, 'version_tag': version_tag,
                   'training_config': self.config, 'mode': self.mode, 'run_id': self.run_id}
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
            self.model = model.eval()
            self.tokenizer = Tokenizer(state=payload['tokenizer_state'])
            self.optimizer = torch.optim.AdamW(model.parameters(), foreach=False if self.device == 'xpu' else None)
            self.optimizer.load_state_dict(payload['optimizer_state_dict'])
            if self.device == 'xpu':
                for group in self.optimizer.param_groups:
                    group['foreach'] = False
            self.step, self.val_loss, self.config = payload['step'], payload['val_loss'], payload.get('training_config', {})
            self.run_id, self.mode = payload.get('run_id'), payload.get('mode', 'pretrain')
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
            return {j['output']: {'sft':j.get('rows') or 0, 'dpo':(j.get('rows') or 0) if k == 'hh-rlhf' else 0} for k,j in jobs.items() if j.get('state') in ('completed','sampled') and j.get('output','').endswith('.jsonl')}
        except (ValueError, OSError):
            return {}

    def status(self):
        return {'imported_pairs': self.imported_pairs(), 'state': self.state, 'device': self.device, 'device_name': device_name(self.device), 'step': self.step, 'target_step': self.target_step, 'mode': self.mode,
                'run_id': self.run_id, 'checkpoint': self.checkpoint, 'error': self.error, 'val_loss': self.val_loss,
                'parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
                'model_config': asdict(self.model.config) if self.model else None, 'config': self.config,
                'metrics': self.metrics[-3000:], 'prompts': self.prompts}
