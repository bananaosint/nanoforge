"""Asynchronous advice, bounded local policy, and checkpoint telemetry."""
from collections import deque
from functools import wraps
import copy
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid
from .gemini_client import BackgroundCall


def synchronized(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        with self.lock:return method(self,*args,**kwargs)
    return call


class CheckpointWatchdog:
    def __init__(self,trainer,service):
        self.trainer,self.service=trainer,service
        self.lock=threading.RLock()
        self.worker=BackgroundCall();self.reset()

    @synchronized
    def reset(self,state=None):
        self.epoch=uuid.uuid4().hex;self.updates=0;self.last_request=0;self.rollbacks=0
        self.validations=deque(maxlen=5);self.routing=deque(maxlen=8);self.events=deque(maxlen=30)
        self.best_val_loss=None;self.best_file=None;self.last_eval_step=None;self.last_submitted=None
        if state:
            # Embedded state is optional; legacy Studio checkpoints remain valid.
            try:
                self.validations.extend(state.get('validation_history',[])[-5:])
                self.routing.extend(state.get('routing_checkpoints',[])[-8:])
                self.events.extend(state.get('events',[])[-30:])
                name=state.get('best_file') or ''
                if re.fullmatch(r'\.watchdog/best-[a-f0-9]{12}-s[0-9]+\.pt',name) and (self.trainer.directory/name).is_file():
                    value=float(state['best_val_loss'])
                    if math.isfinite(value):self.best_file=name;self.best_val_loss=value
            except (TypeError,ValueError,KeyError):
                self.validations.clear();self.routing.clear();self.best_file=None;self.best_val_loss=None

    def enabled(self):
        return self.trainer.mode=='pretrain' and self.trainer.config.get('watchdog_enabled',False)

    @synchronized
    def event(self,**values):
        self.events.append({'step':self.trainer.step,'time':time.time(),**values})

    @synchronized
    def evaluated(self):
        value=self.trainer.val_loss
        self.last_eval_step=self.trainer.step
        self.validations.append({'step':self.trainer.step,'loss':value})

    @synchronized
    def checkpoint(self,path):
        if not self.enabled():return
        trainer=self.trainer
        # Capture only training-window routing, never the later eval/generation pass.
        recent=trainer.metrics[-min(100,len(trainer.metrics)):]
        routed=[row for row in recent if row.get('expert_load_pct')]
        if routed:
            layers=len(routed[-1]['expert_load_pct']);experts=len(routed[-1]['expert_load_pct'][0])
            denominator=sum(row.get('routing_tokens',1) for row in routed)
            snapshot={'step':trainer.step,'checkpoint':path.name,
                                 'expert_load_pct':[[sum(row['expert_load_pct'][i][j]*row.get('routing_tokens',1) for row in routed)/denominator
                                                    for j in range(experts)] for i in range(layers)]}
            if self.routing and self.routing[-1]['step']==trainer.step:self.routing[-1]=snapshot
            else:self.routing.append(snapshot)
        if self.last_eval_step==trainer.step and trainer.val_loss is not None and (self.best_val_loss is None or trainer.val_loss<self.best_val_loss):
            folder=trainer.directory/'.watchdog';folder.mkdir(exist_ok=True)
            target=folder/('best-'+trainer.run_id+'-s'+str(trainer.step)+'.pt');pending=target.with_suffix('.pending')
            pending.unlink(missing_ok=True)
            try:os.link(path,pending)  # Same filesystem: no second weights copy.
            except OSError:shutil.copyfile(path,pending)  # Drive/FUSE may not support links.
            pending.replace(target)
            previous=self.best_file
            self.best_file=str(target.relative_to(trainer.directory));self.best_val_loss=trainer.val_loss
            if previous and previous!=self.best_file:
                try:(trainer.directory/previous).unlink(missing_ok=True)
                except OSError:pass

    @synchronized
    def state(self):
        return {'best_val_loss':self.best_val_loss,'best_file':self.best_file,
                'validation_history':list(self.validations),'routing_checkpoints':list(self.routing),'events':list(self.events)}

    @synchronized
    def checkpoint_state(self):
        state=self.state();trainer=self.trainer
        if self.enabled() and self.last_eval_step==trainer.step and trainer.val_loss is not None and (self.best_val_loss is None or trainer.val_loss<self.best_val_loss):
            state.update(best_file='.watchdog/best-'+trainer.run_id+'-s'+str(trainer.step)+'.pt',best_val_loss=trainer.val_loss)
        return state

    def status(self):
        # A late response is visible as advice only; it cannot mutate a finished run.
        if not self.trainer.busy():
            result=self.worker.poll()
            if result:
                if result.get('epoch')!=self.epoch or result.get('provider_revision')!=self.service.revision:
                    self.event(action='IGNORED',reason='Stale decision discarded.')
                elif result.get('error'):self.event(action='ERROR',reason=result['error'])
                else:
                    self.event(**result['decision'],applied=False,guard='Run finished; no intervention applied.')
        with self.lock:
            return {**self.state(),'pending':self.worker.busy(),'events':list(self.events),'rollbacks':self.rollbacks}

    def payload(self):
        trainer=self.trainer;recent=trainer.metrics[-100:]
        def window(key):return [{'step':row['step'],'value':row[key]} for row in recent if row.get(key) is not None]
        return {'step':trainer.step,'total_steps':trainer.target_step,'learning_rate':trainer.optimizer.param_groups[0]['lr'],
                'balance_loss':recent[-1].get('balance_loss') if recent else None,
                'z_loss':recent[-1].get('z_loss') if recent else None,
                'balance_loss_coefficient':trainer.model.config.router_aux_loss,
                'z_loss_coefficient':trainer.model.config.router_z_loss,
                'train_loss_window':window('train_loss'),'gradient_norm_window':window('grad_norm'),
                'validation_loss_history':list(self.validations),'best_val_loss':self.best_val_loss,
                'routing_checkpoints':list(self.routing)[-2:],
                'routing_units':'Percent of tokens routed to each expert, per layer. With top-k, each layer sums to 100*top_k.',
                'experts_per_token':trainer.model.config.experts_per_token if trainer.model.config.architecture=='moe' else None}

    def boundary(self):
        """Called between complete optimizer steps under the trainer lock. Never waits."""
        if not self.enabled():return
        try:
            self._consume()
            self.updates+=1
            interval=self.trainer.config.get('watchdog_interval_steps',500)
            if self.updates-self.last_request<interval or self.trainer.stop_event.is_set():return
            self.last_request=self.updates
            # Snapshot checkpoint and telemetry on the owning training thread.
            self.trainer._save()
            payload=self.payload();client,revision=self.service.client()
            context={'epoch':self.epoch,'updates':self.updates,'provider_revision':revision}
            if self.worker.submit(client,'watchdog',copy.deepcopy(payload),context):
                self.last_submitted=payload;self.event(action='REQUESTED',reason='Structured checkpoint telemetry submitted.')
            else:self.event(action='SKIPPED',reason='A prior Gemini request is still pending; no backlog added.')
        except Exception:
            self.event(action='ERROR',reason='Watchdog hook failed; training continues unchanged.')

    def _consume(self):
        result=self.worker.poll()
        if not result:return
        interval=self.trainer.config.get('watchdog_interval_steps',500)
        if result.get('epoch')!=self.epoch or result.get('provider_revision')!=self.service.revision or self.updates-result.get('updates',0)>2*interval:
            self.event(action='IGNORED',reason='Stale decision discarded.');return
        if result.get('error'):
            self.event(action='ERROR',reason=result['error']);return
        decision=result['decision'];action=decision['action']
        if decision['confidence']<self.trainer.config.get('watchdog_min_confidence',0.8):
            self.event(**decision,applied=False,guard='Below minimum confidence.');return
        allowed=self.guard(action)
        if not allowed:
            self.event(**decision,applied=False,guard='Local numeric evidence or recovery checkpoint is insufficient.');return
        try:
            if action=='ROLLBACK_AND_DAMPEN':
                self.trainer._watchdog_rollback(self.best_file)
                self.rollbacks+=1;self.epoch=uuid.uuid4().hex
            elif action=='BUMP_BALANCE_LOSS':
                config=self.trainer.model.config
                coefficient=min(1.0,max(1e-5,config.router_aux_loss*self.trainer.config.get('watchdog_balance_multiplier',2.0)))
                config.router_aux_loss=coefficient
                for block in self.trainer.model.blocks:block.mlp.aux_coef=coefficient
                self.trainer.config['model']={**self.trainer.config.get('model',{}),'router_aux_loss':coefficient}
            elif action=='EARLY_STOP':self.trainer.stop_event.set()
            self.event(**decision,applied=True)
        except Exception:
            self.event(**decision,applied=False,guard='Action failed; current model retained.')

    def guard(self,action):
        if action=='CONTINUE':return True
        if action=='ROLLBACK_AND_DAMPEN':
            if not self.best_file or not (self.trainer.directory/self.best_file).is_file() or self.rollbacks>=3:return False
            values=[row['train_loss'] for row in self.trainer.metrics[-20:]]
            rising=len(values)>=4 and sum(values[-2:])/2>sum(values[:2])/2*1.25
            regressed=len(self.validations)>=2 and self.validations[-1]['loss']>self.best_val_loss*1.1
            return rising or regressed
        if action=='EARLY_STOP':
            return len(self.validations)>=5 and self.best_val_loss is not None and all(row['loss']>self.best_val_loss*1.01 for row in self.validations)
        if action=='BUMP_BALANCE_LOSS':
            if self.trainer.model.config.architecture!='moe' or len(self.routing)<2:return False
            ideal=100*self.trainer.model.config.experts_per_token/self.trainer.model.config.n_experts
            return all(any(max(layer)>min(95,ideal*1.75) and min(layer)<ideal*.25 for layer in row['expert_load_pct']) for row in list(self.routing)[-2:])
        return False
