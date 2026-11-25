"""Gemini drafts become training data only after explicit human approval."""
import json
import threading
import uuid
from .gemini_client import BackgroundCall
from .api.schemas import Feedback
from .storage import now


class TeachingJobs:
    def __init__(self,store,service):
        self.store,self.service=store,service;self.lock=threading.RLock();self.worker=BackgroundCall();self.active=None
        store.execute('CREATE TABLE IF NOT EXISTS teaching_jobs(id TEXT PRIMARY KEY,payload TEXT NOT NULL,created_at TEXT NOT NULL)')
        for row in store.rows('SELECT * FROM teaching_jobs'):
            job=json.loads(row['payload'])
            if job['state']=='running':
                job.update(state='error',error='Studio restarted before the teacher request finished. Submit a new request.')
                self._save(job)

    def _save(self,job):
        self.store.execute('INSERT OR REPLACE INTO teaching_jobs VALUES(?,?,?)',(job['id'],json.dumps(job),job['created_at']))

    def _poll(self):
        result=self.worker.poll()
        if not result:return
        job=self.get(result['id'],poll=False)
        if job['state']!='running':return
        if result.get('error'):job.update(state='error',error=result['error'])
        else:job.update(state='draft',draft=result['decision'])
        self._save(job);self.active=None

    def start(self,request):
        with self.lock:
            self._poll()
            if self.worker.busy():raise ValueError('A teacher request is already running. Wait for it to finish.')
            client,revision=self.service.client();settings=self.service.status()
            job={**request,'id':uuid.uuid4().hex,'state':'running','created_at':now(),
                 'provider':{'mode':settings['mode'],'model':settings['model']},'provider_revision':revision,
                 'draft':None,'error':None,'feedback_id':None}
            self._save(job)
            if not self.worker.submit(client,'teach',{'prompt':request['prompt'],'student_output':request['student_output']},{'id':job['id']}):
                job.update(state='error',error='Teacher worker unavailable. Retry the request.');self._save(job)
            else:self.active=job['id']
            return job

    def get(self,job_id,poll=True):
        with self.lock:
            if poll:self._poll()
            rows=self.store.rows('SELECT payload FROM teaching_jobs WHERE id=?',(job_id,))
            if not rows:raise ValueError('Teacher job not found.')
            return json.loads(rows[0]['payload'])

    def list(self):
        with self.lock:
            self._poll()
            return [json.loads(row['payload']) for row in self.store.rows('SELECT payload FROM teaching_jobs ORDER BY created_at DESC LIMIT 50')]

    def approve(self,job_id,corrected_output):
        with self.lock:
            job=self.get(job_id)
            if job['state']=='approved':return {'id':job['feedback_id'],'job':job}
            if job['state']!='draft':raise ValueError('Only a completed teacher draft can be approved.')
            provenance={'job_id':job_id,**job['provider'],'confidence':job['draft']['confidence'],
                        'reason':job['draft']['reason'],'human_edited':corrected_output!=job['draft']['corrected_output']}
            feedback=Feedback.model_validate({'kind':'sft','prompt':job['prompt'],'original_output':job['student_output'],
                                             'corrected_output':corrected_output,'checkpoint_a':job.get('checkpoint'),
                                             'teacher_provenance':provenance})
            # One SQLite transaction makes double-click/retry approval idempotent.
            with self.store.lock:
                try:
                    cursor=self.store.db.execute('INSERT INTO feedback(kind,payload,created_at) VALUES(?,?,?)',
                                                 ('sft',json.dumps(feedback.model_dump(exclude={'kind'})),now()))
                    job.update(state='approved',feedback_id=cursor.lastrowid,approved_output=corrected_output)
                    self.store.db.execute('UPDATE teaching_jobs SET payload=? WHERE id=?',(json.dumps(job),job_id))
                    self.store.db.commit()
                except Exception:self.store.db.rollback();raise
            return {'id':job['feedback_id'],'job':job}

    def shutdown(self):
        # Provider worker never writes to SQLite, so closing Store cannot race it.
        with self.lock:
            self._poll()
            if self.active:
                job=self.get(self.active,poll=False)
                job.update(state='error',error='Studio stopped before the teacher request completed.');self._save(job)
