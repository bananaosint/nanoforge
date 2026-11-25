"""Gradio event handlers must be decorated directly so ZeroGPU discovers them."""
import spaces  # Must precede torch/any backend imports on ZeroGPU.
import os
from pathlib import Path
import gradio as gr
import torch
from backend import zerogpu as engine
from backend.importers.hub import HubImports, CATALOG
from backend.ml.dataset import DatasetManager

# Avoid inherited CPU OpenMP worker pools when ZeroGPU forks a GPU request.
torch.set_num_threads(1)
ROOT=Path(os.environ.get('NANOFORGE_ZEROGPU_HOME','/tmp/nanoforge-zerogpu'))
ROOT.mkdir(parents=True,exist_ok=True)
SIZE=os.environ.get('NANOFORGE_ZERO_GPU_SIZE',engine.PROFILE['default_size'])
if SIZE not in ('large','xlarge'):raise ValueError('NANOFORGE_ZERO_GPU_SIZE must be large or xlarge.')


def prepare(dataset,checkpoint,mode,architecture,layers,heads,width,context,experts,top_k,batch,learning_rate,precision,tokenizer):
    options={**engine.PROFILE['training'],'mode':mode,'tokenizer':tokenizer,'precision':precision,
             'batch_size':int(batch),'learning_rate':float(learning_rate),
             'model':dict(n_layer=int(layers),n_head=int(heads),n_embd=int(width),block_size=int(context),dropout=0.1,
                          architecture=architecture,n_experts=int(experts),experts_per_token=int(top_k))}
    try:return engine.prepare_session(ROOT,options,dataset,checkpoint)
    except Exception as exc:raise gr.Error(str(exc)) from exc


def sample_dataset(source):
    if source not in {item['id'] for item in CATALOG}:raise gr.Error('Choose a dataset from the catalog.')
    folder=ROOT/'samples'/source
    manager=HubImports(DatasetManager(folder))
    try:
        manager.start(source,1000);manager.thread.join(180)
        if manager.thread.is_alive():manager.pause();raise ValueError('Import is still preparing. Retry after it pauses.')
        job=manager.status()['jobs'][source]
        if job['state'] not in ('sampled','completed'):raise ValueError(job.get('error') or 'Dataset import failed.')
        return str(folder/job['output'])
    finally:manager.shutdown()


@spaces.GPU(duration=engine.PROFILE['lease_seconds'],size=SIZE)
def train(session,steps):
    try:return engine.train_chunk(ROOT,session,int(steps))
    except Exception as exc:raise gr.Error(str(exc)) from exc


@spaces.GPU(duration=engine.PROFILE['lease_seconds'],size=SIZE)
def generate(session,prompt,tokens):
    try:return engine.generate(ROOT,session,prompt,int(tokens))
    except Exception as exc:raise gr.Error(str(exc)) from exc


with gr.Blocks(title='NanoForge ZeroGPU') as demo:
    gr.Markdown('''# NanoForge · ZeroGPU Blackwell
Create/import on CPU, then request short GPU training or generation calls. Each completed training chunk saves a portable `.pt`; click again to resume. This is a bounded demo/fine-tuning adapter, not the continuous Colab Studio. Default allocation: **48 GB**; owner-selected `xlarge` uses 96 GB and twice the quota.

**Limits:** 10M total parameters, context 256, batch 8, 25 updates/call, 60-second GPU lease with a 35-second soft work budget. Preparation/tokenization uses CPU. If a request is interrupted or quota is exhausted, the previous completed checkpoint remains. Download outputs before the Space restarts; storage is temporary.''')
    session=gr.State('')
    with gr.Row():
        dataset=gr.File(label='UTF-8 dataset · .txt/.jsonl · max 20 MiB',file_types=['.txt','.jsonl'],type='filepath')
        checkpoint=gr.File(label='Optional NanoForge checkpoint · max 128 MiB',file_types=['.pt'],type='filepath')
    sample_source=gr.Dropdown([(item['label']+' · '+item['repo'],item['id']) for item in CATALOG],value='dolly',label='Direct dataset download (CPU sample)')
    sample=gr.Button('Import dataset sample (1,000 source rows, CPU)')
    gr.Markdown('Settings below apply when you **create/import a session**. Import your latest checkpoint to resume with a different objective, batch or learning rate. The chunk-length control applies to each training click.')
    with gr.Row():
        mode=gr.Dropdown(['pretrain','sft','dpo'],value='pretrain',label='Training objective')
        tokenizer=gr.Dropdown(['magic8k','char','bpe'],value='magic8k',label='Tokenizer (match imported checkpoint)')
        architecture=gr.Dropdown(['dense','moe'],value='dense',label='Architecture (new models only)')
        precision=gr.Dropdown(['bfloat16','float16','float32'],value='bfloat16',label='GPU precision')
    with gr.Row():
        layers=gr.Slider(1,8,value=4,step=1,label='Layers')
        heads=gr.Dropdown([1,2,4,8],value=4,label='Heads')
        width=gr.Slider(16,256,value=128,step=16,label='Width (divisible by heads)')
        context=gr.Slider(32,256,value=128,step=32,label='Context')
    with gr.Row():
        experts=gr.Slider(2,8,value=4,step=1,label='MoE experts per layer')
        top_k=gr.Slider(1,2,value=2,step=1,label='MoE experts per token')
        batch=gr.Slider(1,8,value=2,step=1,label='Batch size')
        learning_rate=gr.Number(value=0.0003,label='Learning rate')
    create=gr.Button('Create / import session on CPU')
    status=gr.Textbox(label='Session status',interactive=False)
    with gr.Row():
        steps=gr.Slider(1,25,value=10,step=1,label='Additional optimizer updates this call')
        train_button=gr.Button('Train next chunk',variant='primary')
    metrics=gr.JSON(label='Completed chunk')
    output_file=gr.File(label='Download latest portable checkpoint',interactive=False)
    prompt=gr.Textbox(value='Once upon a time',label='Generation prompt')
    tokens=gr.Slider(1,64,value=32,step=1,label='New tokens')
    generate_button=gr.Button('Generate with current checkpoint')
    text=gr.Textbox(label='Completion',interactive=False)
    sample.click(sample_dataset,inputs=sample_source,outputs=dataset,concurrency_id='nanoforge-cpu')
    create.click(prepare,inputs=[dataset,checkpoint,mode,architecture,layers,heads,width,context,experts,top_k,batch,learning_rate,precision,tokenizer],
                 outputs=[session,status,output_file],concurrency_id='nanoforge-cpu')
    train_button.click(train,inputs=[session,steps],outputs=[metrics,output_file],concurrency_id='nanoforge-gpu',concurrency_limit=1)
    generate_button.click(generate,inputs=[session,prompt,tokens],outputs=text,concurrency_id='nanoforge-gpu',concurrency_limit=1)

if __name__=='__main__':
    demo.queue(default_concurrency_limit=1,max_size=8).launch(max_file_size='128mb',show_error=False)
