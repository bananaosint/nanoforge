"""Headless training using the exact same model, trainer and .pt format as Studio."""
import argparse
import json
import os
from pathlib import Path
from .api.schemas import NewRun
from .checkpoints import register_checkpoint
from .ml.dataset import DatasetManager
from .ml.trainer import Trainer
from .storage import Store


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--home',type=Path,default=Path(os.environ.get('NANOFORGE_HOME',Path(__file__).resolve().parents[1])))
    parser.add_argument('--device',choices=['auto','cpu','cuda','mps','xpu'],default='auto')
    parser.add_argument('--resume',type=Path,help='A full NanoForge .pt checkpoint; steps are additional steps after this checkpoint.')
    args=parser.parse_args()
    os.environ['NANOFORGE_DEVICE']=args.device
    args.home.mkdir(parents=True,exist_ok=True)
    config=NewRun.model_validate(json.loads(args.config.read_text())).model_dump(exclude_none=True)
    store=Store(args.home/'nanoforge.sqlite3')
    trainer=None
    try:
        trainer=Trainer(DatasetManager(args.home/'data'),store,args.home/'checkpoints')
        if args.resume:
            metadata=register_checkpoint(args.resume,trainer.directory)
            trainer.load(metadata['name'])
            if trainer.tokenizer.kind!=config['tokenizer']:
                raise ValueError('Resume tokenizer differs from the config. Keep the checkpoint tokenizer or create a new model.')
        else:
            trainer.create(config)
        trainer.start(config)
        last=-1
        while trainer.thread.is_alive():
            status=trainer.status()
            if status['step']!=last:
                print(json.dumps({key:status[key] for key in ('state','device','device_name','precision','step','target_step','val_loss')}),flush=True)
                last=status['step']
            trainer.thread.join(1)
        if trainer.state=='error':
            raise RuntimeError(trainer.error)
        print(json.dumps({'state':trainer.state,'step':trainer.step,'checkpoint':str(trainer.directory/trainer.checkpoint),'metadata':str((trainer.directory/trainer.checkpoint).with_suffix('.json'))}),flush=True)
    except KeyboardInterrupt:
        if trainer:
            trainer.shutdown()
            print('Stopped safely; the final checkpoint was saved.',flush=True)
    finally:
        if trainer:trainer.shutdown()
        store.close()


if __name__=='__main__':
    main()
