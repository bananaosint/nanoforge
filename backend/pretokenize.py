"""Prepare packed token caches before a long local or Colab training run."""
import argparse
import json
import os
from pathlib import Path
from .ml.corpus import FileCorpus,vocabulary_text
from .ml.dataset import DatasetManager
from .ml.tokenizer import Tokenizer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',type=Path,default=Path(os.environ.get('NANOFORGE_HOME',Path(__file__).resolve().parents[1])))
    parser.add_argument('--tokenizer',choices=['char','bpe','magic8k'],default='magic8k')
    parser.add_argument('--dataset',action='append',required=True,help='A filename in home/data; repeat to share one character vocabulary across sources.')
    args=parser.parse_args()
    datasets=DatasetManager(args.home/'data')
    corpora=[FileCorpus(datasets.path(name)) for name in args.dataset]
    tokenizer=Tokenizer(args.tokenizer,vocabulary_text([(source,1) for source in corpora]) if args.tokenizer=='char' else '')
    for source in corpora:
        ids=source.tokens(tokenizer)
        print(json.dumps({'source':source.path.name,'tokenizer':args.tokenizer,'vocab_size':tokenizer.vocab_size,'tokens':len(ids),'dtype':str(ids.dtype),'bytes':ids.nbytes,'cache':str(ids.filename)}),flush=True)


if __name__=='__main__':
    main()
