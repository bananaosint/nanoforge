"""Create a self-contained Gradio Space directory; no upload/login is performed."""
import argparse
from pathlib import Path
import shutil


def export(destination):
    root=Path(__file__).resolve().parents[1]
    destination=Path(destination).resolve()
    if destination==root or root in destination.parents:
        raise ValueError('Choose a separate output directory.')
    if destination.exists() and any(destination.iterdir()):raise ValueError('Output directory must be empty; existing files are preserved.')
    destination.mkdir(parents=True,exist_ok=True)
    shutil.copytree(root/'backend',destination/'backend',ignore=shutil.ignore_patterns('__pycache__','*.pyc','tests'))
    (destination/'configs').mkdir()
    shutil.copyfile(root/'configs/zerogpu.json',destination/'configs/zerogpu.json')
    for name in ('app.py','requirements.txt','README.md'):shutil.copyfile(root/'spaces/zerogpu'/name,destination/name)
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(export(parser.parse_args().output))
