"""Optional terminal alternative to the Data & Mixer import buttons."""
import argparse
import json
import os
import time
from pathlib import Path
from .hub import HubImports, BY_ID
from ..ml.dataset import DatasetManager


def main():
    parser = argparse.ArgumentParser(description='Direct, resumable NanoForge dataset imports')
    parser.add_argument('--dataset', choices=['all', *BY_ID], default='all')
    parser.add_argument('--max-records', type=int, default=0, help='0 imports the full train split; positive values produce a clearly marked sample')
    parser.add_argument('--home', type=Path, default=Path(os.environ.get('NANOFORGE_HOME', Path(__file__).resolve().parents[2])))
    args = parser.parse_args()
    if args.max_records < 0:
        parser.error('--max-records must be nonnegative')
    manager = HubImports(DatasetManager(args.home / 'data'))
    try:
        manager.start(args.dataset, args.max_records)
        last = None
        while manager.thread.is_alive():
            summary = {key:{field:job.get(field) for field in ('state','rows','current_record','bytes_downloaded','bytes_total','error')} for key,job in manager.status()['jobs'].items()}
            line = json.dumps(summary)
            if line != last:
                print(line, flush=True)
                last = line
            manager.thread.join(1)
    except KeyboardInterrupt:
        print('Pausing after the current operation. Run the same command to resume.', flush=True)
        manager.pause()
    finally:
        manager.shutdown()
    print(json.dumps(manager.status()['jobs'], indent=2), flush=True)
    if any(job['state'] == 'error' for job in manager.jobs.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
