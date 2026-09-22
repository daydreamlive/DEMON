"""Keep verified YuE2 weights resident while running serial research phases.

Write a JSON command to <runtime>/round2-command.json. Supported commands:
{"action":"run", "phase":"trt_ring", "case":"jingle_full"}
{"action":"stop"}
State and phase errors are written to disk. An idle worker exits after 20 min.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from scripts.spikes.yue2_feasibility import load_verified_model, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    worker = SimpleNamespace(root=args.model_root, runtime=args.runtime, output=args.output)
    args.runtime.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    status = args.runtime / 'round2-status.json'
    command = args.runtime / 'round2-command.json'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    write_json(status, {'status': 'loading'})
    worker.model, worker.model_identity = load_verified_model(args.model_root/'YuE2-3B')
    worker.vae, worker.vae_identity = load_verified_model(args.model_root/'YuE2-Vae', vae=True)
    while True:
        write_json(status, {'status': 'ready'})
        deadline = time.monotonic() + 1200
        while not command.exists() and time.monotonic() < deadline:
            time.sleep(.25)
        if not command.exists():
            break
        request = json.loads(command.read_text(encoding='utf-8-sig'))
        command.unlink()
        if request['action'] == 'stop':
            break
        try:
            write_json(status, {'status': 'running', 'request': request})
            module = importlib.import_module('scripts.spikes.yue2_round2')
            module = importlib.reload(module)
            with torch.inference_mode():
                module.run(worker, request)
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            write_json(args.output / ('error_' + str(request.get('id', 'phase')) + '.json'),
                       {'error': error, 'request': request})
    write_json(status, {'status': 'stopped'})
    print('Worker stopped; releasing GPU weights.', flush=True)


if __name__ == '__main__':
    main()
