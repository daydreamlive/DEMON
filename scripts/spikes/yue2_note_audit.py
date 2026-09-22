"""Independent note/instrument transcription of short YuE2 review excerpts.

MuScriptor labels are model estimates, not listening judgments. No instrument
restriction or reference score is passed to the transcriber.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import soundfile as sf
import torch
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.events import NoteEndEvent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = TranscriptionModel.load_model(str(args.model), device='cpu', dtype='float32')
    cases = [('concise_4bar_s381', args.root/'concise_4bar_s381/audio.wav', 0, 10),
             ('concise_1bar_s381', args.root/'concise_1bar_s381/audio.wav', 0, 6.4),
             ('jingle_full', args.root/'jingle_full/audio.wav', 6, 10),
             ('piano_1bar', args.root/'piano_1bar/audio.wav', 0, 10),
             ('piano_4bar', args.root/'piano_4bar/audio.wav', 0, 10),
             ('drums_off', args.root.parent/'demon_torch_2_9/drums_off/audio.wav', 0, 10)]
    results = []
    for name, path, start, duration in cases:
        wave, sr = sf.read(path, dtype='float32', always_2d=True)
        wave = wave[int(start*sr):int((start+duration)*sr)].T.copy()
        notes = []
        begin = time.perf_counter()
        for event in model.transcribe((torch.from_numpy(wave), sr), instruments=None, batch_size=1):
            if isinstance(event, NoteEndEvent):
                note = event.start_event
                notes.append(dict(instrument=note.instrument, pitch=note.pitch,
                                  start=note.start_time+start, end=event.end_time+start))
        row = dict(case=name, excerpt_start_s=start, excerpt_seconds=wave.shape[-1]/sr,
                   reference_instruments_supplied=False, reference_score_supplied=False,
                   model=str(args.model), device='cpu', wall_s=time.perf_counter()-begin,
                   instrument_note_counts=dict(Counter(n['instrument'] for n in notes)), notes=notes)
        results.append(row)
        (args.root/'notes.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='notes'}),flush=True)


if __name__ == '__main__':
    main()
