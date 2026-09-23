"""Independent, local CPU ASR evidence for YuE2 lyric completion.

ASR can recover words; it cannot certify musical quality or absence of humming.
No reference lyrics or prompt are supplied to the recognizer.
"""
import argparse
import json
from pathlib import Path
import time

from faster_whisper import WhisperModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--asr-model', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    args = parser.parse_args()
    output = args.root / 'asr.json'
    results = json.loads(output.read_text()) if output.exists() else []
    done = {row['case'] for row in results}
    model = WhisperModel(str(args.asr_model), device='cpu', compute_type='int8', cpu_threads=4,
                         num_workers=1, local_files_only=True)
    for name in args.cases:
        path = args.root / name / 'audio.wav'
        if name in done or not path.exists():
            continue
        started = time.perf_counter()
        segments, info = model.transcribe(str(path), language='en', beam_size=3,
                                           condition_on_previous_text=False, vad_filter=False,
                                           word_timestamps=True)
        rows = [dict(start=s.start, end=s.end, text=s.text, no_speech_prob=s.no_speech_prob,
                     avg_logprob=s.avg_logprob,
                     words=[dict(start=w.start, end=w.end, word=w.word, probability=w.probability) for w in s.words or []])
                for s in segments]
        row = dict(case=name, model=str(args.asr_model), backend='faster-whisper CPU int8',
                   reference_lyrics_supplied=False, seconds=info.duration,
                   wall_s=time.perf_counter()-started, segments=rows)
        results.append(row)
        output.write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
