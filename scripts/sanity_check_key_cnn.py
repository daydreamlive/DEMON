"""Sanity check: feed synthetic triad arpeggios to detect_key. If the
port is correct, the model should land in the right key family."""
import numpy as np
from acestep.audio.key_detection import detect_key

sr = 44100
duration = 8

def arpeggio(notes_hz):
    t = np.arange(int(duration * sr)) / sr
    y = np.zeros_like(t)
    seg_len = sr  # 1s per note
    for i, f in enumerate(notes_hz):
        s = i * seg_len
        e = s + seg_len
        if e > len(y):
            break
        # Fundamental + first 3 harmonics so it sounds like a real instrument
        for h, amp in enumerate([1.0, 0.5, 0.25, 0.12], start=1):
            y[s:e] += amp * np.sin(2 * np.pi * f * h * np.arange(seg_len) / sr)
    y = y / np.max(np.abs(y))
    return y.astype(np.float32)

# Repeat 2x to get 8s of audio.
def make(notes_hz):
    one = arpeggio(notes_hz)[: 4 * sr]
    return np.tile(one, 2).astype(np.float32)

print("C major (C E G C):", detect_key(make([261.63, 329.63, 392.00, 523.25]), sr))
print("A minor (A C E A):", detect_key(make([220.00, 261.63, 329.63, 440.00]), sr))
print("E minor (E G B E):", detect_key(make([329.63, 392.00, 493.88, 659.25]), sr))
print("F# major (F# A# C# F#):", detect_key(make([369.99, 466.16, 554.37, 739.99]), sr))
