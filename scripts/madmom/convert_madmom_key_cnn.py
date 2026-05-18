"""Convert madmom's bundled key_cnn.pkl into a clean torch state dict
plus a filterbank matrix. Run once; commit the output ``.pt``.

The pickle is full of madmom-specific Python objects, so we use a
tolerant unpickler that turns unknown classes into stubs that just
record their pickled state. Conv layers + the following BatchNorm are
folded into a single (weight, bias) pair so the runtime doesn't need
BN buffers.

Output: ``<repo>/acestep/audio/assets/key_cnn.pt``
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from inspect_madmom_pkl import load as load_pickle  # type: ignore


# Frequencies of FFT bins for sr=44100 Hz, n_fft=8192. Used to build
# madmom's logarithmic filterbank.
SR = 44100
N_FFT = 8192
HOP = SR // 5  # fps=5
FMIN = 65.0
FMAX = 2100.0
NUM_BANDS_PER_OCTAVE = 24
A4 = 440.0


# Class label ordering used by madmom's CNNKeyRecognitionProcessor
# (must match the network's softmax output order).
KEY_LABELS = [
    'A major', 'Bb major', 'B major', 'C major', 'Db major',
    'D major', 'Eb major', 'E major', 'F major', 'F# major',
    'G major', 'Ab major', 'A minor', 'Bb minor', 'B minor',
    'C minor', 'C# minor', 'D minor', 'D# minor', 'E minor',
    'F minor', 'F# minor', 'G minor', 'G# minor',
]


def _log_frequencies(bands_per_octave: int, fmin: float, fmax: float, fref: float) -> np.ndarray:
    """Port of madmom.audio.filters.log_frequencies."""
    left = np.floor(np.log2(fmin / fref) * bands_per_octave)
    right = np.ceil(np.log2(fmax / fref) * bands_per_octave)
    freqs = fref * 2.0 ** (np.arange(left, right) / bands_per_octave)
    freqs = freqs[np.searchsorted(freqs, fmin):]
    freqs = freqs[:np.searchsorted(freqs, fmax, side="right")]
    return freqs


def _frequencies_to_bins(freqs: np.ndarray, bin_freqs: np.ndarray, unique: bool) -> np.ndarray:
    """Port of madmom.audio.filters.frequencies2bins."""
    indices = bin_freqs.searchsorted(freqs)
    indices = np.clip(indices, 1, len(bin_freqs) - 1)
    left = bin_freqs[indices - 1]
    right = bin_freqs[indices]
    indices -= freqs - left < right - freqs
    if unique:
        indices = np.unique(indices)
    return indices


def _build_filterbank() -> np.ndarray:
    """Build the (num_fft_bins, num_filters) filterbank matrix madmom
    uses for the key CNN. Triangular, log-spaced, no normalization."""
    num_fft_bins = N_FFT // 2 + 1
    bin_freqs = np.linspace(0, SR / 2, num_fft_bins)
    centers_hz = _log_frequencies(NUM_BANDS_PER_OCTAVE, FMIN, FMAX, A4)
    centers = _frequencies_to_bins(centers_hz, bin_freqs, unique=True)

    filters = []
    i = 0
    while i + 3 <= len(centers):
        start, center, stop = centers[i:i + 3]
        if stop - start < 2:
            center = start
            stop = start + 1
        col = np.zeros(num_fft_bins, dtype=np.float32)
        # rising edge (without center)
        if center > start:
            col[start:center] = np.linspace(0.0, 1.0, center - start, endpoint=False)
        # falling edge (incl. center, excl. stop)
        if stop > center:
            col[center:stop] = np.linspace(1.0, 0.0, stop - center, endpoint=False)
        filters.append(col)
        i += 1

    fb = np.stack(filters, axis=1)  # (num_fft_bins, num_filters)
    return fb


def _fold_bn_into_conv(conv: dict, bn: dict) -> tuple[np.ndarray, np.ndarray]:
    """Fold BN(gamma, beta, mean, inv_std) into the preceding conv.

    madmom conv weights: (in_ch, out_ch, kH, kW), uses true convolution
    (flipped kernel). torch Conv2d uses cross-correlation, so we flip
    H and W and transpose (in,out,...) -> (out,in,...).

    Returns (weight, bias) ready for torch state-dict population.
    """
    w = conv["weights"].astype(np.float32)
    cb = np.asarray(conv["bias"], dtype=np.float32).reshape(-1)
    gamma = bn["gamma"].astype(np.float32)
    beta = bn["beta"].astype(np.float32)
    mean = bn["mean"].astype(np.float32)
    inv_std = bn["inv_std"].astype(np.float32)

    # in,out,kH,kW -> out,in,kH,kW with flipped kernel
    w = w.transpose(1, 0, 2, 3)[:, :, ::-1, ::-1].copy()

    scale = (gamma * inv_std).astype(np.float32)
    new_w = w * scale[:, None, None, None]

    # madmom conv stored bias is shape (1,) and is functionally zero;
    # broadcast it across all out channels.
    if cb.size == 1:
        cb = np.broadcast_to(cb, (gamma.size,)).astype(np.float32)
    new_b = (cb - mean) * scale + beta
    return new_w.astype(np.float32), new_b.astype(np.float32)


# --- (kernel, padding) for each conv block, in order ----------------------
# Mirrors the architecture dump from inspect_madmom_pkl.py.
_BLOCKS = [
    (5, 2),  # 1 -> 24
    (3, 1),  # 24 -> 24
    # maxpool
    (3, 1),  # 24 -> 48
    (3, 1),  # 48 -> 48
    # maxpool
    (3, 1),  # 48 -> 96
    (3, 1),  # 96 -> 96
    # maxpool
    (3, 1),  # 96 -> 192
    (3, 1),  # 192 -> 192
    (1, 0),  # 192 -> 24 (classifier head, no pad)
]


def _walk_layers(net) -> list:
    """Flatten the pickled madmom layer list into a list of (cls, state) tuples."""
    out = []
    for layer in net._state["layers"]:
        out.append((layer._cls, layer._state))
    return out


def main() -> None:
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/madmom_weights/key_cnn.pkl"
    out_path = (
        Path(__file__).resolve().parent.parent
        / "acestep" / "audio" / "assets" / "key_cnn.pt"
    )

    net = load_pickle(pkl_path)
    layers = _walk_layers(net)

    # Sanity-check layer pattern: pad, conv, bn, [pad, conv, bn,] pool, ...
    convs_bns: list[tuple[dict, dict]] = []
    for cls_name, state in layers:
        if cls_name.endswith("ConvolutionalLayer"):
            convs_bns.append([state, None])  # type: ignore[list-item]
        elif cls_name.endswith("BatchNormLayer"):
            convs_bns[-1][1] = state  # type: ignore[index]
    if len(convs_bns) != len(_BLOCKS):
        raise SystemExit(
            f"expected {len(_BLOCKS)} conv blocks, got {len(convs_bns)}"
        )

    state_dict: dict[str, torch.Tensor] = {}
    for i, (conv, bn) in enumerate(convs_bns):
        w, b = _fold_bn_into_conv(conv, bn)
        state_dict[f"conv{i}.weight"] = torch.from_numpy(w)
        state_dict[f"conv{i}.bias"] = torch.from_numpy(b)

    fb = _build_filterbank()
    state_dict["filterbank"] = torch.from_numpy(fb)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": state_dict,
        "key_labels": KEY_LABELS,
        "sample_rate": SR,
        "n_fft": N_FFT,
        "hop": HOP,
        "fmin": FMIN,
        "fmax": FMAX,
        "blocks": _BLOCKS,
    }, out_path)
    print(f"wrote {out_path}")
    print(f"  filterbank: {fb.shape}, conv blocks: {len(convs_bns)}")
    for i, (k, v) in enumerate(state_dict.items()):
        print(f"  {k}: {tuple(v.shape)}")


if __name__ == "__main__":
    main()
