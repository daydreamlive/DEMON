"""Compare our port's output against madmom's reference activations on
the upstream sample files. Bit-close match means the port is faithful;
mismatch tells us preprocessing/conv has a bug."""
import sys
import numpy as np
import soundfile as sf
import torch

from acestep.audio.key_detection import _load, _resample_to_44100

DATA = "C:/Users/ryanf/AppData/Local/Temp/madmom_test"

def predict(path):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    bundle = _load()
    audio = _resample_to_44100(np.ascontiguousarray(y), sr)
    if audio.size < bundle["n_fft"]:
        audio = np.pad(audio, (0, bundle["n_fft"] - audio.size))
    audio_t = torch.from_numpy(audio.astype(np.float32)).to(bundle["device"])
    spec = torch.stft(
        audio_t, n_fft=bundle["n_fft"], hop_length=bundle["hop"],
        win_length=bundle["n_fft"], window=bundle["window"],
        center=True, pad_mode="reflect", return_complex=True,
    )
    mag = spec.abs().T
    filt = mag @ bundle["filterbank"]
    log_spec = torch.log10(filt + 1.0)
    with torch.inference_mode():
        logits = bundle["model"](log_spec.unsqueeze(0).unsqueeze(0)).squeeze().cpu().numpy()
    # madmom applies softmax in the .pkl pipeline (network ends with softmax).
    # Our model is the conv stack only (no softmax), so apply it here.
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return probs, bundle["labels"]


def reference(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    arr = z[z.files[0]]
    if arr.ndim == 2:
        arr = arr[0]
    return arr


for name, expected in [("sample", "Ab major"), ("sample2", "A minor")]:
    ours, labels = predict(f"{DATA}/{name}.wav")
    ref = reference(f"{DATA}/{name}.key_cnn.npz")
    our_label = labels[int(ours.argmax())]
    ref_label = labels[int(ref.argmax())]
    print(f"{name}: madmom-expected={expected}  ref-argmax={ref_label}  ours={our_label}")
    print(f"  max abs diff:  {np.abs(ours - ref).max():.4e}")
    print(f"  cosine sim:    {ours.dot(ref) / (np.linalg.norm(ours) * np.linalg.norm(ref)):.4f}")
    print(f"  ours top3: {[labels[i] for i in np.argsort(-ours)[:3]]}  {ours[np.argsort(-ours)[:3]].round(3)}")
    print(f"  ref  top3: {[labels[i] for i in np.argsort(-ref)[:3]]}  {ref[np.argsort(-ref)[:3]].round(3)}")
    print()
