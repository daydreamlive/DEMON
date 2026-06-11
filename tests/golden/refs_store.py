"""Canonical-reference storage: sha256-pinned bundles on a HF dataset.

The repo never carries reference audio; it carries ``refs.json`` (this
directory), a manifest mapping each scenario to a tarball on the
HuggingFace dataset repo plus the tarball's sha256, the tier-1
``canonical_sha256``, optional tier-2 thresholds, and the capture
environment. Bundles are fetched on demand into a local cache.

Lifecycle:
    # 1. capture baselines on the baseline build (e.g. main on the pod)
    python -m tests.golden.runner --pod-url ws://POD:1318 --scenario all \
        --out runs/baseline
    # 2. pack bundles + stamp the manifest
    python -m tests.golden.refs_store pack --runs runs/baseline
    # 3. upload tarballs (needs HF write token for the dataset repo)
    python -m tests.golden.refs_store upload
    # 4. anyone, anywhere:
    python -m tests.golden.refs_store fetch
"""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

MANIFEST = Path(__file__).parent / "refs.json"
DEFAULT_DATASET = "daydreamlive/demon-test-refs"
CACHE = Path.home() / ".cache" / "demon" / "test-refs"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def save_manifest(m: dict) -> None:
    MANIFEST.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ref_dir(scenario: str) -> Path | None:
    """Cached, verified reference bundle dir for a scenario, fetching it
    if needed. None when the manifest has no entry (no baseline yet)."""
    m = load_manifest()
    entry = m["bundles"].get(scenario)
    if entry is None:
        return None
    dest = CACHE / scenario
    marker = dest / ".sha256"
    if marker.exists() and marker.read_text() == entry["sha256"]:
        return dest
    tarball = _fetch_tarball(m, entry)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(dest, filter="data")
    got = _sha256_file(dest / "canonical.f32.raw")
    if got != entry["canonical_sha256"]:
        raise RuntimeError(
            f"{scenario}: extracted canonical sha {got[:12]} != manifest "
            f"{entry['canonical_sha256'][:12]}: corrupt ref bundle?")
    marker.write_text(entry["sha256"])
    return dest


def _fetch_tarball(manifest: dict, entry: dict) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(
        repo_id=manifest.get("dataset", DEFAULT_DATASET),
        filename=entry["file"],
        repo_type="dataset",
        revision=manifest.get("revision") or None,
    ))
    got = _sha256_file(path)
    if got != entry["sha256"]:
        raise RuntimeError(
            f"{entry['file']}: downloaded sha {got[:12]} != manifest "
            f"{entry['sha256'][:12]}: manifest/dataset out of sync?")
    return path


# ── CLI verbs ───────────────────────────────────────────────────────────

def cmd_fetch(args) -> int:
    m = load_manifest()
    names = (sorted(m["bundles"]) if args.scenario == "all"
             else [n.strip() for n in args.scenario.split(",")])
    if not names:
        print("manifest has no bundles yet: capture + pack first")
        return 1
    for name in names:
        d = ref_dir(name)
        print(f"  {name}: {'MISSING from manifest' if d is None else d}")
    return 0


def cmd_pack(args) -> int:
    """Tar each scenario bundle from a runner output dir into
    ``refs-out/`` and stamp the manifest with its hashes + env."""
    runs = Path(args.runs)
    env = {}
    env_file = runs / "env.json"
    if env_file.exists():
        env = json.loads(env_file.read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = load_manifest()
    packed = 0
    for bundle in sorted(p for p in runs.iterdir() if p.is_dir()):
        metrics_file = bundle / "metrics.json"
        if not metrics_file.exists():
            continue
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        if metrics.get("status") != "ok":
            print(f"  {bundle.name}: status={metrics.get('status')}, "
                  f"not packing")
            continue
        name = metrics["scenario"]
        tar_path = out / f"{name}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            for f in sorted(bundle.rglob("*")):
                tf.add(f, arcname=str(f.relative_to(bundle)))
        m["bundles"][name] = {
            "file": tar_path.name,
            "sha256": _sha256_file(tar_path),
            "canonical_sha256": metrics["canonical_sha256"],
            "thresholds": m["bundles"].get(name, {}).get("thresholds"),
            "env": env,
        }
        print(f"  packed {name} -> {tar_path} "
              f"(canonical {metrics['canonical_sha256'][:12]})")
        packed += 1
    save_manifest(m)
    print(f"manifest updated: {MANIFEST}" if packed else "nothing packed")
    return 0 if packed else 1


def cmd_install(args) -> int:
    """Extract locally packed tarballs straight into the cache (same
    layout fetch produces). Lets a box that just captured + packed a
    baseline run the suite against it without the HF round-trip."""
    m = load_manifest()
    src = Path(args.dir)
    installed = 0
    for name, entry in m["bundles"].items():
        tarball = src / entry["file"]
        if not tarball.exists():
            continue
        got = _sha256_file(tarball)
        if got != entry["sha256"]:
            print(f"  {name}: tarball sha mismatch vs manifest, skipping")
            continue
        dest = CACHE / name
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(dest, filter="data")
        (dest / ".sha256").write_text(entry["sha256"])
        print(f"  installed {name} -> {dest}")
        installed += 1
    return 0 if installed else 1


def cmd_upload(args) -> int:
    from huggingface_hub import HfApi

    m = load_manifest()
    repo = m.get("dataset", DEFAULT_DATASET)
    api = HfApi()
    src = Path(args.dir)
    files = [src / e["file"] for e in m["bundles"].values()
             if (src / e["file"]).exists()]
    if not files:
        print(f"no manifest tarballs found under {src}: run pack first")
        return 1
    print(f"uploading {len(files)} bundle(s) to dataset {repo}:")
    for f in files:
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                        repo_id=repo, repo_type="dataset")
    print("done: commit the updated refs.json alongside this upload")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch", help="download + verify ref bundles")
    p.add_argument("--scenario", default="all")
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("pack", help="tar runner bundles + stamp manifest")
    p.add_argument("--runs", required=True,
                   help="runner output root (contains env.json)")
    p.add_argument("--out", default="refs-out")
    p.set_defaults(fn=cmd_pack)
    p = sub.add_parser("install", help="extract locally packed tarballs "
                                       "into the cache (no HF)")
    p.add_argument("--dir", default="refs-out")
    p.set_defaults(fn=cmd_install)
    p = sub.add_parser("upload", help="push packed tarballs to the HF "
                                      "dataset (needs write token)")
    p.add_argument("--dir", default="refs-out")
    p.set_defaults(fn=cmd_upload)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
