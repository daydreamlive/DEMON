#!/usr/bin/env python3
"""End-to-end H100 XL turbo TRT pipeline with subprocess-isolated build attempts.

Each TRT build runs as a subprocess so a SIGSEGV in TRT doesn't kill the
orchestrator. After each successful build, runs numerical validation against
PT eager. Only the first config that builds AND validates wins. Then runs
the t2m benchmark and programmatic audio compare.

Run with venv python directly to avoid uv re-sync:
    /workspace/acestep/.venv/bin/python h100_full_pipeline.py
"""

import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("ACESTEP_MODELS_DIR", "/root/.daydream-scope/models/rtmg")
os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
sys.path.insert(0, "/workspace/acestep")

import numpy as np
import onnx
import torch
from onnx import numpy_helper

# ---------- thresholds ----------
CORR_THRESHOLD = 0.95
MIN_RMS = 1e-4

# ---------- paths ----------
WORK = Path("/workspace/build")
ONNX_DIR = WORK / "onnx"
ONNX_PATH = ONNX_DIR / "decoder_bf16_mixed.onnx"
ONNX_DYNBATCH_PATH = ONNX_DIR / "decoder_bf16_mixed_dynbatch.onnx"
ONNX_BF16_DIR = WORK / "onnx_bf16"
ONNX_BF16_PATH = ONNX_BF16_DIR / "decoder_bf16.onnx"
ONNX_BF16_DYNBATCH_PATH = ONNX_BF16_DIR / "decoder_bf16_dynbatch.onnx"

ENGINE_DIR = Path("/root/.daydream-scope/models/rtmg/trt_engines/decoder_xl-turbo_bf16mix_dynbatch_b8_240s")
ENGINE_PATH = ENGINE_DIR / "decoder_xl-turbo_bf16mix_dynbatch_b8_240s.engine"

RESULTS_DIR = Path("/workspace/test_output_h100")
REPORT_PATH = Path("/workspace/h100_pipeline_report.json")

CHECKPOINT = "acestep-v15-xl-turbo"
VENV_PY = "/workspace/acestep/.venv/bin/python"
BUILDER_SCRIPT = "/workspace/h100_build_one.py"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------- step 1: export ONNX (bf16_mixed and optionally bf16) ----------

def export_onnx(precision: str, out_path: Path):
    """Export decoder ONNX with given precision (bf16_mixed or bf16)."""
    if out_path.exists():
        log(f"  ONNX exists: {out_path}")
        return
    log(f"  exporting decoder ONNX (precision={precision}) ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from acestep.engine.model_context import ModelContext
    from acestep.engine.trt.export import OnnxExportConfig, export_decoder_onnx as _export

    ctx = ModelContext(
        config_path=CHECKPOINT,
        device="cuda",
        compile_decoder=False,
        compile_vae=False,
        skip_vae=True,
    )
    cfg = OnnxExportConfig(
        precision=precision,
        batch_size=1,
        seq_len=1500,
        enc_len=200,
    )
    t0 = time.time()
    _export(ctx.model, out_path, device="cuda", config=cfg)
    log(f"  exported in {time.time()-t0:.1f}s")
    del ctx
    gc.collect()
    torch.cuda.empty_cache()


# ---------- step 2: dynbatch patch ----------

def patch_dynbatch(in_path: Path, out_path: Path):
    if out_path.exists():
        log(f"  patched ONNX exists: {out_path}")
        return
    log(f"  patching reshape constants in {in_path.name} ...")
    model = onnx.load(str(in_path), load_external_data=False)
    graph = model.graph

    def get_const(name):
        for node in graph.node:
            if node.op_type == "Constant" and name in node.output:
                for attr in node.attribute:
                    if attr.name == "value":
                        return numpy_helper.to_array(attr.t).flatten().tolist()
        for init in graph.initializer:
            if init.name == name:
                return numpy_helper.to_array(init).flatten().tolist()
        return None

    def set_const(name, new_shape):
        new_arr = np.asarray(new_shape, dtype=np.int64)
        for node in graph.node:
            if node.op_type == "Constant" and name in node.output:
                for attr in node.attribute:
                    if attr.name == "value":
                        attr.t.CopyFrom(numpy_helper.from_array(new_arr, name=name))
                        return True
        for init in graph.initializer:
            if init.name == name:
                init.CopyFrom(numpy_helper.from_array(new_arr, name=name))
                return True
        return False

    fixes = 0
    seen = set()
    for node in graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        sn = node.input[1]
        sa = get_const(sn)
        if not sa:
            continue
        if sa[0] == 1 and len(sa) >= 2 and -1 not in sa[1:] and sn not in seen:
            if set_const(sn, [-1] + list(sa[1:])):
                fixes += 1
                seen.add(sn)
    log(f"  patched {fixes} unique reshape constants")
    onnx.save(model, str(out_path))
    log(f"  saved -> {out_path} ({out_path.stat().st_size/1e6:.1f} MB protobuf)")


# ---------- step 3: build attempts (subprocess-isolated) ----------

# (label, onnx_path_field, strongly_typed, opt, batch_max, set_bf16, set_fp16)
BUILD_CONFIGS = [
    # bf16_mixed graph (preferred — has fp32 islands for safety)
    ("mix_strong_o3_b8", "mix", 1, 3, 8, 0, 0),
    ("mix_strong_o0_b8", "mix", 1, 0, 8, 0, 0),
    ("mix_strong_o3_b1", "mix", 1, 3, 1, 0, 0),
    ("mix_strong_o0_b1", "mix", 1, 0, 1, 0, 0),
    # pure bf16 graph (no fp32 islands, smaller graph)
    ("bf16_strong_o3_b8", "bf16", 1, 3, 8, 0, 0),
    ("bf16_strong_o0_b8", "bf16", 1, 0, 8, 0, 0),
    ("bf16_strong_o3_b1", "bf16", 1, 3, 1, 0, 0),
    # last resort: non-strongly-typed (TRT picks precision)
    ("mix_nonstrong_bf16_o3_b8", "mix", 0, 3, 8, 1, 0),
    ("bf16_nonstrong_bf16_o3_b8", "bf16", 0, 3, 8, 1, 0),
]


def try_build(cfg, onnx_path):
    label = cfg[0]
    log(f"=== build attempt: {label} ===")
    if ENGINE_PATH.exists():
        ENGINE_PATH.unlink()
    cmd = [
        VENV_PY, "-u", BUILDER_SCRIPT,
        "--onnx", str(onnx_path),
        "--engine", str(ENGINE_PATH),
        "--strongly-typed", str(cfg[2]),
        "--opt", str(cfg[3]),
        "--batch-max", str(cfg[4]),
        "--set-bf16", str(cfg[5]),
        "--set-fp16", str(cfg[6]),
    ]
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=900,
        )
        elapsed = time.time() - t0
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT after 900s")
        return False, "timeout"

    log(f"  exit_code={result.returncode} elapsed={elapsed:.1f}s")
    # echo last few lines of child output
    for line in result.stdout.splitlines()[-6:]:
        log(f"    {line}")
    if result.stderr:
        for line in result.stderr.splitlines()[-3:]:
            log(f"    [stderr] {line}")

    if result.returncode == 0 and ENGINE_PATH.exists():
        return True, f"ok_{elapsed:.1f}s"
    if result.returncode in (-11, 139):
        return False, "SIGSEGV"
    return False, f"exit_{result.returncode}"


# ---------- step 4: validate engine numerically ----------

def validate_decoder_engine():
    log("validating engine numerically (PT vs TRT decoder forward) ...")

    from acestep.engine.model_context import ModelContext
    from acestep.engine.trt.runtime import TRTDecoder

    ctx = ModelContext(
        config_path=CHECKPOINT,
        device="cuda",
        compile_decoder=False,
        compile_vae=False,
        skip_vae=True,
    )
    pt_decoder = ctx.model.decoder
    pt_decoder.eval()

    trt_decoder = TRTDecoder(str(ENGINE_PATH), device=torch.device("cuda"))

    B, T, D = 1, 1450, 64
    L = 200
    torch.manual_seed(7)
    hidden_states = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
    timestep = torch.full((B,), 0.5, device="cuda", dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(B, L, 2048, device="cuda", dtype=torch.bfloat16)
    encoder_attention_mask = torch.ones(B, L, device="cuda", dtype=torch.bfloat16)
    context_latents = torch.randn(B, T, 128, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        pt_out = pt_decoder(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
        )
    if isinstance(pt_out, (tuple, list)):
        pt_out = pt_out[0]
    pt_out = pt_out.float()

    trt_out = trt_decoder(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        context_latents=context_latents,
    ).float()

    pt_flat = pt_out.flatten()
    trt_flat = trt_out.flatten()

    cos_sim = float(torch.nn.functional.cosine_similarity(pt_flat[None], trt_flat[None]).item())
    max_abs_diff = float((pt_flat - trt_flat).abs().max().item())
    pt_max = float(pt_flat.abs().max().item())
    trt_max = float(trt_flat.abs().max().item())
    rel_diff = max_abs_diff / max(pt_max, 1e-8)
    pt_rms = float(torch.sqrt((pt_flat ** 2).mean()).item())
    trt_rms = float(torch.sqrt((trt_flat ** 2).mean()).item())
    has_nan = bool(torch.isnan(trt_out).any().item())

    result = {
        "cos_sim": cos_sim,
        "max_abs_diff": max_abs_diff,
        "rel_diff": rel_diff,
        "pt_max": pt_max,
        "trt_max": trt_max,
        "pt_rms": pt_rms,
        "trt_rms": trt_rms,
        "has_nan": has_nan,
    }
    log(f"  cos_sim={cos_sim:.4f}  max_abs_diff={max_abs_diff:.4g}  rel_diff={rel_diff:.4g}")
    log(f"  pt_rms={pt_rms:.4g}  trt_rms={trt_rms:.4g}  has_nan={has_nan}")

    passed = (
        not has_nan
        and cos_sim >= CORR_THRESHOLD
        and trt_rms > MIN_RMS
        and rel_diff < 0.5
    )
    result["passed"] = passed
    log(f"  validation: {'PASS' if passed else 'FAIL'}")

    del trt_decoder, ctx, pt_decoder
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ---------- step 5: t2m benchmark + audio compare ----------

def run_benchmark():
    log("running t2m benchmark + audio compare ...")
    import soundfile as sf
    from acestep.constants import TASK_INSTRUCTIONS
    from acestep.engine.session import Session
    from acestep.paths import trt_engine_path

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    TAGS = "dance music, four on the floor, kick drum, electronic, club, energetic synth bass, bright leads"
    LYRICS = "[instrumental]"
    BPM = 128
    KEY = "F minor"
    DURATION = 58.0
    SEEDS = [1528, 42, 9999]
    INFER_STEPS = 8
    SHIFT = 3.0

    def run_pass(label, decoder_backend, trt_engines):
        log(f"  pass={label} decoder={decoder_backend}")
        t0 = time.time()
        session = Session(
            config_path=CHECKPOINT,
            decoder_backend=decoder_backend,
            vae_backend="eager",
            trt_engines=trt_engines,
        )
        log(f"    session ready in {time.time()-t0:.1f}s")
        cond = session.encode_text(
            tags=TAGS, lyrics=LYRICS, instruction=TASK_INSTRUCTIONS["text2music"],
            bpm=BPM, duration=DURATION, key=KEY,
        )
        timings = []
        wavs = {}
        for seed in SEEDS:
            t0 = time.time()
            latent = session.generate(
                conditioning=cond, seed=seed, duration=DURATION,
                steps=INFER_STEPS, shift=SHIFT,
            )
            t_gen = time.time() - t0
            t0 = time.time()
            audio = session.decode(latent)
            t_dec = time.time() - t0
            timings.append({"seed": seed, "gen_s": t_gen, "dec_s": t_dec})
            wav = audio.waveform
            if wav.dim() == 3:
                wav = wav.squeeze(0)
            wav_np = wav.detach().cpu().float().numpy()
            wavs[seed] = wav_np
            sf.write(RESULTS_DIR / f"t2m_xl_turbo_h100_{label}_seed_{seed}.wav",
                     wav_np.T, audio.sample_rate)
            log(f"    seed={seed} gen={t_gen:.3f}s dec={t_dec:.3f}s")
        del session
        gc.collect()
        torch.cuda.empty_cache()
        return timings, wavs

    pt_t, pt_wavs = run_pass("pytorch", "eager", None)
    trt_engines = {"decoder": str(trt_engine_path("decoder_xl-turbo_bf16mix_dynbatch_b8_240s"))}
    trt_t, trt_wavs = run_pass("tensorrt", "tensorrt", trt_engines)

    log("  computing PT vs TRT audio similarity ...")
    audio_compare = []
    for seed in [1528, 42, 9999]:
        pw = pt_wavs[seed].astype(np.float64)
        tw = trt_wavs[seed].astype(np.float64)
        n = min(pw.shape[-1], tw.shape[-1])
        pw = pw[..., :n].flatten()
        tw = tw[..., :n].flatten()
        if pw.std() > 0 and tw.std() > 0:
            corr = float(np.corrcoef(pw, tw)[0, 1])
        else:
            corr = float("nan")
        pt_rms = float(np.sqrt((pw ** 2).mean()))
        trt_rms = float(np.sqrt((tw ** 2).mean()))
        max_diff = float(np.abs(pw - tw).max())
        record = {
            "seed": seed, "samples": int(n),
            "corr": corr, "pt_rms": pt_rms, "trt_rms": trt_rms,
            "max_abs_diff": max_diff,
            "has_nan_trt": bool(np.isnan(tw).any()),
        }
        audio_compare.append(record)
        log(f"    seed={seed} corr={corr:.4f} pt_rms={pt_rms:.4g} trt_rms={trt_rms:.4g}")

    return {
        "pt_timings": pt_t,
        "trt_timings": trt_t,
        "audio_compare": audio_compare,
        "all_audio_passed": all(
            (not r["has_nan_trt"]) and r["trt_rms"] > MIN_RMS and r["corr"] > CORR_THRESHOLD
            for r in audio_compare
        ),
    }


# ---------- main ----------

def main():
    report = {
        "checkpoint": CHECKPOINT,
        "stages": {},
        "build_attempts": [],
        "validation": None,
        "benchmark": None,
        "final_status": "incomplete",
    }

    try:
        log("=" * 70)
        log("H100 XL TURBO PIPELINE v2 (subprocess-isolated builds)")
        log("=" * 70)
        import tensorrt as trt
        log(f"orchestrator TRT version: {trt.__version__}")
        log(f"GPU: {torch.cuda.get_device_name(0)}")

        # ---- step 1: export ONNX(s) ----
        # Always export bf16_mixed; export bf16 only if at least one config needs it
        needs_bf16 = any(c[1] == "bf16" for c in BUILD_CONFIGS)
        try:
            log("step1: ONNX export (bf16_mixed) ...")
            export_onnx("bf16_mixed", ONNX_PATH)
            patch_dynbatch(ONNX_PATH, ONNX_DYNBATCH_PATH)
            if needs_bf16:
                log("step1b: ONNX export (bf16 pure) ...")
                export_onnx("bf16", ONNX_BF16_PATH)
                patch_dynbatch(ONNX_BF16_PATH, ONNX_BF16_DYNBATCH_PATH)
            report["stages"]["export"] = "ok"
        except Exception as e:
            report["stages"]["export"] = f"FAIL: {e}"
            traceback.print_exc()
            raise

        # ---- step 3: try build configs (subprocess each) ----
        winning = None
        for cfg in BUILD_CONFIGS:
            label = cfg[0]
            onnx_kind = cfg[1]
            onnx_path = ONNX_DYNBATCH_PATH if onnx_kind == "mix" else ONNX_BF16_DYNBATCH_PATH

            attempt = {"label": label, "onnx": onnx_kind,
                       "strongly_typed": cfg[2], "opt": cfg[3],
                       "batch_max": cfg[4], "set_bf16": cfg[5], "set_fp16": cfg[6]}

            ok, info = try_build(cfg, onnx_path)
            attempt["build"] = info
            attempt["build_ok"] = ok

            if not ok:
                report["build_attempts"].append(attempt)
                # save report after each attempt for incremental visibility
                with open(REPORT_PATH, "w") as f:
                    json.dump(report, f, indent=2, default=str)
                continue

            try:
                vresult = validate_decoder_engine()
                attempt["validation"] = vresult
            except Exception as e:
                attempt["validation"] = f"exception: {type(e).__name__}: {e}"
                report["build_attempts"].append(attempt)
                with open(REPORT_PATH, "w") as f:
                    json.dump(report, f, indent=2, default=str)
                traceback.print_exc()
                continue

            report["build_attempts"].append(attempt)
            with open(REPORT_PATH, "w") as f:
                json.dump(report, f, indent=2, default=str)

            if vresult.get("passed"):
                log(f"\n*** WINNER: {label} ***")
                winning = label
                report["validation"] = vresult
                break
            else:
                log(f"  {label} built but failed validation, trying next")

        if winning is None:
            report["final_status"] = "no_working_engine"
            log("\nNO WORKING ENGINE across all configs")
            return report

        report["winning_config"] = winning

        # ---- step 5: benchmark + audio compare ----
        try:
            bench = run_benchmark()
            report["benchmark"] = bench
            if bench["all_audio_passed"]:
                report["final_status"] = "success"
                log("\n*** PIPELINE SUCCESS ***")
            else:
                report["final_status"] = "audio_mismatch"
                log("\nbenchmark ran but audio compare failed")
        except Exception as e:
            report["benchmark"] = f"exception: {type(e).__name__}: {e}"
            report["final_status"] = "benchmark_crashed"
            traceback.print_exc()

    finally:
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log(f"\nreport saved to {REPORT_PATH}")
        log(f"final_status={report['final_status']}")

    return report


if __name__ == "__main__":
    main()
