# YuE2 demo page

A minimal page for the `yue2` family: YuE2 composes a song from a style
prompt and lyrics, then the acoustic flow-matching stage runs in the
DEMON ring so the knobs reshape the song while it loops.

## Run

Set the YuE2 environment (see the YuE2 section of `docs/FAMILIES.md`):

```bash
export DEMON_YUE2_ROOT=/path/to/yue2            # weights (YuE2-3B + codec)
export DEMON_YUE2_YUE_SRC=/path/to/YuE/src      # upstream YuE source at the pinned revision
export DEMON_YUE2_EXTRA_PATH=/path/to/extra-deps  # optional extra import path
export DEMON_YUE2_TRT_DIR=/path/to/engines      # optional; eager NAR without it
python -u -m demos.realtime_motion_graph_web.server --port 1318 --checkpoint yue2-3b
```

Open http://localhost:1318/yue2/.

## Use

- **Style A / Style B**: tag prompts. B is a restyle of the same
  composition (same lyrics and semantic tokens); the A/B switch flips
  between them.
- **Lyrics** and **Duration** are fixed once you press Start: they decide
  the composition, which takes seconds of AR before audio starts
  ("composing").
- Audio starts from the composed song (the anchor) and loops.
- **yue2_denoise**: how much of the 32-step solve each pass re-runs.
- **x0_target**: pull toward the anchor.
- **seed**: new acoustic noise for the same composition.
- **Send style** applies a new Style A while the song plays.

YuE2 weights are CC BY-NC 4.0 (non-commercial).
