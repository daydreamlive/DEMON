# YuE2 demo page

A minimal page for the `yue2` family: YuE2 composes a song from a style
prompt and lyrics, then the acoustic flow-matching stage runs in the
DEMON ring so the knobs reshape the song while it loops.

## Run

Set the YuE2 environment (see the YuE2 section of `docs/FAMILIES.md`):

```bash
export DEMON_YUE2_ROOT=/path/to/yue2            # weights (YuE2-3B + YuE2-Vae)
export DEMON_YUE2_YUE_SRC=/path/to/YuE/src      # upstream YuE source at the pinned revision
export DEMON_YUE2_EXTRA_PATH=/path/to/extra-deps  # optional extra import path
export DEMON_YUE2_TRT_DIR=/path/to/engines      # optional; eager NAR without it
python -u -m demos.realtime_motion_graph_web.server --port 1318 --checkpoint yue2-3b
```

Open http://localhost:1318/yue2/.

## Use

- **Style / Style B**: tag prompts. Leave B empty for one song; a
  distinct B composes a second song from the same lyrics at the same
  length (create takes about twice as long) and the A/B switch flips
  between the two songs.
- **Lyrics** and **Longest song** are fixed once you press Start: they decide
  the composition, which takes seconds of AR before audio starts
  ("composing").
- Audio starts from the composed song (the anchor) and loops.
- **yue2_denoise**: how much of the 32-step solve each pass re-runs.
- **x0_target**: pull toward the anchor.
- **seed** (number box): new acoustic noise for the same composition.
- **feedback** / **feedback depth**: blend one of the last few outputs
  back into the source (the shared ACE / SA3 delay tap); with feedback
  above 0 the ring never settles and keeps regenerating.
- The ring runs at depth 1 whatever the client asks: the ring settles
  after each change, so a second slot buys no throughput; on a 5090
  depth 2 ran a 102-124 ms tick against 70 ms and every update was
  40-75% slower.
- A failed re-compose shows its message in the status line; the current
  song keeps playing.
- **Re-compose** sends the current Style (and B): the song keeps playing
  while YuE2 composes the new one in the background, then the new song
  replaces it. A YuE2 style lives in its semantic tokens, so a new style
  means a new composition (same lyrics, same length), not a re-colouring
  of the current one.

YuE2 weights are CC BY-NC 4.0 (non-commercial).
