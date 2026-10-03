# MiniMax-Music3 demo (`/minimax`)

A point-cloud sphere that swells with the bass and ripples with the mids while
MiniMax-Music3 composes a continuous piece from a style prompt (and optional lyrics).

Prerequisites: the MiniMaxAI/MiniMax-Music3 weights in the HF cache or under
`<models>/checkpoints/MiniMax-Music3` (`DEMON_MINIMAX_DIR` overrides), a GPU with
more than 20 GB free. Optional DiT TensorRT engine dir: `DEMON_MINIMAX_TRT_DIR`.

Run the backend from the repo root:

    python -u -m demos.realtime_motion_graph_web.server --port 1318 --checkpoint minimax-music3

Open http://localhost:1318/minimax and press Start.

- Style prompt: "Send prompt" restyles the running piece within a few seconds.
- Lyrics and Duration are handshake fields: they apply only on Reconnect (new session).
- Knobs: every `minimax_*` knob in the session's manifest (temperature, top k, ar guidance,
  guidance, shift, cond strength, hop, steps, reprompt history s, endless, lead). The page
  turns `endless` on and sets `reprompt history s` to 2.5 so prompt changes pivot quickly.

First seconds: the status reads "waiting for first audio" for about 10 s (the LM writes
its first frames and the renderer commits one chunk), then "playing" with the seconds of
audio buffered ahead of the playhead.
