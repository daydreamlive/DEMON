# /mrt2: Magenta RealTime 2 demo

A plain 2D canvas draws 64 spectrum bars from the live Magenta RealTime 2 stream;
their colour moves from cool (prompt A) to warm (prompt B) with the blend slider.
No third-party dependencies: one JS module on /sdk/demon-client.js, like /sa3.

Prerequisites: the MRT2 sidecar running in a Linux/WSL venv with `magenta_rt` + JAX
(`python scripts/mrt2_sidecar.py --model mrt2_small`, listens on 127.0.0.1:7531 after a
~30 s JIT warmup; override with `DEMON_MRT2_SIDECAR=host:port`). See docs/FAMILIES.md.

Server:

    python -u -m demos.realtime_motion_graph_web.server --port 1318 --checkpoint mrt2-sidecar

Open http://localhost:1318/mrt2/ and press Start. The status line shows connecting,
then waiting for first audio, then playing with the seconds buffered ahead of the playhead.
Audio is append-only: prompt, blend and knob changes are heard after roughly `mrt2_lead`
seconds (0.75 s default). Uploaded audio is ignored; the page sends a short silent stub.

Knobs (from the session's knob manifest, `mrt2_` prefix only; the family declares no
generic knobs): `mrt2_temperature`, `mrt2_top_k`, `mrt2_cfg_musiccoca`, `mrt2_cfg_notes`,
`mrt2_cfg_drums`, `mrt2_lead`. Prompts and blend are live; nothing here needs a reconnect.
