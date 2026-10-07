# SA3 steering demo: how to use

URL: http://localhost:1318/sa3/   (worktree DEMON-steer, branch ryanontheinside/feat/steering-generic)
Launch: `bash C:/_dev/projects/DEMON/notes/family_merge/demo_logs/LAUNCH_sa3_steer.sh` (foreground; log in demo_logs/sa3_steer.log).
Server is ready ~20 s after launch (log line `server_ready`). Checkpoint alias `sa3-medium`.

## What to click
1. Open the URL, pick a fixture and a prompt, press **Start** (same as the plain /sa3 page).
2. Under the knob grid a **Steer** section appears with one slider per steering pack found for SA3 medium:
   bright, density, percussive, rough, warm. Range -30..+30, default 0 (off). Double-click a slider to reset it to 0.
3. Knob 10 adds one full prompt-pair mean difference at the pack's block; 1..10 is the useful range, beyond ~15 it gets extreme.
4. The section is hidden when the server has no packs (then the normal fp8 engine runs, ~9 ms per tick faster).

## What each slider does (measured, not listened)
Headless client on this server, fixture low_fi_Gm_loop_60s_gnm.wav, proxy over the last 6 s heard, knob 0 / +10 / -10 / back to 0:
- bright (centroid Hz): 589 / 1860 / 169 / 500. Clear, right way.
- warm (low/high dB): 12.8 / 58.0 / -8.8 / 17.2. Clear, right way.
- percussive (perc ratio): 0.146 / 0.134 / 0.045 / 0.135. -10 strips drums; +10 weak on this loop.
  (Text-to-music sanity: 3/3 prompts right way at +/-1 and +/-10.)
- rough, density: the spectral proxies do not track these concepts (prompt-pair sign agreement -0.25 / -0.56);
  the sliders change the sound but direction is unverified. Listen before relying on them.
Knob 0 is a bit-identical no-op (checked offline on the same engine: packs at 0 vs no packs).

## Timing
- Steering DiT engine (sa3_m_dit_steer_l1_646_646) is selected only when packs exist: tick ~52 ms vs ~43 ms on fp8.
- Knob changes ride the params channel; effect heard after the next generation (~1.5 s).

## Stop
    netstat -ano | findstr ":1318"        # find PID
    taskkill /F /PID <pid>
Confirm `netstat -ano | findstr ":1318 :1319"` empty and nvidia-smi back to ~4 GB.
