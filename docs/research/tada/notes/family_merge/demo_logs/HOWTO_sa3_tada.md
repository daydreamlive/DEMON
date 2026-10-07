# SA3 TADA demo: how to use

URL: http://localhost:1318/sa3/   (worktree DEMON-tada-sa3, branch ryanontheinside/feat/tada-sa3)
Launch: `bash C:/_dev/projects/DEMON/notes/family_merge/demo_logs/LAUNCH_sa3_tada.sh` (foreground; log in demo_logs/sa3_tada.log).
Server is ready ~15 s after launch (log line `server_ready`). Checkpoint alias `sa3-medium`.
Packs: E:/Projects/tada-replication/packs/sa3/medium (via DEMON_STEERING_PACKS_DIR). Nothing in the page names a concept.

## What to click
1. Open the URL, pick a fixture and a prompt, press **Start**.
2. The **Steer** section shows one slider per pack: 9 TADA CAA concepts (electronic music, guitar electronic, mood, piano,
   rock genre, tempo, violin, vocal gender, vocal style) and the same 9 as AUSteer (`..._austeer`). Range -30..+30, 0 = off.
3. Full scale (+-30) equals the calibrated strength at which that concept's audio drifts as far as a full prompt swap
   (PCI cutoff); +-10 to +-20 is the useful range. Positive = paper's positive pole (piano, happy, fast, female vocal,
   rap vocal, acoustic guitar, violin, jazz, classical), negative = the opposite pole.
4. The sliders steer the cross-attention outputs of blocks 3, 6, 7 (TADA's localised set for SA3), every step.

## What to expect (measured)
- Offline benchmark: only tempo steers clearly (MuQ AUC 0.05); rock genre and vocal style weakly; the rest are near noise.
  Expect audible change at strength, but the concept may not follow. See status_tada_sa3.md RESULTS.
- Headless check 2026-10-04: steer_tempo +30 raised the onset rate from ~2 to 3 per s to 6.5 per s.
- Knob 0 builds no steering at all (identical path to unset).

## Timing
- With TADA packs present the session runs the cross-attention steering DiT engine (sa3_m_dit_steerxa_l1_646_646),
  12.8 ms per DiT step vs 10.5 ms on fp8.
- Knob changes ride the params channel; heard after the next generation (~1.5 s).

## Stop
    netstat -ano | findstr ":1318"        # find PID
    taskkill /F /PID <pid>
Confirm `netstat -ano | findstr ":1318 :1319"` empty and nvidia-smi back to idle.
