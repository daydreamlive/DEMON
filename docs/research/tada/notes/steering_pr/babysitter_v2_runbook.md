# Babysitter runbook: many-knobs v2 overnight (2026-10-05 23:00Z to 2026-10-06 12:00Z)

You are an independent watchdog for the run described in many_knobs_v2_runbook.md (read it, box_status.md, and
status_many_knobs_v2.md first). The run is executed by launcher-3 (agent id ad76a48855a032725); you do not do its
work, you make sure it never stalls. Box 51480126 (4x RTX 5090, shared with other lanes). Tree /root/DEMON-steerbench,
data /dev/shm/steerbench/out, eval dir many_knobs_v2_eval (status file names the exact paths).

## Token rule (absolute)
No sleep loops, no polling, no periodic wakes, no "still fine" reports. You wake ONLY when an alarm fires. The box does
the watching with a shell script; you arm ONE Monitor on an ssh tail of its alarm log with an until-condition that
matches lines starting with ALARM. HEARTBEAT lines (one every 2 h) must NOT wake you. If the ssh stream drops, re-arm
once; that is the only non-alarm wake. Report to the parent (SendMessage to "main") only on a state change you caused
or a terminal state; at most 8 lines. Silence means fine.

## Watchdog script (install once, nohup, on the box under /dev/shm/steerbench/logs/)
Every 10 min, a shell loop (not you) samples: nvidia-smi utilisation per GPU; mtime of the status file's box-side
twin (launcher-3 writes many_knobs_v2_eval/p6_status.txt or similar, find the file it actually updates); count of live
driver processes (the claim drivers from logs/p6_claim.sh or whatever logs/ script launcher-3 uses, by cmdline);
queue.txt lines minus claims; /dev/shm free. Writes ALARM <reason> when: any GPU < 15% util in 3 consecutive samples
while the queue is non-empty; no live driver on a GPU while the queue is non-empty; /dev/shm free < 4 GB; the
launcher's own status output not updated for 120 min; clock >= 10:35Z and drivers still claiming (freeze missed); clock
>= 11:40Z and no results_many_knobs_v2.md (hand-back missed). Writes HEARTBEAT every 2 h. Never writes anything else.

## On ALARM
1. Diagnose from the box in at most 5 commands (logs/, nvidia-smi, ps, df, the status file). Do not touch other lanes.
2. Message launcher-3 (SendMessage to ad76a48855a032725) with the reason and what you see; arm one Monitor on the
   status file mtime (ssh stat loop, until changed) for 20 min.
3. If launcher-3 fixed it: say nothing, re-arm the alarm Monitor.
4. If not, fix it yourself within the runbook: relaunch dead claim drivers with the same script launcher-3 used
   (2 per GPU, expandable_segments); if disk, move already-mirrored v1 eval dirs to /dev/shm/steerbench/steer-bench-trash
   and empty it (standing authorisation: the trashcan only); if the freeze was missed, stop ONLY the claim drivers
   launcher-3 started and tell launcher-3 to run the freeze steps; if the hand-back is missed at 11:40Z, run the
   freeze steps yourself from the runbook (packs, results, scp to E:\Projects\DEMON\steering-bench\many_knobs_v2\,
   B2 via /dev/shm/steerbench/b2_mirror.sh shape) and report to main.
5. Permission refusal on anything: do not retry; message main with the exact command and continue with what else
   is possible.

## Never
Stop, reboot or destroy the box; kill a process you did not start (except launcher-3's claim drivers in step 4, which
are this lane's); delete anything outside the trashcan; rm locally (mv to ~/.claude-trash/); push; add attribution.
