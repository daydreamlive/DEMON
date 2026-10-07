# Ship runbook: every shipped SA3 steering knob under one protocol (drafted 2026-10-06, NOT LAUNCHED)

Purpose: the minimum work after which every pack in the shipped set carries numbers measured the same way, on the same
vector, at the gain that ships. Inputs: inventory_many_knobs_v2_2026-10-06.md (gaps 1-5, 8, 10),
astra_second_opinion_many_knobs_v2_2026-10-06.md (Q1). Box 51480126 shared; same rules as many_knobs_v2_runbook.md.
Data root E:\Projects\DEMON\steering-bench\many_knobs_v2\ship\; B2 prefix sa3/steering_bench/many_knobs_v2_ship_2026-10-06/.

## Decision rules, frozen before any render (CPU, step 0)
- Quality bar (absolute, every origin, no exemptions): on at least one sign, CLAP and MuQ intended-sign fraction >= 0.6
  as the median over seeds 0,1,2 at the SHIPPED gain; cutoff reached on that sign on at least 2 of 3 seeds.
- Shipped gain per sign = median over seeds 0,1,2 of the calibrated gain (first probe at or above the PCI-all cutoff).
- Regression gate (only where a v1 and a v2 vector exist for the same concept): both measured by this runbook; v2 wins if
  fractions >= v1 - 0.05 on both scorers, LPAPS/cutoff <= 1.1 x v1's, reach >= v1. No counterpart = "not compared", never
  "won". A concept whose every vector fails the bar ships nothing (this is the sfx_laden case; "never regress" yields to
  the bar, stated in the table).
- Hold drift and cross-effect are reported, not gating (no threshold exists yet; see followups).
- applies_to: measured categories only; no catalogue default when the measurement is empty.
- c packs: header neg gain stored in the dn vector's own units; the runtime applies it to vectors_neg. One unit, documented.

## Vectors to measure (51 old vectors, unchanged bytes)
39 kept_v1 + 5 legacy (bright, dense_arrangement, percussive, rough, warm) + the 6 displaced v1 vectors (aggressive,
black_metal, jet, polyrhythmic, rain_on_surface, ukulele) + v1 sfx_laden. Packs from packs_bench / the trash copies; no
re-extraction, no screening.

## Shipped gain = interpolated to the cutoff (folded in, no extra GPU)
The v2 gain is "first probe at or above cutoff" (median overshoot 1.19x). For every knob-sign and seed, interpolate the gain
between the two probes that bracket the PCI-all cutoff on the LPAPS curve (log-linear in alpha). Shipped gain = median over
seeds of the interpolated gains. The fixed-gain pass below renders at that gain, so it validates the interpolation (LPAPS at
shipped gain is reported per seed). Old probe data for the 63 new knobs already exists; the 51 old vectors get it from stage A.

## GPU work: one priority queue, 8 drivers (2 per GPU), no stage barriers
Same claim mechanism as many_knobs_v2_runbook.md (queue.txt + mkdir claims). Drivers always take the lowest prio available;
CPU stages (scoring, interpolation, bar, gate, make_packs) run alongside and append items as they become ready. A GPU is never
waiting on a stage: items of different stages interleave by prio the moment their inputs exist.

P0 (enqueue at t=0, 342 items, all independent):
  - A: protocol seeds 0,1,2 on the 51 old vectors = 153 items minus the 14 legacy items already done = 139 (warm s2 + 39 kept
    + 6 displaced + sfx_laden, x3 seeds). PCI dirs per seed reused from last night.
  - B-new: fixed-gain pass for the 63 new knobs at the interpolated shipped gain: 63 knobs x 3 seeds = 189 items (both signs,
    50 prompts, each item ~2 min). Gains come from CPU interpolation of last night's probes, ready before the box starts.
  - SFX prompt pass: the 20 SFX-category packs scored on the 12 sfx applicability prompts x 3 seeds at shipped gain = 60 items
    (so SFX knobs carry a fraction measured on SFX prompts; CLAP+MuQ both recorded, bar decision per followups).
P1 (appended per knob as A finishes its 3 seeds and the CPU interpolates the gain): B-old fixed-gain pass, 51 x 3 = 153 items.
P2 (appended per knob as soon as its B items are scored): C = hold (6 prompts, 30 s) + applicability (36 prompts) + cross-effect
  render for every knob at shipped gain. Run for ALL 114 vectors, not only bar passers (the bar is decided on CPU in parallel;
  discarding a few wasted items is cheaper than a barrier). New-63: skip the knob if its P4/crossfx gain at 07:45Z/08:25Z is
  within 5% of the new shipped gain (CPU list, decided before launch). 
P3 (appended when the final set is known): D = stacking, 300 combos at shipped gains x 4 rules = 1200 items of 12 prompts.
P4 filler: alpha-0 refs for any prompt/seed not yet rendered (should be none), then an extra seed (3) of B for every knob.

Ordering inside P0: interleave A and B-new round-robin so that the P1 pipeline (which depends on A) fills early and B-new
never starves the queue of A's outputs. Enqueue A in order kept_v1 (39) first, then displaced 6, sfx_laden, warm.

Cost (rates from last night: P2 protocol 153 packs x 1 seed = 4.7 h on 8 drivers, i.e. ~15 min per pack-seed per driver;
fixed-gain item ~2 min; hold/app/crossfx ~6 min per knob; stacking item ~0.5 min):
  A 139 x 15 min / 8 = 4.3 h  (the long pole; nothing else can shorten it except the legacy reuse already applied)
  B 342 x 2 min / 8 = 1.4 h; SFX 60 x 2 / 8 = 0.25 h; C 114 x 6 / 8 = 1.4 h; D 1200 x 0.5 / 8 = 1.25 h
  Sum of driver time ~8.6 h / 8 drivers = GPU-bound ~8.6 h of driver work across 4 GPUs, i.e. about 4.5 to 5 h wall with the
  queue kept full (A dominates; B/C/D fill the gaps as A knobs complete). The earlier "~3 h" was wrong: it omitted B for the
  63 and the 7 extra old vectors. Hard floor is A alone at 4.3 h.
Accounting: per-GPU items done and idle minutes to the status file hourly; idle minutes > 5 while queue non-empty = FLAG.

## CPU work after renders
make_packs for the whole shipped set with ONE metadata shape (dict per sign: median, min, max, reached[], seeds, cutoff;
the v1 flat shape retired); provenance.kept_from kept; results table: knob, origin, variant, blocks, gain pos/neg with seed
spread, fractions at shipped gain, bar pass, gate outcome (compared / not compared), hold drift signed, cross-effect, applies_to.
Install to ~/.daydream-scope/models/demon/steering_packs/sa3/medium/ (old 107 to ~/.claude-trash/), restart the backend.
B2 mirror, then local copy, then status + results files, then hand back (<= 10 lines).

## Expected outcome
Around 101 to 102 knobs. Known bar failures on current numbers: close_mic, live_recording, sensual, wide_stereo, warm;
sfx_laden stays out. Anything else that drops is reported with its numbers, not silently.
