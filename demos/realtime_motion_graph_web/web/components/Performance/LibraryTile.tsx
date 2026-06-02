"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
} from "react";

import { loraStrengthDispatcher } from "@/engine/lora/dispatcher";
import { listHiddenLoras, listLoras } from "@/engine/lora/listLoras";
import { useConfig } from "@/lib/config";
import { LORA_LABELS, loraDisplayName } from "@/lib/loraLabels";
import { LOCAL_MODE } from "@/lib/runtime";
import { isLoraCompatibleWithScale, useLoraStore } from "@/store/useLoraStore";
import { useMidiStore } from "@/store/useMidiStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { LORA_SLIDER_MAX } from "@/types/engine";
import type { LoraCatalogEntry, LoraMetadata } from "@/types/protocol";

// LoRA library tile — redesigned for a large catalog (40+ genre LoRAs).
//
// The old layout was one long scroll of full-width rows, each carrying an
// always-visible strength slider, with the description trapped in a hover
// tooltip. With 40+ LoRAs that meant scrolling a wall of sliders and
// fighting tooltips. This layout splits the tile in two:
//
//  ── ACTIVE rack ──  Only the *enabled* LoRAs, each with its strength
//     slider. Usually 1–4 entries, so there is no slider wall — you only
//     ever see sliders for things you are actually using.
//
//  ── BROWSE accordion ──  Every LoRA, grouped into collapsible genre
//     categories (Electronic / Rock / Pop / …), all collapsed by default.
//     Each browse row is a compact click-to-enable button showing the
//     name + a short inline description (no hover tooltip). Enabling a
//     LoRA pops it up into the ACTIVE rack with a slider.
//
// A search box at the top filters across name + description + tags +
// genre; while a query is active the accordion flattens to a plain
// results list. Right-click any row → shared MIDI context menu with a
// Copy trigger action. Enabling/disabling a LoRA does not mutate the Tags A/B
// textareas — the trigger word rides the WS `prompt` message, injected by
// RemoteBackend.sendPrompt (enabledLoraTriggerPrefix), and the toggle
// re-sends the prompt so the engine re-encodes with the new trigger set.

// ── Genre → category map ────────────────────────────────────────────────
//
// Ryan's 40 acestep1.5 LoRAs each carry their own `primary_genre`, so
// grouping by genre directly would yield 40 groups of one. This static
// map folds the genres into a handful of browsable categories. Genres
// not listed (and the older daydreamlive LoRAs) fall through to "Other".

const GENRE_CATEGORY: Record<string, string> = {
  // Electronic
  edm: "Electronic",
  electronic: "Electronic",
  electropop: "Electronic",
  house: "Electronic",
  deep_house: "Electronic",
  techno: "Electronic",
  dubstep: "Electronic",
  future_bass: "Electronic",
  synthwave: "Electronic",
  synth_pop: "Electronic",
  trap: "Electronic",
  phonk: "Electronic",
  industrial: "Electronic",
  // Rock
  rock: "Rock",
  hardrock: "Rock",
  alternative: "Rock",
  alternative_rock: "Rock",
  indie_rock: "Rock",
  progressive_rock: "Rock",
  psychedelic_rock: "Rock",
  post_punk: "Rock",
  punk: "Rock",
  grunge: "Rock",
  metal: "Rock",
  metalcore: "Rock",
  emo: "Rock",
  shoegaze: "Rock",
  // Pop
  pop: "Pop",
  j_pop: "Pop",
  dream_pop: "Pop",
  indie: "Pop",
  // Hip-Hop
  hip_hop: "Hip-Hop",
  rap: "Hip-Hop",
  // Acoustic & Folk
  acoustic: "Acoustic & Folk",
  folk: "Acoustic & Folk",
  jazz: "Acoustic & Folk",
  // Chill & Other
  ambient: "Chill & Other",
  lo_fi: "Chill & Other",
  funk: "Chill & Other",
  experimental: "Chill & Other",
  r_b: "Chill & Other",
};

const CATEGORY_ORDER = [
  "Electronic",
  "Rock",
  "Pop",
  "Hip-Hop",
  "Acoustic & Folk",
  "Chill & Other",
  "Other",
];

function normGenre(g: string | null | undefined): string {
  return (g ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function categoryOf(entry: LoraCatalogEntry): string {
  return GENRE_CATEGORY[normGenre(entry.metadata?.primary_genre)] ?? "Other";
}

function displayNameOf(entry: LoraCatalogEntry): string {
  return loraDisplayName(entry);
}

function byDisplayName(a: LoraCatalogEntry, b: LoraCatalogEntry): number {
  return displayNameOf(a).localeCompare(displayNameOf(b));
}

// ── Search ──────────────────────────────────────────────────────────────

function matchesQuery(entry: LoraCatalogEntry, q: string): boolean {
  if (!q) return true;
  const needle = q.trim().toLowerCase();
  if (!needle) return true;
  const md = entry.metadata;
  const parts: (string | null | undefined)[] = [
    entry.id,
    entry.name,
    LORA_LABELS[entry.id],
    md?.name,
    md?.description,
    md?.primary_trigger_word,
    md?.primary_genre,
    ...(md?.tags ?? []),
    ...(md?.moods ?? []),
    ...(md?.secondary_genres ?? []),
    ...(md?.trigger_words ?? []),
  ];
  return parts
    .filter((s): s is string => typeof s === "string" && s.length > 0)
    .some((s) => s.toLowerCase().includes(needle));
}

// ── Inline description ──────────────────────────────────────────────────
//
// Replaces the old hover tooltip. Returns a short line shown directly
// under the LoRA name in browse rows — the real description when present,
// otherwise a moods/recommended-strength fallback.

function shortDescription(md: LoraMetadata | undefined): string | undefined {
  if (!md || !md.has_metadata) return undefined;
  if (md.description) return md.description;
  const bits: string[] = [];
  if (md.moods && md.moods.length > 0) {
    bits.push(md.moods.slice(0, 3).join(", "));
  }
  if (md.recommended_strength != null) {
    bits.push(`rec. strength ${md.recommended_strength.toFixed(2)}`);
  }
  return bits.length > 0 ? bits.join(" · ") : undefined;
}

// ── Context menu ────────────────────────────────────────────────────────

// ── Clipboard ───────────────────────────────────────────────────────────

async function copyTriggerToClipboard(
  trigger: string,
  onFlash: (text: string) => void,
): Promise<void> {
  try {
    if (
      typeof navigator !== "undefined" &&
      navigator.clipboard &&
      window.isSecureContext
    ) {
      await navigator.clipboard.writeText(trigger);
    } else {
      _legacyCopy(trigger);
    }
    onFlash(`Copied "${trigger}"`);
  } catch {
    onFlash("Copy failed");
  }
}

function _legacyCopy(text: string): void {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  try {
    document.execCommand("copy");
  } finally {
    document.body.removeChild(ta);
  }
}

// ── Shared row hooks ────────────────────────────────────────────────────

// Enable/disable a LoRA and immediately re-send the prompt so the engine
// re-encodes with the new enabled-LoRA trigger set. The promptA/promptB
// read here are the operator's CLEAN textarea text; sendPrompt injects
// the trigger prefix on the wire (see enabledLoraTriggerPrefix).
function useLoraToggle() {
  const enable = useLoraStore((s) => s.enable);
  const disable = useLoraStore((s) => s.disable);
  return useCallback(
    (id: string, currentlyEnabled: boolean) => {
      const remote = useSessionStore.getState().remote;
      if (currentlyEnabled) {
        disable(id);
        remote?.sendDisableLora(id);
      } else {
        // Cap-aware: consult the store BEFORE the WS round-trip so we
        // don't ship a sendEnableLora for a LoRA the store will refuse
        // to add (server would materialize ~1.2 GB of refit state for
        // a UI that won't reflect it). The store's enable() also
        // enforces the cap defensively — this is the visible side.
        if (!useLoraStore.getState().canEnableMore()) return;
        enable(id);
        const s = useLoraStore.getState().strengths[id] ?? 0;
        remote?.sendEnableLora(id, s);
      }
      const perf = usePerformanceStore.getState();
      remote?.sendPrompt(
        perf.promptA,
        perf.activeKey,
        perf.activeTimeSignature,
        perf.promptB,
      );
    },
    [enable, disable],
  );
}

// Transient confirmation text (e.g. "Copied 'word'") that briefly
// replaces a row's name. Replaces the old data-dd-tooltip-show flash.
function useConfirmFlash(): [string | null, (text: string) => void] {
  const [msg, setMsg] = useState<string | null>(null);
  const timerRef = useRef<number | null>(null);
  const flash = useCallback((text: string) => {
    setMsg(text);
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => {
      setMsg(null);
      timerRef.current = null;
    }, 1500);
  }, []);
  useEffect(
    () => () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    },
    [],
  );
  return [msg, flash];
}

function openLoraMidiMenu(
  e: ReactMouseEvent<HTMLElement>,
  id: string,
  trigger: string | null,
  flashConfirm: (text: string) => void,
): void {
  e.preventDefault();
  e.stopPropagation();
  useMidiStore.getState().openMenu({
    x: e.clientX,
    y: e.clientY,
    kind: "cc",
    target: `lora_str_${id}`,
    el: e.currentTarget,
    extraItems: trigger
      ? [
          {
            label: `Copy trigger "${trigger}"`,
            dividerBefore: true,
            onClick: () => {
              void copyTriggerToClipboard(trigger, flashConfirm);
            },
          },
        ]
      : undefined,
  });
}

// ── Active rack row ─────────────────────────────────────────────────────

function ActiveLoraRow({ entry }: { entry: LoraCatalogEntry }) {
  const { id } = entry;
  const strength = usePerformanceStore(
    (s) => s.sliderTargets[`lora_str_${id}`],
  );
  const fallbackStrength = useLoraStore((s) => s.strengths[id] ?? 0);
  const value = typeof strength === "number" ? strength : fallbackStrength;
  const toggle = useLoraToggle();
  // Refit-pending lock: after a user commit gesture the slider is
  // locked until the LoRA store's REFIT_LOCK_MS timer expires. UI
  // disables pointer input on the track during this window so users
  // can't queue rapid commits the server's refit pipeline can't keep
  // up with. MIDI/curves/MCP-driven changes bypass this — they don't
  // go through pointer gestures and shouldn't be UX-rate-limited.
  const isRefitPending = useLoraStore((s) => s.pendingRefit.has(id));

  const md = entry.metadata;
  const displayName = displayNameOf(entry);
  const trigger = md?.primary_trigger_word ?? null;

  const rowRef = useRef<HTMLDivElement | null>(null);
  const trackRef = useRef<HTMLDivElement | null>(null);
  const [confirmMsg, flashConfirm] = useConfirmFlash();

  const setFromClientX = useCallback(
    (clientX: number) => {
      const el = trackRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const t = (clientX - rect.left) / rect.width;
      loraStrengthDispatcher.set(id, Math.max(0, Math.min(1, t)) * LORA_SLIDER_MAX);
    },
    [id],
  );

  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    let dragging = false;
    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0) return; // right-click → MIDI learn (context menu)
      // Refuse new gestures while a prior commit's refit window is
      // still open. The dispatcher owns the lock (marked at commit
      // time, not on pointerup) — going through dispatcher.isLocked
      // keeps every pointer entry point (LibraryTile, DesktopEdgeDrag,
      // useLoraFaderDrag) reading from one source of truth.
      if (loraStrengthDispatcher.isLocked(id)) return;
      dragging = true;
      el.setPointerCapture(e.pointerId);
      setFromClientX(e.clientX);
    };
    const onPointerMove = (e: PointerEvent) => {
      if (dragging) setFromClientX(e.clientX);
    };
    const onPointerUp = (e: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      el.releasePointerCapture(e.pointerId);
      // No lock call here — the dispatcher's debounced commit (300 ms
      // after the last pointermove) marks the refit window itself.
      // A v1 lock fired here cleared too early because it ignored
      // the DEBOUNCE_MS wait before commit.
    };
    el.addEventListener("pointerdown", onPointerDown);
    el.addEventListener("pointermove", onPointerMove);
    el.addEventListener("pointerup", onPointerUp);
    el.addEventListener("pointercancel", onPointerUp);
    return () => {
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("pointermove", onPointerMove);
      el.removeEventListener("pointerup", onPointerUp);
      el.removeEventListener("pointercancel", onPointerUp);
    };
  }, [id, setFromClientX]);

  const pct = Math.max(0, Math.min(1, value / LORA_SLIDER_MAX)) * 100;
  return (
    <div
      ref={rowRef}
      className={`lora-active-row${isRefitPending ? " refit-pending" : ""}`}
      data-param={`lora_str_${id}`}
      data-midi-kind="cc"
      data-midi-target={`lora_str_${id}`}
      data-dd-tooltip-title={displayName}
      onContextMenu={(e) => openLoraMidiMenu(e, id, trigger, flashConfirm)}
    >
        <span className="lora-active-name" title={displayName}>
          {confirmMsg ?? displayName}
        </span>
        <button
          type="button"
          className="lora-active-remove"
          onClick={() => toggle(id, true)}
          aria-label={`Disable ${displayName}`}
        >
          ✕
        </button>
        <div className="lora-strength">
          <div
            className="lora-strength-track"
            ref={trackRef}
            aria-busy={isRefitPending}
            title={
              isRefitPending
                ? "Applying… (LoRA refit in progress)"
                : undefined
            }
          >
            <div className="lora-strength-fill" style={{ width: `${pct}%` }} />
            <div
              className="lora-strength-thumb"
              style={{ left: `${pct}%` }}
              aria-hidden="true"
            />
          </div>
          <span className="lora-strength-value">{value.toFixed(2)}</span>
        </div>
    </div>
  );
}

// ── Browse row ──────────────────────────────────────────────────────────

function BrowseLoraRow({ entry }: { entry: LoraCatalogEntry }) {
  const { id } = entry;
  const enabled = useLoraStore((s) => s.enabled.has(id));
  // Subscribe to the cap state so the row re-renders when another LoRA
  // is enabled/disabled elsewhere and our affordance needs to flip
  // between "+" (clickable) and "max" (disabled).
  const atCap = useLoraStore(
    (s) =>
      s.maxEnabled !== null &&
      s.maxEnabled >= 0 &&
      s.enabled.size >= s.maxEnabled,
  );
  const cap = useLoraStore((s) => s.maxEnabled);
  const toggle = useLoraToggle();

  const md = entry.metadata;
  const displayName = displayNameOf(entry);
  const trigger = md?.primary_trigger_word ?? null;
  const desc = useMemo(() => shortDescription(md), [md]);

  const rowRef = useRef<HTMLButtonElement | null>(null);
  const [confirmMsg, flashConfirm] = useConfirmFlash();

  // Cap blocks new enables, never disables. So a row that's already
  // enabled stays clickable (so the user can free a slot); only the
  // disabled-and-cap-reached rows become inert.
  const capBlocked = atCap && !enabled;
  const rowTitle = capBlocked
    ? `Maximum ${cap} LoRAs active — disable one to enable this`
    : displayName;

  return (
    <button
      ref={rowRef}
      type="button"
      className={`lora-browse-row${enabled ? " enabled" : ""}${capBlocked ? " cap-blocked" : ""}`}
      data-midi-kind="cc"
      data-midi-target={`lora_str_${id}`}
      data-dd-tooltip-title={displayName}
      onClick={() => toggle(id, enabled)}
      onContextMenu={(e) => openLoraMidiMenu(e, id, trigger, flashConfirm)}
      aria-pressed={enabled}
      aria-disabled={capBlocked}
      disabled={capBlocked}
      title={rowTitle}
    >
        <span className="lora-browse-main">
          <span className="lora-browse-name" title={displayName}>
            {confirmMsg ?? displayName}
          </span>
          <span className="lora-browse-add" aria-hidden="true">
            {enabled ? "✓" : capBlocked ? "—" : "+"}
          </span>
        </span>
        {desc && <span className="lora-browse-desc">{desc}</span>}
    </button>
  );
}

// ── Genre accordion section ─────────────────────────────────────────────

interface GenreSectionProps {
  label: string;
  entries: LoraCatalogEntry[];
  enabledCount: number;
  open: boolean;
  onToggle: () => void;
}

function GenreSection({
  label,
  entries,
  enabledCount,
  open,
  onToggle,
}: GenreSectionProps) {
  return (
    <div className="lora-genre">
      <button
        type="button"
        className="lora-genre-head"
        onClick={onToggle}
        aria-expanded={open}
      >
        <span className="lora-genre-caret" aria-hidden="true">
          {open ? "▾" : "▸"}
        </span>
        <span className="lora-genre-name">{label}</span>
        {enabledCount > 0 && (
          <span className="lora-genre-on">{enabledCount} on</span>
        )}
        <span className="lora-genre-count">{entries.length}</span>
      </button>
      {open && (
        <div className="lora-genre-body">
          {entries.map((e) => (
            <BrowseLoraRow key={e.id} entry={e} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Tile ────────────────────────────────────────────────────────────────

export function LibraryTile() {
  const catalog = useLoraStore((s) => s.catalog);
  const setCatalog = useLoraStore((s) => s.setCatalog);
  const enabledSet = useLoraStore((s) => s.enabled);
  const sessionWsUrl = useSessionStore((s) => s.wsUrl);
  const sessionScale = useSessionStore((s) => s.checkpointScale);
  const cfg = useConfig();
  const cfgShowAll = cfg.engine.show_incompatible_loras ?? false;

  const [query, setQuery] = useState("");
  // Per-session override that flips the incompatible-LoRA filter when
  // the operator clicks "show all" in the footer. Doesn't touch the
  // persistent config — closing and reopening the tile resets to the
  // configured default.
  const [showAllOverride, setShowAllOverride] = useState(false);
  const showAll = cfgShowAll || showAllOverride;
  // Which genre sections are expanded. All collapsed by default — the
  // tile opens compact, the operator expands only what they want.
  const [openCats, setOpenCats] = useState<Set<string>>(() => new Set());
  // Admin-hidden LoRA ids (orchestrator state, fetched app-origin).
  const [adminHidden, setAdminHidden] = useState<Set<string>>(
    () => new Set(),
  );

  useEffect(() => {
    if (!sessionWsUrl && !LOCAL_MODE) return;
    void listLoras().then(setCatalog).catch(() => {});
    void listHiddenLoras().then(setAdminHidden).catch(() => {});
  }, [setCatalog, sessionWsUrl]);

  // Force-disable any admin-hidden LoRA that's currently enabled — an
  // operator must not keep applying a LoRA an admin has retired. Runs
  // when the hidden set or the catalog changes (the latter covers the
  // setCatalog auto-seed possibly enabling a hidden id). disable()
  // mutates useLoraStore.enabled, which useLoraTriggerSync observes to
  // re-send the prompt without the now-dropped trigger.
  useEffect(() => {
    if (adminHidden.size === 0) return;
    const { enabled, disable } = useLoraStore.getState();
    const remote = useSessionStore.getState().remote;
    for (const id of enabled) {
      if (adminHidden.has(id)) {
        disable(id);
        remote?.sendDisableLora(id);
      }
    }
  }, [adminHidden, catalog]);

  // Admin-hidden LoRAs drop out entirely — no row, no count. The
  // scale-compat filter then runs on what's left so the "N hidden"
  // footer count tracks scale incompatibility specifically.
  const visible = useMemo(
    () => catalog.filter((entry) => !adminHidden.has(entry.id)),
    [catalog, adminHidden],
  );
  const compatible = useMemo(
    () =>
      showAll
        ? visible
        : visible.filter((entry) =>
            isLoraCompatibleWithScale(entry, sessionScale),
          ),
    [visible, sessionScale, showAll],
  );
  const hiddenCount = visible.length - compatible.length;

  const filtered = useMemo(
    () => compatible.filter((entry) => matchesQuery(entry, query)),
    [compatible, query],
  );
  const searching = query.trim().length > 0;

  const activeEntries = useMemo(
    () => filtered.filter((entry) => enabledSet.has(entry.id)),
    [filtered, enabledSet],
  );

  // Browse list grouped into genre categories, in CATEGORY_ORDER, each
  // category alphabetised by display name.
  const groups = useMemo(() => {
    const byCat = new Map<string, LoraCatalogEntry[]>();
    for (const entry of filtered) {
      const cat = categoryOf(entry);
      const arr = byCat.get(cat);
      if (arr) arr.push(entry);
      else byCat.set(cat, [entry]);
    }
    const ordered: { label: string; entries: LoraCatalogEntry[] }[] = [];
    for (const cat of CATEGORY_ORDER) {
      const arr = byCat.get(cat);
      if (arr) {
        arr.sort(byDisplayName);
        ordered.push({ label: cat, entries: arr });
        byCat.delete(cat);
      }
    }
    // Any category not in CATEGORY_ORDER (defensive — shouldn't happen).
    for (const cat of [...byCat.keys()].sort()) {
      const arr = byCat.get(cat)!;
      arr.sort(byDisplayName);
      ordered.push({ label: cat, entries: arr });
    }
    return ordered;
  }, [filtered]);

  const toggleCat = useCallback((label: string) => {
    setOpenCats((prev) => {
      const next = new Set(prev);
      if (next.has(label)) next.delete(label);
      else next.add(label);
      return next;
    });
  }, []);

  if (catalog.length === 0) {
    return (
      <div className="mixer-tile" data-tile="library">
        <div className="mixer-tile-label">LoRA Library</div>
        <div className="lora-empty">no LoRAs found</div>
      </div>
    );
  }

  return (
    <div className="mixer-tile" data-tile="library">
      <div className="mixer-tile-label">LoRA Library</div>
      <div className="lora-search">
        <input
          type="text"
          className="lora-search-input"
          placeholder="search LoRAs"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search LoRA library"
        />
      </div>

      {activeEntries.length > 0 && (
        <div className="lora-section lora-active">
          <div className="lora-section-head">Active · {activeEntries.length}</div>
          <div className="lora-active-list">
            {activeEntries.map((entry) => (
              <ActiveLoraRow key={entry.id} entry={entry} />
            ))}
          </div>
        </div>
      )}

      <div className="lora-section">
        <div className="lora-section-head">
          {searching
            ? `Results · ${filtered.length}`
            : `All LoRAs · ${compatible.length}`}
        </div>
        <div className="lora-browse">
          {filtered.length === 0 ? (
            <div className="lora-empty">no matches</div>
          ) : searching ? (
            filtered.map((entry) => (
              <BrowseLoraRow key={entry.id} entry={entry} />
            ))
          ) : (
            groups.map((g) => (
              <GenreSection
                key={g.label}
                label={g.label}
                entries={g.entries}
                enabledCount={
                  g.entries.filter((e) => enabledSet.has(e.id)).length
                }
                open={openCats.has(g.label)}
                onToggle={() => toggleCat(g.label)}
              />
            ))
          )}
        </div>
      </div>

      {hiddenCount > 0 && (
        <div className="lora-hidden-footer">
          <span>
            {hiddenCount} hidden
            {sessionScale ? ` (not ${sessionScale})` : ""}
          </span>
          <button
            type="button"
            className="lora-hidden-toggle"
            onClick={() => setShowAllOverride((v) => !v)}
          >
            {showAllOverride ? "hide" : "show all"}
          </button>
        </div>
      )}
    </div>
  );
}
