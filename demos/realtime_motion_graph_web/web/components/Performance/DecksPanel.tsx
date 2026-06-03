"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";

import {
  decodeAudioFile,
  listFixtures,
  pickDefaultFixture,
  type DecodedFixture,
  type StemSourceMode,
} from "@/engine/audio/loadFixture";
import { useDeckAssets } from "@/hooks/useDeckAssets";
import { useSeedUserUploads } from "@/hooks/useSeedUserUploads";
import { commitUploadedTrack } from "@/lib/audio/commitUploadedTrack";
import { deckAssetSource } from "@/lib/audio/deckAssets";
import { trimAudioBuffer } from "@/lib/audio/trimAudioBuffer";
import { useConfig } from "@/lib/config";
import { LOCAL_MODE } from "@/lib/runtime";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import {
  DECK_IDS,
  DECK_STEM_OVERLAY_MAX,
  MAX_DECKS,
  useDeckStore,
  type DeckId,
} from "@/store/useDeckStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import type { TimeSignature } from "@/types/engine";

import { AlmostReadyDialog } from "./AlmostReadyDialog";
import { RefSelect, type RefSelectGroup } from "./RefSelect";
import { WaveformTrimDialog } from "./WaveformTrimDialog";

const DEFAULT_TRIM_CAP_S = 120;
const SOURCE_PARTS: StemSourceMode[] = ["full", "vocals", "instruments"];
const STEM_OVERLAY_PARTS = [
  { kind: "vocals", label: "Vocal" },
  { kind: "instruments", label: "Instr" },
] as const;
type UploadTarget = DeckId | "new";

export function DecksPanel() {
  const decks = useDeckStore((s) => s.decks);
  const deckIds = useDeckStore((s) => s.deckIds);
  const timbreDeckId = useDeckStore((s) => s.timbreDeckId);
  const structureDeckId = useDeckStore((s) => s.structureDeckId);
  const crossfade = useDeckStore((s) => s.crossfade);
  const monitorEnabled = useDeckStore((s) => s.monitorEnabled);
  const inferenceEnabled = useDeckStore((s) => s.inferenceEnabled);
  const sessionWsUrl = useSessionStore((s) => s.wsUrl);
  const activeFixture = usePerformanceStore((s) => s.fixture);
  const timbreName = usePerformanceStore((s) => s.timbreName);
  const structName = usePerformanceStore((s) => s.structName);
  const customNames = useCustomTracksStore((s) => s.names);
  const addCustomTrack = useCustomTracksStore((s) => s.add);
  const [fixtures, setFixtures] = useState<string[]>([]);
  const { assetsByDeck, statuses, errors } = useDeckAssets(decks);
  const trimCapS =
    useConfig().engine.max_source_duration_s ?? DEFAULT_TRIM_CAP_S;
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const uploadTargetRef = useRef<UploadTarget>("A");
  const [uploadingTarget, setUploadingTarget] = useState<UploadTarget | null>(null);
  const [addDeckOpen, setAddDeckOpen] = useState(false);
  const [trimming, setTrimming] = useState<{
    target: UploadTarget;
    decoded: DecodedFixture;
    fileName: string;
    originalFile: File;
  } | null>(null);
  const [pending, setPending] = useState<{
    target: UploadTarget;
    decoded: DecodedFixture;
    fileName: string;
    originalFile: File;
  } | null>(null);

  useSeedUserUploads();

  useEffect(() => {
    if (!sessionWsUrl && !LOCAL_MODE) return;
    void listFixtures()
      .then((names) => {
        setFixtures(names);
        const defaultTrack = activeFixture || pickDefaultFixture(names);
        if (defaultTrack) {
          if (!activeFixture) usePerformanceStore.getState().setFixture(defaultTrack);
          useDeckStore.getState().ensureInitialDeck(defaultTrack);
        }
      })
      .catch(() => setFixtures([]));
  }, [activeFixture, sessionWsUrl]);

  useEffect(() => {
    const defaultTrack = activeFixture || pickDefaultFixture(fixtures) || customNames[0];
    if (defaultTrack) {
      useDeckStore.getState().ensureInitialDeck(defaultTrack);
    }
  }, [activeFixture, customNames, fixtures]);

  async function onFilePicked(target: UploadTarget, file: File) {
    const { setStatus } = useSessionStore.getState();
    setUploadingTarget(target);
    setStatus(useSessionStore.getState().status, "");
    try {
      const decoded = await decodeAudioFile(file);
      const baseName = file.name;
      let chosen = baseName;
      let i = 1;
      while (useCustomTracksStore.getState().has(chosen)) {
        chosen = `${baseName} (${i++})`;
      }
      setTrimming({ target, decoded, fileName: chosen, originalFile: file });
      setStatus(useSessionStore.getState().status, "");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus(
        useSessionStore.getState().status,
        target === "new" ? `New deck: ${msg}` : `Deck ${target}: ${msg}`,
      );
    } finally {
      setUploadingTarget(null);
    }
  }

  function onTrimConfirm(startS: number, endS: number) {
    if (!trimming) return;
    const trimmed = trimAudioBuffer(trimming.decoded, startS, endS);
    setPending({
      target: trimming.target,
      decoded: trimmed,
      fileName: trimming.fileName,
      originalFile: trimming.originalFile,
    });
    setTrimming(null);
  }

  async function commitPending(
    keyOverride: string | null,
    timeSignatureOverride: TimeSignature | null,
    sourceMode: StemSourceMode,
  ) {
    if (!pending) return;
    await commitUploadedTrack({
      pending,
      keyOverride,
      timeSignatureOverride,
      sourceMode,
      addCustomTrack,
      setFixture: (name) => {
        if (pending.target === "new") {
          const added = useDeckStore.getState().addDeck(name);
          if (added) setAddDeckOpen(false);
          return;
        }
        setDeckTrack(pending.target, name);
      },
      setPending: (next) =>
        setPending(next ? { ...next, target: pending.target } : null),
      setUploading: (next) => setUploadingTarget(next ? pending.target : null),
    });
  }

  function shortTrackName(name: string | null): string {
    if (!name) return "Loading track";
    return name.replace(/\.(wav|mp3|flac|ogg|m4a|aac)$/i, "");
  }

  function deckDurationLabel(id: DeckId): string {
    const assets = assetsByDeck[id];
    const name = decks[id].trackName;
    if (name && statuses[name] === "loading") return "loading assets";
    if (name && statuses[name] === "failed") return "asset failed";
    if (!assets) return "waiting for assets";
    const seconds = assets.full.frames / assets.full.sampleRate;
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds - mins * 60).toString().padStart(2, "0");
    const stemCount = Number(Boolean(assets.stems.vocals)) + Number(Boolean(assets.stems.instruments));
    return `${mins}:${secs} · ${stemCount}/2 stems`;
  }

  function roleSummary(timbre: boolean, structure: boolean): string {
    const roles = [
      timbre ? "timbre" : null,
      structure ? "structure" : null,
    ].filter(Boolean);
    return roles.length ? ` · ${roles.join(" · ")}` : "";
  }

  function setDeckTrack(id: DeckId, name: string): void {
    useDeckStore.getState().setTrack(id, name);
  }

  function addDeckFromTrack(name: string): void {
    const added = useDeckStore.getState().addDeck(name);
    if (added) setAddDeckOpen(false);
  }

  function clearReference(kind: "timbre" | "structure"): void {
    const session = useSessionStore.getState();
    if (kind === "timbre") {
      session.remote?.sendClearTimbreSource();
      usePerformanceStore.getState().setTimbreRef(null);
      useDeckStore.getState().setTimbreDeck(null);
    } else {
      session.remote?.sendClearStructureSource();
      usePerformanceStore.getState().setStructRef(null);
      useDeckStore.getState().setStructureDeck(null);
    }
    session.setStatus(session.status, `${kind} follows deck mix`);
  }

  function applyReferenceDeck(kind: "timbre" | "structure", id: DeckId): void {
    const currentRole = kind === "timbre" ? timbreDeckId : structureDeckId;
    if (currentRole === id) {
      clearReference(kind);
      return;
    }

    const deck = useDeckStore.getState().decks[id];
    const name = deck.trackName;
    if (!name) return;
    const session = useSessionStore.getState();
    if (session.status !== "ready" || !session.remote) {
      session.setStatus(session.status, `Start playback before assigning ${kind}`);
      return;
    }

    if (fixtures.includes(name)) {
      if (kind === "timbre") {
        session.remote.sendSetTimbreFixture(name);
        usePerformanceStore.getState().setTimbreRef({ mode: "fixture", name });
        useDeckStore.getState().setTimbreDeck(id);
      } else {
        session.remote.sendSetStructureFixture(name);
        usePerformanceStore.getState().setStructRef({ mode: "fixture", name });
        useDeckStore.getState().setStructureDeck(id);
      }
      session.setStatus("ready", `Loading ${kind} ${name}…`);
      return;
    }

    const assets = assetsByDeck[id];
    if (!assets) {
      session.setStatus("ready", `Deck ${id} is still loading ${name}`);
      return;
    }
    const ok =
      kind === "timbre"
        ? session.remote.sendSetTimbreSource(
            assets.full.interleaved,
            assets.full.channels,
            name,
          )
        : session.remote.sendSetStructureSource(
            assets.full.interleaved,
            assets.full.channels,
            name,
          );
    if (ok) {
      if (kind === "timbre") {
        usePerformanceStore.getState().setTimbreRef({ mode: "clip", name });
        useDeckStore.getState().setTimbreDeck(id);
      } else {
        usePerformanceStore.getState().setStructRef({ mode: "clip", name });
        useDeckStore.getState().setStructureDeck(id);
      }
      session.setStatus("ready", `Loading ${kind} ${name}…`);
    } else {
      session.setStatus("ready", `${kind} reference failed`);
    }
  }

  const trackGroups: RefSelectGroup[] = [
    {
      label: "Library",
      options: fixtures.map((name) => ({ value: name, label: name })),
    },
    {
      label: "Your tracks",
      options: customNames.map((name) => ({ value: name, label: name })),
    },
  ];
  const busADeckId = DECK_IDS.find(
    (id) => deckIds.includes(id) && decks[id].crossfadeSide === "left",
  );
  const busBDeckId = DECK_IDS.find(
    (id) => deckIds.includes(id) && decks[id].crossfadeSide === "right",
  );
  const busAColor = busADeckId ? decks[busADeckId].color : "oklch(0.72 0.16 42)";
  const busBColor = busBDeckId ? decks[busBDeckId].color : "oklch(0.70 0.16 305)";
  const busAPct = Math.round((1 - crossfade) * 100);
  const busBPct = Math.round(crossfade * 100);

  return (
    <section className="decks-panel" aria-label="Deck mixer">
      <div className="decks-panel-head">
        <div>
          <span className="decks-panel-kicker">Input mixer</span>
          <h2 className="decks-panel-title">Decks</h2>
        </div>
        <div className="decks-panel-toggles">
          <button
            type="button"
            className={monitorEnabled ? "is-active" : ""}
            data-dd-tooltip="Hear the deck mix directly in the browser. This is instant and separate from the generated model output."
            onClick={() =>
              useDeckStore.getState().setMonitorEnabled(!monitorEnabled)
            }
          >
            Monitor
          </button>
          <button
            type="button"
            className={inferenceEnabled ? "is-active" : ""}
            data-dd-tooltip="Feed the deck mix to inference as live parameters. Loaded, unmuted decks contribute by volume and crossfader position, no source-swap or playback required."
            onClick={() =>
              useDeckStore.getState().setInferenceEnabled(!inferenceEnabled)
            }
          >
            Infer
          </button>
          <button
            type="button"
            className={addDeckOpen ? "is-active" : ""}
            disabled={deckIds.length >= MAX_DECKS}
            data-dd-tooltip="Add a deck by choosing an existing track or uploading new audio. No deck is created until a track is selected."
            onClick={() => setAddDeckOpen((v) => !v)}
          >
            Add deck
          </button>
        </div>
      </div>

      <div
        className="deck-crossfader-card"
        data-dd-tooltip="Inference input is the rendered mixer output. Decks assigned to bus A and bus B are blended here, then the mixed PCM is sent to inference."
        data-dd-tooltip-wide=""
        data-dd-tooltip-title="Deck mix input"
        style={
          {
            "--deck-crossfade": `${crossfade * 100}%`,
            "--deck-bus-a-color": busAColor,
            "--deck-bus-b-color": busBColor,
          } as CSSProperties
        }
      >
        <div className="deck-bus-readout deck-bus-readout--a">
          <span className="deck-bus-label">Bus A</span>
          <strong>{busAPct}%</strong>
          <span>{busADeckId ?? "empty"}</span>
        </div>
        <div className="deck-crossfade-control">
          <div className="deck-crossfade-track" aria-hidden="true">
            <span className="deck-crossfade-fill deck-crossfade-fill--a" />
            <span className="deck-crossfade-fill deck-crossfade-fill--b" />
            <span className="deck-crossfade-cap" />
          </div>
          <input
            className="deck-crossfade-input"
            type="range"
            min={0}
            max={1}
            step={0.001}
            value={crossfade}
            onChange={(e) =>
              useDeckStore.getState().setCrossfade(Number(e.target.value))
            }
            aria-label="Deck bus crossfader"
          />
          <span className="deck-crossfade-caption">mixed inference input</span>
        </div>
        <div className="deck-bus-readout deck-bus-readout--b">
          <span className="deck-bus-label">Bus B</span>
          <strong>{busBPct}%</strong>
          <span>{busBDeckId ?? "empty"}</span>
        </div>
      </div>

      {addDeckOpen && deckIds.length < MAX_DECKS && (
        <div className="deck-add-card">
          <div className="deck-add-copy">
            <span className="decks-panel-kicker">Add deck</span>
            <strong>Choose a track first</strong>
            <span>No empty deck is created until you pick or upload audio.</span>
          </div>
          <RefSelect
            label="track"
            value=""
            pinned={[{ value: "", label: "Select existing track" }]}
            groups={trackGroups}
            onSelect={(value) => {
              if (value) addDeckFromTrack(value);
            }}
            disabled={uploadingTarget === "new"}
            ariaLabel="Track for new deck"
            onUpload={() => {
              uploadTargetRef.current = "new";
              fileInputRef.current?.click();
            }}
            uploadLabel={
              uploadingTarget === "new" ? "Decoding…" : "Upload audio to new deck"
            }
            tooltip="Create a new deck from an existing library/user track, or upload a new track through the same trim and stem-rip flow."
          />
          <button
            type="button"
            className="deck-add-cancel"
            onClick={() => setAddDeckOpen(false)}
          >
            Cancel
          </button>
        </div>
      )}

      <div className="deck-list">
        {deckIds.map((id) => {
          const deck = decks[id];
          const trackStatus = deck.trackName ? statuses[deck.trackName] : "idle";
          const trackError = deck.trackName ? errors[deck.trackName] : undefined;
          const assets = assetsByDeck[id];
          const activeSource = deckAssetSource(assets, deck.sourcePart);
          const missingSelectedStem =
            deck.sourcePart !== "full" && assets && !activeSource;
          const uploadingThisDeck = uploadingTarget === id;
          const deckBusy = uploadingThisDeck || trackStatus === "loading";
          const isTimbre = timbreDeckId === id || timbreName === deck.trackName;
          const isStructure =
            structureDeckId === id || structName === deck.trackName;
          return (
            <article
              key={id}
              className={`deck-card${deckBusy ? " is-loading" : ""}`}
              style={{ "--deck-color": deck.color } as CSSProperties}
              aria-busy={deckBusy}
            >
              <div className="deck-card-head">
                <span className="deck-badge">{id}</span>
                <div className="deck-title-block">
                  <span className="deck-name">Deck {id}</span>
                  <strong className="deck-track-title" title={deck.trackName ?? ""}>
                    {shortTrackName(deck.trackName)}
                  </strong>
                  <span className="deck-track-meta">{deckDurationLabel(id)}</span>
                </div>
                <button
                  type="button"
                  className="deck-remove-button"
                  disabled={deckIds.length <= 1}
                  data-dd-tooltip="Remove this deck. At least one loaded deck always remains."
                  onClick={() => useDeckStore.getState().removeDeck(id)}
                >
                  ×
                </button>
              </div>

              <div className="deck-track-picker">
                <RefSelect
                  label="replace"
                  value={deck.trackName ?? ""}
                  pinned={[]}
                  groups={trackGroups}
                  onSelect={(value) => value && setDeckTrack(id, value)}
                  disabled={uploadingThisDeck}
                  ariaLabel={`Deck ${id} track`}
                  onUpload={() => {
                    uploadTargetRef.current = id;
                    fileInputRef.current?.click();
                  }}
                  uploadLabel={
                    uploadingThisDeck ? "Decoding…" : `Upload audio to deck ${id}`
                  }
                  tooltip="Replace this deck's track. The next deck-mix snapshot uses the replacement as this deck's contribution."
                />
              </div>

              {deckBusy && (
                <div className="deck-loading-strip" aria-hidden="true">
                  <span />
                </div>
              )}

              <div className="deck-role-row" aria-label={`Deck ${id} reference roles`}>
                <button
                  type="button"
                  className={isTimbre ? "is-active" : ""}
                  disabled={deckBusy}
                  data-dd-tooltip="Use this deck's full track as the timbre reference. Click again to return timbre to the deck mix."
                  onClick={() => applyReferenceDeck("timbre", id)}
                >
                  Timbre
                </button>
                <button
                  type="button"
                  className={isStructure ? "is-active" : ""}
                  disabled={deckBusy}
                  data-dd-tooltip="Use this deck's full track as the structure reference. Click again to return structure to the deck mix."
                  onClick={() => applyReferenceDeck("structure", id)}
                >
                  Structure
                </button>
              </div>

              <div
                className="deck-source-parts"
                role="radiogroup"
                data-dd-tooltip="Choose which version of this deck feeds the deck mix: full track, vocal stem, or instrumental stem. Stem options enable when cached RoFormer assets exist."
                data-dd-tooltip-wide=""
                data-dd-tooltip-title={`Deck ${id} source`}
              >
                {SOURCE_PARTS.map((part) => {
                  const stemPart =
                    part === "vocals" || part === "instruments" ? part : null;
                  const disabled =
                    stemPart !== null &&
                    (assets === undefined || !assets.stems[stemPart]);
                  return (
                    <button
                      key={part}
                      type="button"
                      role="radio"
                      aria-checked={deck.sourcePart === part}
                      className={deck.sourcePart === part ? "is-active" : ""}
                      disabled={disabled}
                      onClick={() =>
                        useDeckStore.getState().setSourcePart(id, part)
                      }
                    >
                      {part === "instruments" ? "instr" : part}
                    </button>
                  );
                })}
              </div>

              <div className="deck-performance-row">
                <button
                  type="button"
                  disabled={!deck.trackName}
                  data-dd-tooltip={
                    deck.playing
                      ? "Pause this deck's local monitor transport. The deck still contributes to inference while loaded and unmuted."
                      : "Start local monitoring from this deck's current playhead."
                  }
                  onClick={() =>
                    useDeckStore.getState().setPlaying(id, !deck.playing)
                  }
                >
                  {deck.playing ? "Pause" : "Play"}
                </button>
                <button
                  type="button"
                  disabled={!deck.trackName}
                  data-dd-tooltip="Jump this deck's monitor playhead back to its cue point. The default cue is the start of the song."
                  onClick={() => useDeckStore.getState().jumpToCue(id)}
                >
                  Cue
                </button>
                <button
                  type="button"
                  className={deck.muted ? "is-active" : ""}
                  data-dd-tooltip="Mute this deck in both the monitor and the mixed inference source."
                  onClick={() => useDeckStore.getState().toggleMuted(id)}
                >
                  Mute
                </button>
                <div
                  className="deck-side-switch"
                  role="radiogroup"
                  aria-label={`Deck ${id} crossfader side`}
                >
                  {(["left", "right"] as const).map((side) => (
                    <button
                      key={side}
                      type="button"
                      role="radio"
                      aria-checked={deck.crossfadeSide === side}
                      className={deck.crossfadeSide === side ? "is-active" : ""}
                      data-dd-tooltip={`Assign deck ${id} to mix bus ${side === "left" ? "A" : "B"}. A bus holds one deck at a time; assigning this deck parks the previous one on that bus.`}
                      onClick={() =>
                        useDeckStore.getState().setCrossfadeSide(id, side)
                      }
                    >
                      {side === "left" ? "A" : "B"}
                    </button>
                  ))}
                </div>
              </div>

              <div
                className="deck-fader-row"
                data-dd-tooltip="Deck level before the scene crossfader."
              >
                <span>Level</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.001}
                  value={deck.volume}
                  onChange={(e) =>
                    useDeckStore
                      .getState()
                      .setVolume(id, Number(e.target.value))
                  }
                  aria-label={`Deck ${id} volume`}
                />
              </div>

              {(assets?.stems.vocals || assets?.stems.instruments) && (
                <div
                  className="deck-stem-overlay-controls"
                  data-dd-tooltip="Layer this deck's cached vocal or instrumental stem on top of the generated output. These faders do not change the deck's inference source."
                  data-dd-tooltip-wide=""
                  data-dd-tooltip-title={`Deck ${id} output layers`}
                >
                  {STEM_OVERLAY_PARTS.map(({ kind, label }) => {
                    const source = assets?.stems[kind];
                    const value = deck.stemOverlayLevels?.[kind] ?? 0;
                    return (
                      <div
                        key={kind}
                        className={`deck-stem-fader-row${source ? "" : " is-disabled"}`}
                      >
                        <span>{label}</span>
                        <input
                          type="range"
                          min={0}
                          max={DECK_STEM_OVERLAY_MAX}
                          step={0.001}
                          value={value}
                          disabled={!source}
                          onChange={(e) =>
                            useDeckStore
                              .getState()
                              .setStemOverlayLevel(id, kind, Number(e.target.value))
                          }
                          aria-label={`Deck ${id} ${label.toLowerCase()} output layer`}
                        />
                        <output>{value.toFixed(1)}</output>
                      </div>
                    );
                  })}
                </div>
              )}

              <div className="deck-status">
                {uploadingThisDeck
                  ? `Deck ${id}: decoding upload…`
                  : trackStatus === "loading"
                  ? `Deck ${id}: loading ${shortTrackName(deck.trackName)}…`
                  : trackStatus === "failed"
                    ? `Failed: ${trackError ?? "track unavailable"}`
                    : missingSelectedStem
                      ? "Stem unavailable"
                      : deck.trackName
                        ? `${deck.crossfadeSide === null ? "parked · " : ""}${deck.sourcePart} ready${roleSummary(isTimbre, isStructure)}`
                        : "Loading library…"}
              </div>
            </article>
          );
        })}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a,.aac"
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (file) void onFilePicked(uploadTargetRef.current, file);
        }}
      />

      {trimming && (
        <WaveformTrimDialog
          decoded={trimming.decoded}
          fileName={trimming.fileName}
          capS={trimCapS}
          onConfirm={onTrimConfirm}
          onCancel={() => setTrimming(null)}
        />
      )}
      {pending && (
        <AlmostReadyDialog
          fileName={pending.fileName}
          wasTrimmed={false}
          defaultKey={usePerformanceStore.getState().activeKey}
          defaultTimeSignature={
            usePerformanceStore.getState().activeTimeSignature
          }
          onContinue={({ keyOverride, timeSignatureOverride, sourceMode }) =>
            commitPending(keyOverride, timeSignatureOverride, sourceMode)
          }
          onPickAnother={() => {
            setPending(null);
            setTimeout(() => fileInputRef.current?.click(), 0);
          }}
          onClose={() => setPending(null)}
        />
      )}
    </section>
  );
}
