// Video layer: crossfade player with beat-grid sync and marker detection.
// Direct port of static/video.js. Phase 5 perf: pause inactive <video> once
// the crossfade window finishes, halving video decoder cost between swaps.

import { podHttp } from "@/engine/podUrl";

interface Options {
  videoA: HTMLVideoElement;
  videoB: HTMLVideoElement;
  bpm?: number;
  crossfadeDuration?: number;
  useMarkers?: boolean;
}

type GetAudioPos = () => number;

export class VideoLayer {
  videoA: HTMLVideoElement;
  videoB: HTMLVideoElement;
  activeVideo: HTMLVideoElement;
  inactiveVideo: HTMLVideoElement;
  hasVideo = false;
  videos: string[] = [];
  currentIndex = 0;

  bpm: number;
  beatDuration: number;
  crossfadeDuration: number;

  private useMarkers: boolean;
  private _markerCanvas: HTMLCanvasElement;
  private _markerCtx: CanvasRenderingContext2D | null;
  private _lastMarkerDetected = false;
  private _beatOffset: number | null = null;
  private _markerCount = 0;

  private _getAudioPos: GetAudioPos = () => 0;
  private _hasAudio = false;

  private _markerRaf = 0;
  private _driftInterval: number;

  constructor(opts: Options) {
    this.videoA = opts.videoA;
    this.videoB = opts.videoB;
    this.activeVideo = opts.videoA;
    this.inactiveVideo = opts.videoB;
    this.bpm = opts.bpm ?? 134;
    this.beatDuration = 60 / this.bpm;
    this.crossfadeDuration = opts.crossfadeDuration ?? 1.5;
    this.useMarkers = opts.useMarkers ?? false;

    this._markerCanvas = document.createElement("canvas");
    this._markerCanvas.width = 4;
    this._markerCanvas.height = 4;
    this._markerCtx = this._markerCanvas.getContext("2d", {
      willReadFrequently: true,
    });

    if (this.useMarkers) {
      this._markerRaf = requestAnimationFrame(() => this._markerLoop());
    }

    this._driftInterval = window.setInterval(() => this._correctDrift(), 1000);
  }

  setBpm(bpm: number): void {
    this.bpm = bpm;
    this.beatDuration = 60 / bpm;
  }

  setAudioSource(getPos: GetAudioPos, hasAudio: boolean): void {
    this._getAudioPos = getPos;
    this._hasAudio = hasAudio;
  }

  setVideos(videos: string[]): void {
    this.videos = videos;
    this.currentIndex = 0;
  }

  /** Crossfade to a new video. Pauses the outgoing one once the fade completes. */
  play(filename: string, transition: "crossfade" | "cut" = "crossfade"): void {
    this._resetMarkerState();
    const url = podHttp(`/videos/${encodeURIComponent(filename)}`);

    if (transition === "crossfade" && this.hasVideo) {
      const inactive = this.inactiveVideo;
      const active = this.activeVideo;
      inactive.src = url;
      inactive.loop = true;
      inactive.muted = true;
      const onReady = () => {
        inactive.removeEventListener("canplay", onReady);
        this._syncToBeat(inactive);
        inactive.play().catch(() => {});
        inactive.style.opacity = "1";
        active.style.opacity = "0";
        setTimeout(
          () => {
            // Perf: pause + tear down outgoing instead of leaving it
            // decoding silently in the background. Halves video decoder
            // cost between swaps.
            active.pause();
            active.removeAttribute("src");
            active.load();
            const tmp = this.activeVideo;
            this.activeVideo = this.inactiveVideo;
            this.inactiveVideo = tmp;
          },
          this.crossfadeDuration * 1000 + 100,
        );
      };
      inactive.addEventListener("canplay", onReady);
      inactive.load();
    } else {
      const el = this.activeVideo;
      el.src = url;
      el.loop = true;
      el.muted = true;
      el.style.opacity = "1";
      const onReady = () => {
        el.removeEventListener("canplay", onReady);
        this.hasVideo = true;
        this._syncToBeat(el);
        el.play().catch(() => {});
      };
      el.addEventListener("canplay", onReady);
      el.load();
    }
  }

  next(): void {
    if (this.videos.length === 0) return;
    this.currentIndex = (this.currentIndex + 1) % this.videos.length;
    this.play(this.videos[this.currentIndex]);
  }

  previous(): void {
    if (this.videos.length === 0) return;
    this.currentIndex =
      (this.currentIndex - 1 + this.videos.length) % this.videos.length;
    this.play(this.videos[this.currentIndex]);
  }

  destroy(): void {
    if (this._markerRaf) cancelAnimationFrame(this._markerRaf);
    clearInterval(this._driftInterval);
    try {
      this.activeVideo.pause();
    } catch {}
    try {
      this.inactiveVideo.pause();
    } catch {}
  }

  // ── internals ────────────────────────────────────────────────────────

  private _resetMarkerState(): void {
    this._lastMarkerDetected = false;
    this._beatOffset = null;
    this._markerCount = 0;
  }

  private _markerLoop(): void {
    if (this.hasVideo && this.activeVideo && !this.activeVideo.paused) {
      this._sampleMarker(this.activeVideo);
    }
    this._markerRaf = requestAnimationFrame(() => this._markerLoop());
  }

  private _sampleMarker(el: HTMLVideoElement): void {
    if (!el || el.paused || !el.videoWidth || !this._markerCtx) return;
    this._markerCtx.drawImage(el, 0, 0, 8, 8, 0, 0, 4, 4);
    const px = this._markerCtx.getImageData(1, 1, 1, 1).data;
    const detected = px[1] > 50 && px[1] > px[0] * 3 && px[1] > px[2] * 3;
    if (detected && !this._lastMarkerDetected) this._onVideoDownbeat(el);
    this._lastMarkerDetected = detected;
  }

  private _onVideoDownbeat(el: HTMLVideoElement): void {
    if (!this._hasAudio) return;
    const audioPos = this._getAudioPos();
    const phase = (audioPos % this.beatDuration) / this.beatDuration;
    const near = phase < 0.2 || phase > 0.8;
    this._markerCount++;
    if (near) {
      this._beatOffset =
        el.currentTime -
        ((Math.round(audioPos / this.beatDuration) * this.beatDuration) %
          (el.duration || 1));
    } else {
      const nearest =
        Math.round(audioPos / this.beatDuration) * this.beatDuration;
      const vd = el.duration;
      const target =
        this._beatOffset !== null
          ? (nearest % vd) + this._beatOffset
          : nearest % vd;
      el.currentTime = ((target % vd) + vd) % vd;
    }
  }

  private _syncToBeat(el: HTMLVideoElement): void {
    if (!el.duration || el.duration === Infinity) return;
    if (this._hasAudio) {
      const ap = this._getAudioPos();
      el.currentTime =
        (Math.floor(ap / this.beatDuration) * this.beatDuration) % el.duration;
    } else {
      el.currentTime = 0;
    }
  }

  private _correctDrift(): void {
    if (!this._hasAudio || !this.hasVideo) return;
    if (!this.activeVideo.duration || this.activeVideo.paused) return;
    if (this._markerCount > 0) return;
    const ap = this._getAudioPos();
    const vd = this.activeVideo.duration;
    let drift = (ap % vd) - this.activeVideo.currentTime;
    if (drift > vd / 2) drift -= vd;
    if (drift < -vd / 2) drift += vd;
    if (Math.abs(drift) > this.beatDuration * 0.4) {
      this.activeVideo.currentTime =
        (Math.floor(ap / this.beatDuration) * this.beatDuration) % vd;
    }
  }
}
