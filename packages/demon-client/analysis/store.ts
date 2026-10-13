// Client-side accumulator for one analysis tap: applies `analysis_update`
// range replaces in the order the contract defines and serves the current
// item set to renderers and exporters. Pure; no DOM beyond EventTarget.

import type {
  AnalysisBasis,
  AnalysisItem,
  AnalysisUpdateEventPayload,
  LoopBand,
} from "./types";

/** A stored item: the wire item plus the basis and lateness of the update
 *  that delivered it. */
export type StoredItem<I extends AnalysisItem = AnalysisItem> = I & {
  basis: AnalysisBasis;
  late: boolean;
};

/** An item returned by `itemsIn`, with the shift that maps its canvas time
 *  into the queried (unwrapped) window: draw it at `start_s + shift_s`. */
export interface WindowedItem<I extends AnalysisItem = AnalysisItem> {
  item: StoredItem<I>;
  shift_s: number;
}

export type AnalysisStoreListener = () => void;

/** Range keys are canvas seconds; quantize so 12.0 and 12.000000001 agree. */
function rangeKey(startS: number): string {
  return startS.toFixed(4);
}

function compareItems(a: AnalysisItem, b: AnalysisItem): number {
  return (
    a.start_s - b.start_s ||
    ((a as { pitch?: number }).pitch ?? 0) -
      ((b as { pitch?: number }).pitch ?? 0)
  );
}

/**
 * Accumulates one tap's items from `analysis_update` events.
 *
 * Rules (wire contract): `clear` drops everything first and adopts the
 * update's `source_epoch`; an update older than the last clear's epoch is
 * ignored; per range start, an update whose `seq` is not above the last
 * applied one is ignored; otherwise every item whose `start_s` lies in
 * [range_start_s, range_start_s + span_s) is replaced by `items`.
 */
export class AnalysisStore<I extends AnalysisItem = AnalysisItem> {
  readonly tap: string;
  private _items: StoredItem<I>[] = [];
  private _epoch: number | null = null;
  private _rangeSeq = new Map<string, number>();
  private _listeners = new Set<AnalysisStoreListener>();

  constructor(tap: string) {
    this.tap = tap;
  }

  /** Epoch of the last applied clear, or null before the first one. */
  get epoch(): number | null {
    return this._epoch;
  }

  get size(): number {
    return this._items.length;
  }

  /**
   * Apply one update. Returns true when the item set changed shape (the
   * update was accepted), false when it was ignored (other tap, stale
   * epoch, stale seq).
   */
  apply(update: AnalysisUpdateEventPayload): boolean {
    if (update.tap !== this.tap) return false;
    const epoch = update.source_epoch;
    if (this._epoch !== null && epoch < this._epoch) return false;

    if (update.clear) {
      this._items = [];
      this._rangeSeq.clear();
      this._epoch = epoch;
    } else {
      const key = rangeKey(update.range_start_s);
      const last = this._rangeSeq.get(key);
      if (last !== undefined && update.seq <= last) return false;
    }

    const t0 = update.range_start_s;
    const t1 = t0 + update.span_s;
    const incoming = (update.items as unknown as I[]).map(
      (item) =>
        ({ ...item, basis: update.basis, late: update.late }) as StoredItem<I>,
    );
    const kept = this._items.filter(
      (item) => item.start_s < t0 || item.start_s >= t1,
    );
    this._items = kept.concat(incoming).sort(compareItems);
    this._rangeSeq.set(rangeKey(t0), update.seq);
    this._emit();
    return true;
  }

  /** Drop everything and forget the epoch and per-range seqs (on disable,
   *  session stop or reconnect; the next enable sends a fresh clear). */
  reset(): void {
    this._items = [];
    this._rangeSeq.clear();
    this._epoch = null;
    this._emit();
  }

  /** Every item, sorted by start time then pitch. Do not mutate. */
  items(): readonly StoredItem<I>[] {
    return this._items;
  }

  /** Items whose basis is `settled`, sorted. */
  settledItems(): StoredItem<I>[] {
    return this._items.filter((item) => item.basis === "settled");
  }

  /**
   * Items sounding anywhere in the window [t0, t1) of play time.
   *
   * Without `wrap` the window is taken literally in canvas seconds. With
   * `wrap` (the loop band, or `{start_s: 0, end_s: duration}` for a
   * looping canvas) the window is unrolled over the band's period: the
   * parts of [t0, t1) beyond either band edge read the band's other end,
   * and each returned item carries the `shift_s` that places it in the
   * window. Items are ordered by window position.
   */
  itemsIn(t0: number, t1: number, wrap?: LoopBand | null): WindowedItem<I>[] {
    if (!(t1 > t0)) return [];
    const overlap = (a: number, b: number, shift: number): WindowedItem<I>[] => {
      const out: WindowedItem<I>[] = [];
      for (const item of this._items) {
        if (item.start_s >= b) break;
        if (item.end_s > a || item.start_s >= a) {
          out.push({ item, shift_s: shift });
        }
      }
      return out;
    };
    if (!wrap || !(wrap.end_s > wrap.start_s)) return overlap(t0, t1, 0);

    const period = wrap.end_s - wrap.start_s;
    const out: WindowedItem<I>[] = [];
    const kFirst = Math.floor((t0 - wrap.start_s) / period);
    const kLast = Math.ceil((t1 - wrap.start_s) / period) - 1;
    for (let k = kFirst; k <= kLast; k++) {
      const shift = k * period;
      const a = Math.max(t0, wrap.start_s + shift) - shift;
      const b = Math.min(t1, wrap.end_s + shift) - shift;
      if (b > a) out.push(...overlap(a, b, shift));
    }
    return out;
  }

  /** Call `listener` after every accepted update or reset. Returns the
   *  unsubscribe function. */
  subscribe(listener: AnalysisStoreListener): () => void {
    this._listeners.add(listener);
    return () => {
      this._listeners.delete(listener);
    };
  }

  /**
   * Feed this store from a RemoteBackend (or any EventTarget dispatching
   * `analysis_update` CustomEvents). Returns the detach function.
   */
  attach(target: EventTarget): () => void {
    const onUpdate = (event: Event) => {
      const detail = (event as CustomEvent<AnalysisUpdateEventPayload>).detail;
      if (detail) this.apply(detail);
    };
    target.addEventListener("analysis_update", onUpdate);
    return () => target.removeEventListener("analysis_update", onUpdate);
  }

  private _emit(): void {
    for (const listener of this._listeners) {
      try {
        listener();
      } catch (err) {
        console.error("[analysis] store listener failed", err);
      }
    }
  }
}
