import { useCallback, useEffect, useState, type RefObject } from "react";

// Mirror a horizontal scroll-snap track's scroll position to an active
// tab id, and provide a programmatic `gotoTab` that scrolls the target
// section into view. The track holds N `<section>` elements, each tagged
// `data-{attribute}="<tab id>"`; the IntersectionObserver picks the
// section with the highest intersectionRatio as the active tab.
//
// Used by the mobile LiteControls (Mix / Track) and MobileFullSheet
// (Core / Styles / Mod / Channels / Saved / Config). Same gesture, same
// observer thresholds — the only per-caller difference is which dataset
// key the sections carry and whether the parent is currently open.
//
// Caveat: when `enabled` flips off the observer disconnects and the
// active tab is *not* reset — re-opening lands on whichever tab was
// last visible, which is the intended behavior for both consumers.

interface UseScrollSyncedTabsOptions<T extends string> {
  /** Dataset key on each `<section>` — e.g., `"tab"` matches
   *  `data-tab="..."`. */
  attribute: string;
  /** Tab id to seed state with on mount. */
  initial: T;
  /** Skip the observer when the surface isn't mounted/visible (e.g.,
   *  MobileFullSheet uses `open` here). Defaults to `true`. */
  enabled?: boolean;
}

interface ScrollSyncedTabsResult<T extends string> {
  activeTab: T;
  gotoTab: (id: T) => void;
}

export function useScrollSyncedTabs<T extends string>(
  trackRef: RefObject<HTMLElement | null>,
  tabIds: readonly T[],
  options: UseScrollSyncedTabsOptions<T>,
): ScrollSyncedTabsResult<T> {
  const { attribute, initial, enabled = true } = options;
  const [activeTab, setActiveTab] = useState<T>(initial);

  useEffect(() => {
    if (!enabled) return;
    const root = trackRef.current;
    if (!root) return;

    const obs = new IntersectionObserver(
      (entries) => {
        let best: IntersectionObserverEntry | null = null;
        for (const e of entries) {
          if (!e.isIntersecting) continue;
          if (!best || e.intersectionRatio > best.intersectionRatio) best = e;
        }
        if (!best) return;
        const id = (best.target as HTMLElement).dataset[attribute] as T | undefined;
        if (id) setActiveTab(id);
      },
      { root, threshold: [0.5, 0.75, 1] },
    );
    for (const t of tabIds) {
      const el = root.querySelector<HTMLElement>(`[data-${attribute}="${t}"]`);
      if (el) obs.observe(el);
    }
    return () => obs.disconnect();
  }, [trackRef, tabIds, attribute, enabled]);

  const gotoTab = useCallback(
    (id: T) => {
      const root = trackRef.current;
      if (!root) return;
      const el = root.querySelector<HTMLElement>(`[data-${attribute}="${id}"]`);
      if (!el) return;
      el.scrollIntoView({ behavior: "smooth", inline: "start", block: "nearest" });
    },
    [trackRef, attribute],
  );

  return { activeTab, gotoTab };
}
