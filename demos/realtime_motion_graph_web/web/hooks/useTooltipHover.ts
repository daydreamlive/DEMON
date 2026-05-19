import { useEffect, useState, type RefObject } from "react";

// Resolve the data-dd-tooltip nearest to the pointer and surface its
// long-form copy + title. Backs both DrawerHelpBar (scoped to the
// Full Controls panel) and HudHelpReadout (ambient, listens at window
// scope). Title precedence:
//   1. explicit `data-dd-tooltip-title` — knobs/sliders/ref controls
//      set this so the readout shows the canonical label.
//   2. `aria-label` — buttons.
//   3. first 4 tokens of textContent — last-resort fallback.
//
// Without (1), a knob wrapper's textContent leaked the value + kbd
// shortcut into the title ("Denoise 0.50 A + ▲▼"), which is why the
// explicit attr exists.

interface UseTooltipHoverOptions {
  /** Element to scope listeners to. If omitted, listens on window
   *  pointermove + document pointerleave (ambient mode). */
  getRoot?: () => Element | null;
  /** Optional element to exclude from hover detection — point hovering
   *  the readout itself shouldn't clear/overwrite the text being read. */
  selfRef?: RefObject<HTMLElement | null>;
}

interface TooltipHoverResult {
  title: string | null;
  text: string | null;
}

export function useTooltipHover(
  options: UseTooltipHoverOptions = {},
): TooltipHoverResult {
  const { getRoot, selfRef } = options;
  const [text, setText] = useState<string | null>(null);
  const [title, setTitle] = useState<string | null>(null);

  useEffect(() => {
    const root = getRoot ? getRoot() : null;
    if (getRoot && !root) return;

    const onMove = (e: PointerEvent) => {
      const target = e.target as Element | null;
      if (!target) return;
      // Ignore hover over the readout itself — otherwise hovering the
      // text would clear / overwrite the very text being read.
      if (selfRef?.current && selfRef.current.contains(target)) return;
      const el = target.closest<HTMLElement>("[data-dd-tooltip]");
      if (!el) {
        setText(null);
        setTitle(null);
        return;
      }
      const t = el.getAttribute("data-dd-tooltip");
      if (!t) {
        setText(null);
        setTitle(null);
        return;
      }
      setText(t);
      const explicit = el.getAttribute("data-dd-tooltip-title");
      const aria = el.getAttribute("aria-label");
      const visible = el.textContent?.trim().split(/\s+/).slice(0, 4).join(" ");
      setTitle(explicit || aria || visible || null);
    };

    const onLeave = (e: PointerEvent) => {
      const next = e.relatedTarget as Element | null;
      // Stay populated as long as the pointer is still inside the
      // listening surface — without this, moving between siblings
      // inside the scoped root would flicker the readout.
      if (next && (!root || root.contains(next))) return;
      setText(null);
      setTitle(null);
    };

    const moveTarget: EventTarget = root ?? window;
    const leaveTarget: EventTarget = root ?? document;
    moveTarget.addEventListener("pointermove", onMove as EventListener);
    leaveTarget.addEventListener("pointerleave", onLeave as EventListener);
    return () => {
      moveTarget.removeEventListener("pointermove", onMove as EventListener);
      leaveTarget.removeEventListener("pointerleave", onLeave as EventListener);
    };
  }, [getRoot, selfRef]);

  return { title, text };
}
