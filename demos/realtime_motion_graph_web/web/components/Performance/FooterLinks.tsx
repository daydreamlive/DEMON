"use client";

// Top-right CTA pair. Both open Tally as an in-page modal (lazy-loaded
// script on first click) instead of opening a new tab.
//   • VST Waitlist — filled accent pill (`footer-link--cta`). Primary.
//   • Feedback     — outlined accent pill. Secondary.

declare global {
  interface Window {
    Tally?: {
      openPopup: (
        formId: string,
        options?: { layout?: "default" | "modal"; width?: number },
      ) => void;
      closePopup: (formId: string) => void;
    };
  }
}

const TALLY_SCRIPT_SRC = "https://tally.so/widgets/embed.js";
const VST_FORM_ID = "q4jxo9";
const FEEDBACK_FORM_ID = "oblP5X";

let tallyLoaderPromise: Promise<void> | null = null;

function loadTally(): Promise<void> {
  if (typeof window === "undefined") return Promise.resolve();
  if (window.Tally) return Promise.resolve();
  if (tallyLoaderPromise) return tallyLoaderPromise;
  tallyLoaderPromise = new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(
      `script[src="${TALLY_SCRIPT_SRC}"]`,
    );
    if (existing) {
      if (window.Tally) {
        resolve();
        return;
      }
      existing.addEventListener("load", () => resolve(), { once: true });
      existing.addEventListener("error", () => reject(new Error("tally load failed")), { once: true });
      return;
    }
    const script = document.createElement("script");
    script.src = TALLY_SCRIPT_SRC;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("tally load failed"));
    document.head.appendChild(script);
  });
  return tallyLoaderPromise;
}

async function openTally(formId: string): Promise<void> {
  try {
    await loadTally();
    if (window.Tally) {
      window.Tally.openPopup(formId, { layout: "modal", width: 700 });
      return;
    }
    throw new Error("Tally global missing after load");
  } catch {
    // Fallback if the script can't load (ad blocker, offline, etc.):
    // open the Tally share URL in a new tab so the user can still
    // reach the form.
    window.open(
      `https://tally.so/r/${formId}`,
      "_blank",
      "noopener,noreferrer",
    );
  }
}

export function FooterLinks() {
  return (
    <div className="footer-links" aria-label="Help and feedback">
      <button
        type="button"
        className="footer-link footer-link--cta"
        onClick={() => void openTally(VST_FORM_ID)}
        title="Join the VST waitlist"
      >
        <span className="footer-link-icon" aria-hidden="true">
          <svg
            viewBox="0 0 16 16"
            width={12}
            height={12}
            fill="none"
            stroke="currentColor"
            strokeWidth={1.4}
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="4" y1="2.5" x2="4" y2="13.5" />
            <line x1="8" y1="2.5" x2="8" y2="13.5" />
            <line x1="12" y1="2.5" x2="12" y2="13.5" />
            <rect x="2.5" y="6" width="3" height="2" rx="0.4" />
            <rect x="6.5" y="9.5" width="3" height="2" rx="0.4" />
            <rect x="10.5" y="4.5" width="3" height="2" rx="0.4" />
          </svg>
        </span>
        <span className="footer-link-label">VST Waitlist</span>
      </button>
      <button
        type="button"
        className="footer-link"
        onClick={() => void openTally(FEEDBACK_FORM_ID)}
        title="Report a bug or send feedback"
      >
        <span className="footer-link-icon" aria-hidden="true">
          <svg
            viewBox="0 0 16 16"
            width={12}
            height={12}
            fill="none"
            stroke="currentColor"
            strokeWidth={1.4}
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M2.5 3.5h11a1 1 0 0 1 1 1v6a1 1 0 0 1-1 1H9.5l-3 2.5v-2.5H2.5a1 1 0 0 1-1-1v-6a1 1 0 0 1 1-1z" />
            <line x1="5" y1="7" x2="11" y2="7" />
            <line x1="5" y1="9.5" x2="9" y2="9.5" />
          </svg>
        </span>
        <span className="footer-link-label">Feedback</span>
      </button>
    </div>
  );
}
