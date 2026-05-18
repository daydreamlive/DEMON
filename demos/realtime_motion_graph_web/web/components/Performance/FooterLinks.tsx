"use client";

// Always-visible top-right chrome: VST waitlist + bugs/feedback. Quiet
// hairline pills — the high-value CTAs are one click away from any
// canvas state without competing with the hardware-knob vocabulary.
// No icons, no fills, no bevels: just a 1px frame-line outline around a
// mono uppercase label. Hover lifts to text-strong + accent-line border.

export function FooterLinks() {
  return (
    <div className="footer-links" aria-label="Help and feedback">
      <a
        href="https://tally.so/r/q4jxo9"
        className="footer-link"
        target="_blank"
        rel="noopener noreferrer"
        title="Join the VST waitlist"
      >
        VST Waitlist
      </a>
      <a
        href="https://tally.so/r/oblP5X"
        className="footer-link"
        target="_blank"
        rel="noopener noreferrer"
        title="Report a bug or send feedback"
      >
        Feedback
      </a>
    </div>
  );
}
