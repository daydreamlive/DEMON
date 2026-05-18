"use client";

// Always-visible footer pill: VST WAITLIST · BUGS & FEEDBACK. Both
// were previously buried inside the HaloBadge menu — moved here so the
// two highest-value calls to action (sign up for the VST, tell us
// what's broken) are reachable in one click from any canvas state.
//
// Pinned to the very bottom of the viewport, behind/beneath the
// drawer handle. When the drawer opens upward it visually covers the
// strip — fine, the user has chosen to focus on controls in that
// state. Mobile: hidden (the lite drawer + halo menu cover this).

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
      <span className="footer-link-sep" aria-hidden="true">
        ·
      </span>
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
