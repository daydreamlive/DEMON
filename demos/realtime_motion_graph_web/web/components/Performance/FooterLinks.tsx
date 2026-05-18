"use client";

// Top-right CTA pair. Two visible tiers so the more important
// conversion target reads first:
//   • VST Waitlist — filled accent pill (`footer-link--cta`). Primary.
//   • Feedback     — outlined accent pill. Secondary, paired weight.
// Both share a brand-gradient hairline at the top (same vocabulary as
// the halo menu + audio-source-fan) so they read as part of the brand
// chrome family.

export function FooterLinks() {
  return (
    <div className="footer-links" aria-label="Help and feedback">
      <a
        href="https://tally.so/r/q4jxo9"
        className="footer-link footer-link--cta"
        target="_blank"
        rel="noopener noreferrer"
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
      </a>
      <a
        href="https://tally.so/r/oblP5X"
        className="footer-link"
        target="_blank"
        rel="noopener noreferrer"
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
      </a>
    </div>
  );
}
