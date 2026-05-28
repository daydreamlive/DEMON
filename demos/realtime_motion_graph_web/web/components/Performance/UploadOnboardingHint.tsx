"use client";

// Onboarding affordance for the bottom-right Upload button — appears
// after the Strength hint dismisses (see useUploadOnboardingHint).
// Issue #156: users reported the Upload button is hard to find tucked
// in the corner. Mirrors StrengthOnboardingHint's visual treatment
// (white "label + arrow gif") so the two reads as the same
// vocabulary.
//
// Pure presentational — visibility/lifecycle owned by AudioSourceCrate
// via useUploadOnboardingHint so the dock can keep itself expanded
// while the hint is on screen (the dock otherwise auto-collapses to a
// music-note bubble after 2s of idle pointer).

export function UploadOnboardingHint({ visible }: { visible: boolean }) {
  if (!visible) return null;
  return (
    <div className="upload-onboarding-hint" aria-hidden="true">
      <span className="upload-onboarding-hint-text">Upload your own track</span>
      <img
        className="upload-onboarding-hint-arrow"
        src="/strength-onboarding-arrow.gif"
        alt=""
        width={68}
        height={68}
      />
    </div>
  );
}
