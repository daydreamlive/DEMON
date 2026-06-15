// Loop-phrase wire injection: gating, strip/re-prepend idempotency, and
// composition with the LoRA-trigger prefix in `wirePromptTransform`.

import { beforeEach, describe, expect, it, vi } from "vitest";

// Mutable mock state, hoisted so the vi.mock factories (which run before
// the imports) can close over it. Each test tweaks these fields.
const state = vi.hoisted(() => ({
  engine: {
    auto_prepend_lora_triggers: false,
    auto_prepend_loop_phrase: false,
    loop_phrase: "a short perfect loop of",
  } as Record<string, unknown>,
  lora: { enabled: new Set<string>(), catalog: [] as Array<{ id: string; metadata?: { primary_trigger_word?: string } }> },
}));

vi.mock("@/lib/config", () => ({ getConfig: () => ({ engine: state.engine }) }));
vi.mock("@/store/useLoraStore", () => ({
  useLoraStore: { getState: () => state.lora },
}));

import { loopPhrasePrefix, stripLeadingLoopPhrase } from "@/lib/loopPhrase";
import { wirePromptTransform } from "@/lib/loraTriggers";

beforeEach(() => {
  state.engine = {
    auto_prepend_lora_triggers: false,
    auto_prepend_loop_phrase: false,
    loop_phrase: "a short perfect loop of",
  };
  state.lora = { enabled: new Set<string>(), catalog: [] };
});

describe("loopPhrasePrefix", () => {
  it("is empty when the flag is off (default)", () => {
    expect(loopPhrasePrefix()).toBe("");
  });

  it("is the phrase + trailing space when on", () => {
    state.engine.auto_prepend_loop_phrase = true;
    expect(loopPhrasePrefix()).toBe("a short perfect loop of ");
  });

  it("honours a custom phrase", () => {
    state.engine.auto_prepend_loop_phrase = true;
    state.engine.loop_phrase = "seamless repeating loop of";
    expect(loopPhrasePrefix()).toBe("seamless repeating loop of ");
  });

  it("is empty when the phrase is blank even if the flag is on", () => {
    state.engine.auto_prepend_loop_phrase = true;
    state.engine.loop_phrase = "   ";
    expect(loopPhrasePrefix()).toBe("");
  });
});

describe("stripLeadingLoopPhrase", () => {
  it("removes a single leading phrase", () => {
    expect(stripLeadingLoopPhrase("a short perfect loop of driving techno")).toBe(
      "driving techno",
    );
  });

  it("removes accidental stacking", () => {
    expect(
      stripLeadingLoopPhrase(
        "a short perfect loop of a short perfect loop of pads",
      ),
    ).toBe("pads");
  });

  it("is case-insensitive", () => {
    expect(stripLeadingLoopPhrase("A Short Perfect Loop Of pads")).toBe("pads");
  });

  it("leaves a prompt without the phrase untouched", () => {
    expect(stripLeadingLoopPhrase("driving techno")).toBe("driving techno");
  });

  it("does not strip a bare phrase with no trailing content", () => {
    // No trailing space after the phrase -> not treated as a prefix.
    expect(stripLeadingLoopPhrase("a short perfect loop of")).toBe(
      "a short perfect loop of",
    );
  });
});

describe("wirePromptTransform (loop phrase)", () => {
  it("passes the prompt through untouched when both features are off", () => {
    expect(wirePromptTransform("driving techno")).toBe("driving techno");
  });

  it("prepends the loop phrase when enabled", () => {
    state.engine.auto_prepend_loop_phrase = true;
    expect(wirePromptTransform("driving techno")).toBe(
      "a short perfect loop of driving techno",
    );
  });

  it("is idempotent: re-sending its own output does not double-prepend", () => {
    state.engine.auto_prepend_loop_phrase = true;
    const once = wirePromptTransform("driving techno");
    expect(wirePromptTransform(once)).toBe(once);
  });

  it("orders triggers before the loop phrase, then the clean prompt", () => {
    state.engine.auto_prepend_lora_triggers = true;
    state.engine.auto_prepend_loop_phrase = true;
    state.lora = {
      enabled: new Set(["phonk"]),
      catalog: [{ id: "phonk", metadata: { primary_trigger_word: "phonk" } }],
    };
    expect(wirePromptTransform("driving techno")).toBe(
      "phonk, a short perfect loop of driving techno",
    );
  });

  it("stays idempotent with both prefixes active", () => {
    state.engine.auto_prepend_lora_triggers = true;
    state.engine.auto_prepend_loop_phrase = true;
    state.lora = {
      enabled: new Set(["phonk"]),
      catalog: [{ id: "phonk", metadata: { primary_trigger_word: "phonk" } }],
    };
    const once = wirePromptTransform("driving techno");
    expect(wirePromptTransform(once)).toBe(once);
  });
});
