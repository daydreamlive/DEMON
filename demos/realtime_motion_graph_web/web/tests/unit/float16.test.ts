// Tier A: the hand-rolled float16 -> float32 decoder in sdk/protocol.ts
// (browsers have no native float16; the worker has an identical copy).
// Node 24 ships native Float16Array, which gives us an exhaustive oracle:
// every one of the 65536 half-precision bit patterns must decode to the
// IEEE-exact float32 value (f16 -> f32 widening is always exact).

import { describe, expect, it } from "vitest";

import { float16ArrayToFloat32 } from "@demon/client";

function bitsOf(v: number): number {
  const b = new ArrayBuffer(4);
  new Float32Array(b)[0] = v;
  return new Uint32Array(b)[0];
}

describe("float16ArrayToFloat32", () => {
  it("decodes every finite/subnormal/special bit pattern exactly", () => {
    const u16 = new Uint16Array(65536);
    for (let i = 0; i < 65536; i++) u16[i] = i;

    // Native oracle: reinterpret the same bits as half-precision floats.
    const oracle = new Float16Array(u16.buffer);
    const got = float16ArrayToFloat32(u16);

    let bad = 0;
    let firstBad = -1;
    for (let i = 0; i < 65536; i++) {
      const want = oracle[i]; // exact f32 (f16 widens losslessly)
      const g = got[i];
      const ok = Number.isNaN(want)
        ? Number.isNaN(g)
        : bitsOf(g) === bitsOf(want); // bit compare catches -0 vs +0
      if (!ok) {
        bad++;
        if (firstBad < 0) firstBad = i;
      }
    }
    expect(
      bad,
      `decoder disagrees with native Float16Array (first bad bit ` +
        `pattern 0x${firstBad.toString(16)})`,
    ).toBe(0);
  });

  it("handles the values audio actually carries", () => {
    // Spot checks with hand-derived expectations, independent of the
    // native oracle: zero, +-1, 0.5, the max normal half (65504), and
    // the smallest subnormal (2^-24).
    const cases: [number, number][] = [
      [0x0000, 0],
      [0x8000, -0],
      [0x3c00, 1],
      [0xbc00, -1],
      [0x3800, 0.5],
      [0x7bff, 65504],
      [0x0001, 2 ** -24],
      [0x8001, -(2 ** -24)],
    ];
    const u16 = new Uint16Array(cases.map(([h]) => h));
    const got = float16ArrayToFloat32(u16);
    cases.forEach(([h, want], i) => {
      expect(bitsOf(got[i]), `0x${h.toString(16)}`).toBe(bitsOf(want));
    });
  });

  it("maps infinities and NaN", () => {
    const got = float16ArrayToFloat32(
      new Uint16Array([0x7c00, 0xfc00, 0x7e00]),
    );
    expect(got[0]).toBe(Infinity);
    expect(got[1]).toBe(-Infinity);
    expect(Number.isNaN(got[2])).toBe(true);
  });

  it("preserves length and handles empty input", () => {
    expect(float16ArrayToFloat32(new Uint16Array(0))).toHaveLength(0);
    expect(float16ArrayToFloat32(new Uint16Array(7))).toHaveLength(7);
  });
});
