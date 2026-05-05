// Audio-reactive video shader pipeline.
//
// Lifted from the shaders/ prototype, slimmed to the two effects that
// matter for the install: color parallax (saturation-driven horizontal
// shift) and bloom-on-kick. No depth maps, no optical flow, no edges,
// no preprocessing required: parallax keys off pixel saturation, bloom
// keys off the kick envelope already computed in app.js.
//
// Pipeline (5 passes, raw WebGL2):
//   1. parallax: video -> sceneRT
//   2. bright:   sceneRT -> bloomRT0   (luminance threshold)
//   3. blur H:   bloomRT0 -> bloomRT1
//   4. blur V:   bloomRT1 -> bloomRT0
//   5. composite scene + bloom*kick -> screen

const VERT = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const PARALLAX_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uSource;
uniform float uTime;
uniform float uKick;
uniform float uParallaxStrength;
uniform float uWarpStrength;   // global warp: baseline writhe + kick pulse
in vec2 vUv;
out vec4 fragColor;

const float SWAY_HZ = 0.7;
const float SWAY_AMP = 0.0008;
const float KICK_SHIFT = 0.012;
const float SAT_KNEE_LO = 0.40;
const float SAT_KNEE_HI = 0.70;
const float WARP_SCALE = 0.025;   // peak UV offset at full warp
const float WARP_BASE  = 0.30;    // always-on fraction (writhing baseline)

float satOf(vec3 rgb) {
  float maxC = max(max(rgb.r, rgb.g), rgb.b);
  float minC = min(min(rgb.r, rgb.g), rgb.b);
  return maxC > 0.001 ? (maxC - minC) / maxC : 0.0;
}

// Two-octave trig noise. Each octave is a product of out-of-phase sin/cos
// across x/y, with frequencies that aren't simple multiples of each other,
// so the resulting field reads as organic / random rather than gridded.
// Time advances each component at a slightly different rate so the pattern
// drifts continuously instead of pulsing in lockstep.
vec2 organicWarp(vec2 uv, float t) {
  vec2 p = uv * 4.0;
  vec2 v;
  v.x = sin(p.x * 1.3 + t * 0.9) * cos(p.y * 1.7 - t * 0.6);
  v.y = cos(p.x * 1.9 - t * 0.5) * sin(p.y * 1.1 + t * 1.2);
  v.x += 0.5 * sin(p.x * 3.7 + t * 1.7) * cos(p.y * 4.1 - t * 1.3);
  v.y += 0.5 * cos(p.x * 4.3 - t * 1.5) * sin(p.y * 3.9 + t * 1.9);
  return v;
}

void main() {
  // Global warp first — distorts the whole frame organically before the
  // saturation-masked parallax sampling reads it. Baseline writhe is
  // always present; kick adds a punch on top.
  float warpAmt = (WARP_BASE + (1.0 - WARP_BASE) * uKick) * uWarpStrength;
  vec2 uv = vUv + organicWarp(vUv, uTime) * WARP_SCALE * warpAmt;

  float sway = sin(uTime * 6.2831853 * SWAY_HZ) * SWAY_AMP;
  float kick = uKick * KICK_SHIFT;
  vec2 shift = vec2(sway + kick, 0.0) * uParallaxStrength;

  vec3 srcLocal = texture(uSource, uv).rgb;
  vec3 srcShifted = texture(uSource, uv - shift).rgb;

  float satShifted = satOf(srcShifted);
  float mask = smoothstep(SAT_KNEE_LO, SAT_KNEE_HI, satShifted);
  // Soft falloff when local pixel is also colorful, to avoid hard ghosting.
  float satLocal = satOf(srcLocal);
  mask *= 1.0 - smoothstep(0.6, 1.0, satLocal) * 0.4;

  vec3 col = mix(srcLocal, srcShifted, mask);
  fragColor = vec4(col, 1.0);
}`;

const BRIGHT_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uScene;
uniform float uThreshold;
uniform float uKnee;
in vec2 vUv;
out vec4 fragColor;

void main() {
  vec3 c = texture(uScene, vUv).rgb;
  float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
  float k = max(0.0001, uKnee);
  float w = smoothstep(uThreshold - k, uThreshold + k, lum);
  fragColor = vec4(c * w, 1.0);
}`;

const BLUR_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uTexel;
uniform vec2 uDir;
uniform float uRadius;
in vec2 vUv;
out vec4 fragColor;

const float W0 = 0.227027;
const float W1 = 0.194595;
const float W2 = 0.121622;
const float W3 = 0.054054;
const float W4 = 0.016216;

void main() {
  vec2 stp = uDir * uTexel * (uRadius / 4.0);
  vec3 acc = texture(uTex, vUv).rgb * W0;
  acc += texture(uTex, vUv + stp * 1.0).rgb * W1;
  acc += texture(uTex, vUv - stp * 1.0).rgb * W1;
  acc += texture(uTex, vUv + stp * 2.0).rgb * W2;
  acc += texture(uTex, vUv - stp * 2.0).rgb * W2;
  acc += texture(uTex, vUv + stp * 3.0).rgb * W3;
  acc += texture(uTex, vUv - stp * 3.0).rgb * W3;
  acc += texture(uTex, vUv + stp * 4.0).rgb * W4;
  acc += texture(uTex, vUv - stp * 4.0).rgb * W4;
  fragColor = vec4(acc, 1.0);
}`;

// Composite pass folds in two knob-driven flavors on top of the baseline
// kick-driven bloom:
//   - Dubstep knob -> chromatic aberration in scene sampling. Channels
//     split toward/away from screen center with a quadratic falloff so
//     the center stays clean and the edges stress on bass.
//   - Daft Punk knob -> bloom flavor. Adds an always-on base bloom
//     (uBloomBase) on top of the kick-driven term, and tints the bloom
//     output toward warm rose (uBloomTint) to match fleshy footage.
const COMPOSITE_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uKick;
uniform float uBloomOnKick;
uniform float uBloomBase;     // daft: always-on bloom amount
uniform vec3  uBloomTint;     // daft: bloom color multiplier (white = none)
uniform float uChroma;        // dubstep: base UV offset for RGB split
uniform float uChromaKick;    // dubstep: extra UV offset on kick
in vec2 vUv;
out vec4 fragColor;

vec3 sampleSceneChroma(vec2 uv) {
  vec2 toCenter = uv - vec2(0.5);
  float r2 = dot(toCenter, toCenter);
  float falloff = 0.4 + 1.6 * r2;
  float amount = uChroma + uChromaKick * uKick;
  vec2 dir = normalize(toCenter + 1e-6) * amount * falloff;
  vec3 col;
  col.r = texture(uScene, uv + dir).r;
  col.g = texture(uScene, uv).g;
  col.b = texture(uScene, uv - dir).b;
  return col;
}

void main() {
  vec3 scene = sampleSceneChroma(vUv);
  vec3 bloom = texture(uBloom, vUv).rgb * uBloomTint;
  float bloomAmt = uBloomBase + uKick * uBloomOnKick;
  vec3 col = min(scene + bloom * bloomAmt, vec3(1.0));
  fragColor = vec4(col, 1.0);
}`;

const BLOOM_DOWNSCALE = 0.5;
// We want the blur to span a fixed *fraction of UV* regardless of the
// bloom RT's pixel dims: at the previous default (canvas-sized RTs, bloom
// at 0.5× → ~960 px wide on a 1080p×DPR2 canvas) the kernel covered
// 18 / 960 ≈ 0.01875 UV. Scale uRadius from bloomW so smaller RTs blur
// the same amount on screen.
const BLOOM_BLUR_UV = 18 / 960;

// Internal RT pixel cap. Scene/bloom RTs don't need to track the canvas
// resolution — the source is video, the pipeline is parallax + warp +
// bloom + chroma split (none of which preserve sub-source-pixel detail),
// and the composite blit upsamples to canvas with a linear filter.
// Sizing internal RTs to min(video native, canvas, this cap) keeps VRAM
// flat with the source instead of growing 4–16× with display DPR/4K.
const MAX_INTERNAL_SIDE = 1920;

function compileShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`Shader compile failed: ${log}`);
  }
  return sh;
}

function makeProgram(gl, fragSrc) {
  const prog = gl.createProgram();
  const v = compileShader(gl, gl.VERTEX_SHADER, VERT);
  const f = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  gl.attachShader(prog, v);
  gl.attachShader(prog, f);
  gl.bindAttribLocation(prog, 0, "aPos");
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error(`Program link failed: ${log}`);
  }
  gl.deleteShader(v);
  gl.deleteShader(f);
  return prog;
}

function makeRT(gl, w, h) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { tex, fbo, w, h };
}

function disposeRT(gl, rt) {
  if (!rt) return;
  gl.deleteTexture(rt.tex);
  gl.deleteFramebuffer(rt.fbo);
}

export class EffectsRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", {
      // alpha:true so that on cold load (before the first successful
      // frame upload) the canvas can be cleared to fully transparent
      // and the raw <video> underneath shows through. The composite
      // shader still writes alpha=1 for normal frames so the shader
      // output covers the video opaquely once we're rendering.
      alpha: true,
      antialias: false,
      preserveDrawingBuffer: false,
      powerPreference: "high-performance",
    });
    if (!gl) throw new Error("WebGL2 not supported");
    this.gl = gl;

    // One big triangle covers [-1,1]² — a touch faster than a quad and
    // the off-screen verts get clipped for free.
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);
    gl.bindVertexArray(null);
    this._vao = vao;
    this._buf = buf;

    this._parallax = makeProgram(gl, PARALLAX_FRAG);
    this._bright = makeProgram(gl, BRIGHT_FRAG);
    this._blur = makeProgram(gl, BLUR_FRAG);
    this._composite = makeProgram(gl, COMPOSITE_FRAG);

    this._u = {
      parallax: {
        uSource: gl.getUniformLocation(this._parallax, "uSource"),
        uTime: gl.getUniformLocation(this._parallax, "uTime"),
        uKick: gl.getUniformLocation(this._parallax, "uKick"),
        uParallaxStrength: gl.getUniformLocation(this._parallax, "uParallaxStrength"),
        uWarpStrength: gl.getUniformLocation(this._parallax, "uWarpStrength"),
      },
      bright: {
        uScene: gl.getUniformLocation(this._bright, "uScene"),
        uThreshold: gl.getUniformLocation(this._bright, "uThreshold"),
        uKnee: gl.getUniformLocation(this._bright, "uKnee"),
      },
      blur: {
        uTex: gl.getUniformLocation(this._blur, "uTex"),
        uTexel: gl.getUniformLocation(this._blur, "uTexel"),
        uDir: gl.getUniformLocation(this._blur, "uDir"),
        uRadius: gl.getUniformLocation(this._blur, "uRadius"),
      },
      composite: {
        uScene: gl.getUniformLocation(this._composite, "uScene"),
        uBloom: gl.getUniformLocation(this._composite, "uBloom"),
        uKick: gl.getUniformLocation(this._composite, "uKick"),
        uBloomOnKick: gl.getUniformLocation(this._composite, "uBloomOnKick"),
        uBloomBase: gl.getUniformLocation(this._composite, "uBloomBase"),
        uBloomTint: gl.getUniformLocation(this._composite, "uBloomTint"),
        uChroma: gl.getUniformLocation(this._composite, "uChroma"),
        uChromaKick: gl.getUniformLocation(this._composite, "uChromaKick"),
      },
    };

    this._srcTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this._srcTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    // Set unpack flip once at construction — toggling this per-upload
    // can re-trigger driver slow paths even if the value is unchanged.
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    this._srcW = 0;
    this._srcH = 0;
    // Track the last uploaded video frame so we can skip redundant
    // texSubImage2D calls when the <video> hasn't advanced (paused,
    // wrapped, decoder stalled). Saves a per-frame upload + GPU sync.
    this._lastVideoTime = -1;
    this._lastVideoEl = null;

    this._scene = null;
    this._bloom0 = null;
    this._bloom1 = null;

    this._parallaxStrength = 0.4;
    this._bloomOnKick = 0.3;
    this._bloomThreshold = 0.15;
    this._bloomKnee = 0.20;
    this._warpStrength = 0.4;
    // Knob-driven flavors (0..1). Dubstep adds chromatic aberration;
    // daft punk adds an always-on rose-tinted bloom.
    this._dubstep = 0;
    this._daft = 0;
    // First-frame state. Until we successfully upload at least one
    // video frame we leave the canvas faded out via CSS so the raw
    // <video> shows through, avoiding a black gap on cold load.
    this._everDrew = false;

    this._resizeObs = new ResizeObserver(() => this._resize());
    this._resizeObs.observe(canvas);
    this._resize();
  }

  setParallaxStrength(v) { this._parallaxStrength = v; }
  setBloomOnKick(v) { this._bloomOnKick = v; }
  setBloomThreshold(v) { this._bloomThreshold = v; }
  setWarpStrength(v) { this._warpStrength = v; }
  // Both knobs are clamped 0..1 by the caller (app.js normalizes the
  // slider's max-range value).
  setDubstep(v) { this._dubstep = v < 0 ? 0 : v > 1 ? 1 : v; }
  setDaftPunk(v) { this._daft = v < 0 ? 0 : v > 1 ? 1 : v; }

  _resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(2, Math.floor(rect.width * dpr));
    const h = Math.max(2, Math.floor(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this._reallocateRTs();
  }

  // Choose internal scene/bloom dims based on:
  //   - video native size (cap detail at the source — anything larger is
  //     bilinear upsample and burns VRAM for nothing)
  //   - canvas backing size (don't render bigger than the display)
  //   - MAX_INTERNAL_SIDE (absolute ceiling so 4K monitors with 4K source
  //     still don't allocate gigabytes of FX render targets)
  // Aspect always matches the canvas so the composite shader's vUv
  // sampling lines up with the original screen-space coordinates.
  _reallocateRTs() {
    const gl = this.gl;
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const aspect = cw / ch;

    // Default budget: canvas dims (capped at MAX_INTERNAL_SIDE).
    let target = Math.min(Math.max(cw, ch), MAX_INTERNAL_SIDE);
    // If the video is smaller than the canvas, drop the budget further —
    // the source can't supply more detail than its native res.
    if (this._srcW > 0 && this._srcH > 0) {
      target = Math.min(target, Math.max(this._srcW, this._srcH));
    }

    let iw, ih;
    if (aspect >= 1) {
      iw = target;
      ih = Math.max(2, Math.round(target / aspect));
    } else {
      ih = target;
      iw = Math.max(2, Math.round(target * aspect));
    }

    if (this._scene && this._scene.w === iw && this._scene.h === ih) return;

    disposeRT(gl, this._scene);
    this._scene = makeRT(gl, iw, ih);

    const bw = Math.max(2, Math.round(iw * BLOOM_DOWNSCALE));
    const bh = Math.max(2, Math.round(ih * BLOOM_DOWNSCALE));
    disposeRT(gl, this._bloom0);
    disposeRT(gl, this._bloom1);
    this._bloom0 = makeRT(gl, bw, bh);
    this._bloom1 = makeRT(gl, bw, bh);
  }

  _uploadVideo(videoEl) {
    // Returns true if we have *something* to render (fresh frame or a
    // prior frame still in the texture). Returns false only when we've
    // never had a frame yet — caller should skip the draw entirely so
    // the canvas stays faded out.
    if (!videoEl) return this._srcW > 0;
    const vw = videoEl.videoWidth | 0;
    const vh = videoEl.videoHeight | 0;
    const ready = videoEl.readyState >= 2 && vw > 0 && vh > 0;
    if (!ready) {
      // Common path during a <video loop> wrap: readyState briefly dips
      // below 2. Reuse the previous texture instead of clearing.
      return this._srcW > 0;
    }
    // Skip the upload if the video element hasn't advanced since last
    // frame (paused, decoder stalled, rAF firing faster than playback).
    // Re-uploading the same frame is one of the dominant per-frame VRAM
    // bandwidth costs and produces zero visual change.
    const t = videoEl.currentTime;
    if (videoEl === this._lastVideoEl
        && this._srcW === vw && this._srcH === vh
        && t === this._lastVideoTime) {
      return true;
    }
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this._srcTex);
    if (this._srcW !== vw || this._srcH !== vh) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, videoEl);
      this._srcW = vw;
      this._srcH = vh;
      // Source dims changed — reallocate internal RTs so we don't carry
      // a giant scene RT for a tiny video (or vice versa).
      this._reallocateRTs();
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, videoEl);
    }
    this._lastVideoEl = videoEl;
    this._lastVideoTime = t;
    return true;
  }

  _drawTo(target) {
    const gl = this.gl;
    if (target) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
      gl.viewport(0, 0, target.w, target.h);
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }
    gl.bindVertexArray(this._vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  tick(videoEl, timeSeconds, kick) {
    // Backgrounded tab — don't burn the GPU on warp/bloom passes nobody
    // can see. The browser already throttles rAF here, but skipping the
    // shader chain entirely also frees the video decoder to coast.
    if (typeof document !== "undefined" && document.visibilityState === "hidden") return;

    const gl = this.gl;
    if (!this._uploadVideo(videoEl)) {
      // Cold load — no frame ever decoded. Clear the default
      // framebuffer to fully transparent so the <video> behind the
      // canvas is what the user sees. Combined with the CSS opacity:0
      // baseline this means no black gap on startup.
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      return;
    }

    // 1. Parallax (with global organic warp): video -> scene
    gl.useProgram(this._parallax);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._srcTex);
    gl.uniform1i(this._u.parallax.uSource, 0);
    gl.uniform1f(this._u.parallax.uTime, timeSeconds);
    gl.uniform1f(this._u.parallax.uKick, kick);
    gl.uniform1f(this._u.parallax.uParallaxStrength, this._parallaxStrength);
    gl.uniform1f(this._u.parallax.uWarpStrength, this._warpStrength);
    this._drawTo(this._scene);

    // 2. Brightpass: scene -> bloom0. Daft-punk knob drops the threshold
    // so progressively more of the frame glows as the knob rises; the
    // floor (0.02) keeps the brightpass from blowing out at knob=max.
    gl.useProgram(this._bright);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._scene.tex);
    gl.uniform1i(this._u.bright.uScene, 0);
    const effThreshold = Math.max(0.02, this._bloomThreshold - this._daft * 0.10);
    gl.uniform1f(this._u.bright.uThreshold, effThreshold);
    gl.uniform1f(this._u.bright.uKnee, this._bloomKnee);
    this._drawTo(this._bloom0);

    // 3+4. Two-pass Gaussian blur: bloom0 -> bloom1 -> bloom0
    // Radius scales with bloom RT width so the blur covers the same
    // fraction of UV regardless of internal resolution — visual spread
    // stays constant when the bloom RT shrinks with the source video.
    gl.useProgram(this._blur);
    gl.uniform2f(this._u.blur.uTexel, 1 / this._bloom0.w, 1 / this._bloom0.h);
    gl.uniform1f(this._u.blur.uRadius, Math.max(2, this._bloom0.w * BLOOM_BLUR_UV));

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._bloom0.tex);
    gl.uniform1i(this._u.blur.uTex, 0);
    gl.uniform2f(this._u.blur.uDir, 1, 0);
    this._drawTo(this._bloom1);

    gl.bindTexture(gl.TEXTURE_2D, this._bloom1.tex);
    gl.uniform2f(this._u.blur.uDir, 0, 1);
    this._drawTo(this._bloom0);

    // 5. Composite: chroma-split scene + tinted bloom -> screen.
    gl.useProgram(this._composite);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._scene.tex);
    gl.uniform1i(this._u.composite.uScene, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this._bloom0.tex);
    gl.uniform1i(this._u.composite.uBloom, 1);
    gl.uniform1f(this._u.composite.uKick, kick);
    gl.uniform1f(this._u.composite.uBloomOnKick, this._bloomOnKick);

    // Daft Punk knob: always-on bloom + slight rose tint to match
    // fleshy footage. Tint mix capped at 0.5 so even at knob=max it
    // reads as a tint rather than a coloured light.
    const daft = this._daft;
    gl.uniform1f(this._u.composite.uBloomBase, daft * 0.3);
    const tintMix = daft * 0.5;
    gl.uniform3f(
      this._u.composite.uBloomTint,
      1.0 + (1.0   - 1.0) * tintMix,  // R stays at 1.0
      1.0 + (0.78  - 1.0) * tintMix,
      1.0 + (0.82  - 1.0) * tintMix,
    );

    // Dubstep knob: chromatic aberration in scene sampling.
    const dub = this._dubstep;
    gl.uniform1f(this._u.composite.uChroma, dub * 0.005);
    gl.uniform1f(this._u.composite.uChromaKick, dub * 0.012);

    this._drawTo(null);

    // First successful render: reveal the canvas. CSS handles the fade
    // (see #effects-canvas opacity transition) so the raw <video>
    // beneath crossfades into the shader output.
    if (!this._everDrew) {
      this._everDrew = true;
      this.canvas.classList.add("effects-ready");
    }
  }

  destroy() {
    const gl = this.gl;
    this._resizeObs.disconnect();
    disposeRT(gl, this._scene);
    disposeRT(gl, this._bloom0);
    disposeRT(gl, this._bloom1);
    gl.deleteTexture(this._srcTex);
    gl.deleteVertexArray(this._vao);
    gl.deleteBuffer(this._buf);
    gl.deleteProgram(this._parallax);
    gl.deleteProgram(this._bright);
    gl.deleteProgram(this._blur);
    gl.deleteProgram(this._composite);
  }
}
