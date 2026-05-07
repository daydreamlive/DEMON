// Audio-reactive video shader pipeline (WebGL2, 5-pass).
// Direct port of static/effects.js. Phase 6 perf: blur radius 18 → 12 (looks
// nearly identical, ~1.5x cheaper); render targets resize only on actual
// viewport change (already in place).
//
// Pipeline:
//   1. parallax  video -> sceneRT       (saturation-masked horizontal shift + warp)
//   2. bright    sceneRT -> bloomRT0    (luminance threshold)
//   3. blur H    bloomRT0 -> bloomRT1
//   4. blur V    bloomRT1 -> bloomRT0
//   5. composite scene + bloom*kick -> screen (with chroma + tint flavors)

const VERT = /* glsl */ `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const PARALLAX_FRAG = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D uSource;
uniform float uTime;
uniform float uKick;
uniform float uParallaxStrength;
uniform float uWarpStrength;
in vec2 vUv;
out vec4 fragColor;

const float SWAY_HZ = 0.7;
const float SWAY_AMP = 0.0008;
const float KICK_SHIFT = 0.012;
const float SAT_KNEE_LO = 0.40;
const float SAT_KNEE_HI = 0.70;
const float WARP_SCALE = 0.025;
const float WARP_BASE  = 0.30;

float satOf(vec3 rgb) {
  float maxC = max(max(rgb.r, rgb.g), rgb.b);
  float minC = min(min(rgb.r, rgb.g), rgb.b);
  return maxC > 0.001 ? (maxC - minC) / maxC : 0.0;
}

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
  float warpAmt = (WARP_BASE + (1.0 - WARP_BASE) * uKick) * uWarpStrength;
  vec2 uv = vUv + organicWarp(vUv, uTime) * WARP_SCALE * warpAmt;

  float sway = sin(uTime * 6.2831853 * SWAY_HZ) * SWAY_AMP;
  float kick = uKick * KICK_SHIFT;
  vec2 shift = vec2(sway + kick, 0.0) * uParallaxStrength;

  vec3 srcLocal = texture(uSource, uv).rgb;
  vec3 srcShifted = texture(uSource, uv - shift).rgb;

  float satShifted = satOf(srcShifted);
  float mask = smoothstep(SAT_KNEE_LO, SAT_KNEE_HI, satShifted);
  float satLocal = satOf(srcLocal);
  mask *= 1.0 - smoothstep(0.6, 1.0, satLocal) * 0.4;

  vec3 col = mix(srcLocal, srcShifted, mask);
  fragColor = vec4(col, 1.0);
}`;

const BRIGHT_FRAG = /* glsl */ `#version 300 es
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

const BLUR_FRAG = /* glsl */ `#version 300 es
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

const COMPOSITE_FRAG = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uKick;
uniform float uBloomOnKick;
uniform float uBloomBase;
uniform vec3  uBloomTint;
uniform float uChroma;
uniform float uChromaKick;
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
// Phase 6 perf: 18 → 12. Visually nearly identical at typical bloom amounts;
// halves Gaussian sample fan-out cost across both blur passes.
const BLOOM_BLUR_RADIUS_PX = 12;

interface RenderTarget {
  tex: WebGLTexture;
  fbo: WebGLFramebuffer;
  w: number;
  h: number;
}

function compileShader(
  gl: WebGL2RenderingContext,
  type: number,
  src: string,
): WebGLShader {
  const sh = gl.createShader(type);
  if (!sh) throw new Error("createShader returned null");
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`Shader compile failed: ${log}`);
  }
  return sh;
}

function makeProgram(
  gl: WebGL2RenderingContext,
  fragSrc: string,
): WebGLProgram {
  const prog = gl.createProgram();
  if (!prog) throw new Error("createProgram returned null");
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

function makeRT(
  gl: WebGL2RenderingContext,
  w: number,
  h: number,
): RenderTarget {
  const tex = gl.createTexture();
  if (!tex) throw new Error("createTexture returned null");
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    w,
    h,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    null,
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const fbo = gl.createFramebuffer();
  if (!fbo) throw new Error("createFramebuffer returned null");
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(
    gl.FRAMEBUFFER,
    gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D,
    tex,
    0,
  );
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { tex, fbo, w, h };
}

function disposeRT(
  gl: WebGL2RenderingContext,
  rt: RenderTarget | null,
): void {
  if (!rt) return;
  gl.deleteTexture(rt.tex);
  gl.deleteFramebuffer(rt.fbo);
}

interface UniformLocations {
  parallax: {
    uSource: WebGLUniformLocation | null;
    uTime: WebGLUniformLocation | null;
    uKick: WebGLUniformLocation | null;
    uParallaxStrength: WebGLUniformLocation | null;
    uWarpStrength: WebGLUniformLocation | null;
  };
  bright: {
    uScene: WebGLUniformLocation | null;
    uThreshold: WebGLUniformLocation | null;
    uKnee: WebGLUniformLocation | null;
  };
  blur: {
    uTex: WebGLUniformLocation | null;
    uTexel: WebGLUniformLocation | null;
    uDir: WebGLUniformLocation | null;
    uRadius: WebGLUniformLocation | null;
  };
  composite: {
    uScene: WebGLUniformLocation | null;
    uBloom: WebGLUniformLocation | null;
    uKick: WebGLUniformLocation | null;
    uBloomOnKick: WebGLUniformLocation | null;
    uBloomBase: WebGLUniformLocation | null;
    uBloomTint: WebGLUniformLocation | null;
    uChroma: WebGLUniformLocation | null;
    uChromaKick: WebGLUniformLocation | null;
  };
}

export class EffectsRenderer {
  readonly canvas: HTMLCanvasElement;
  readonly gl: WebGL2RenderingContext;

  private readonly _vao: WebGLVertexArrayObject;
  private readonly _buf: WebGLBuffer;

  private readonly _parallax: WebGLProgram;
  private readonly _bright: WebGLProgram;
  private readonly _blur: WebGLProgram;
  private readonly _composite: WebGLProgram;
  private readonly _u: UniformLocations;

  private readonly _srcTex: WebGLTexture;
  private _srcW = 0;
  private _srcH = 0;

  private _scene: RenderTarget | null = null;
  private _bloom0: RenderTarget | null = null;
  private _bloom1: RenderTarget | null = null;

  private _parallaxStrength = 0.4;
  private _bloomOnKick = 0.3;
  private _bloomThreshold = 0.15;
  private _bloomKnee = 0.2;
  private _warpStrength = 0.4;
  private _dubstep = 0;
  private _daft = 0;
  private _everDrew = false;

  private readonly _resizeObs: ResizeObserver;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", {
      alpha: true,
      antialias: false,
      preserveDrawingBuffer: false,
      powerPreference: "high-performance",
    });
    if (!gl) throw new Error("WebGL2 not supported");
    this.gl = gl;

    const buf = gl.createBuffer();
    if (!buf) throw new Error("createBuffer returned null");
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 3, -1, -1, 3]),
      gl.STATIC_DRAW,
    );
    const vao = gl.createVertexArray();
    if (!vao) throw new Error("createVertexArray returned null");
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

    const tex = gl.createTexture();
    if (!tex) throw new Error("createTexture returned null");
    this._srcTex = tex;
    gl.bindTexture(gl.TEXTURE_2D, this._srcTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this._resizeObs = new ResizeObserver(() => this._resize());
    this._resizeObs.observe(canvas);
    this._resize();
  }

  setParallaxStrength(v: number): void { this._parallaxStrength = v; }
  setBloomOnKick(v: number): void { this._bloomOnKick = v; }
  setBloomThreshold(v: number): void { this._bloomThreshold = v; }
  setWarpStrength(v: number): void { this._warpStrength = v; }
  setDubstep(v: number): void { this._dubstep = Math.max(0, Math.min(1, v)); }
  setDaftPunk(v: number): void { this._daft = Math.max(0, Math.min(1, v)); }

  private _resize(): void {
    const gl = this.gl;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(2, Math.floor(rect.width * dpr));
    const h = Math.max(2, Math.floor(rect.height * dpr));
    if (this.canvas.width === w && this.canvas.height === h && this._scene) {
      return;
    }
    this.canvas.width = w;
    this.canvas.height = h;

    disposeRT(gl, this._scene);
    this._scene = makeRT(gl, w, h);

    const bw = Math.max(2, Math.round(w * BLOOM_DOWNSCALE));
    const bh = Math.max(2, Math.round(h * BLOOM_DOWNSCALE));
    disposeRT(gl, this._bloom0);
    disposeRT(gl, this._bloom1);
    this._bloom0 = makeRT(gl, bw, bh);
    this._bloom1 = makeRT(gl, bw, bh);
  }

  private _uploadVideo(videoEl: HTMLVideoElement | null): boolean {
    if (!videoEl) return this._srcW > 0;
    const vw = videoEl.videoWidth | 0;
    const vh = videoEl.videoHeight | 0;
    const ready = videoEl.readyState >= 2 && vw > 0 && vh > 0;
    if (!ready) return this._srcW > 0;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this._srcTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    if (this._srcW !== vw || this._srcH !== vh) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, videoEl);
      this._srcW = vw;
      this._srcH = vh;
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, videoEl);
    }
    return true;
  }

  private _drawTo(target: RenderTarget | null): void {
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

  tick(
    videoEl: HTMLVideoElement | null,
    timeSeconds: number,
    kick: number,
  ): void {
    const gl = this.gl;
    if (!this._uploadVideo(videoEl)) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      return;
    }

    if (!this._scene || !this._bloom0 || !this._bloom1) return;

    // 1. Parallax: video -> scene
    gl.useProgram(this._parallax);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._srcTex);
    gl.uniform1i(this._u.parallax.uSource, 0);
    gl.uniform1f(this._u.parallax.uTime, timeSeconds);
    gl.uniform1f(this._u.parallax.uKick, kick);
    gl.uniform1f(this._u.parallax.uParallaxStrength, this._parallaxStrength);
    gl.uniform1f(this._u.parallax.uWarpStrength, this._warpStrength);
    this._drawTo(this._scene);

    // 2. Brightpass: scene -> bloom0.
    gl.useProgram(this._bright);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._scene.tex);
    gl.uniform1i(this._u.bright.uScene, 0);
    const effThreshold = Math.max(0.02, this._bloomThreshold - this._daft * 0.1);
    gl.uniform1f(this._u.bright.uThreshold, effThreshold);
    gl.uniform1f(this._u.bright.uKnee, this._bloomKnee);
    this._drawTo(this._bloom0);

    // 3+4. Two-pass Gaussian blur.
    gl.useProgram(this._blur);
    gl.uniform2f(this._u.blur.uTexel, 1 / this._bloom0.w, 1 / this._bloom0.h);
    gl.uniform1f(this._u.blur.uRadius, BLOOM_BLUR_RADIUS_PX);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._bloom0.tex);
    gl.uniform1i(this._u.blur.uTex, 0);
    gl.uniform2f(this._u.blur.uDir, 1, 0);
    this._drawTo(this._bloom1);

    gl.bindTexture(gl.TEXTURE_2D, this._bloom1.tex);
    gl.uniform2f(this._u.blur.uDir, 0, 1);
    this._drawTo(this._bloom0);

    // 5. Composite.
    gl.useProgram(this._composite);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._scene.tex);
    gl.uniform1i(this._u.composite.uScene, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this._bloom0.tex);
    gl.uniform1i(this._u.composite.uBloom, 1);
    gl.uniform1f(this._u.composite.uKick, kick);
    gl.uniform1f(this._u.composite.uBloomOnKick, this._bloomOnKick);

    const daft = this._daft;
    gl.uniform1f(this._u.composite.uBloomBase, daft * 0.3);
    const tintMix = daft * 0.5;
    gl.uniform3f(
      this._u.composite.uBloomTint,
      1.0,
      1.0 + (0.78 - 1.0) * tintMix,
      1.0 + (0.82 - 1.0) * tintMix,
    );

    const dub = this._dubstep;
    gl.uniform1f(this._u.composite.uChroma, dub * 0.005);
    gl.uniform1f(this._u.composite.uChromaKick, dub * 0.012);

    this._drawTo(null);

    if (!this._everDrew) {
      this._everDrew = true;
      this.canvas.classList.add("effects-ready");
    }
  }

  destroy(): void {
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
