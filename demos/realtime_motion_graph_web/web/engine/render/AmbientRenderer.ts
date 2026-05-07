// @ts-nocheck
// ---------------------------------------------------------------------------
// AmbientRenderer.ts -- WebGL2 GPU fluid simulation for ambient light effects.
//
// Direct port of static/ambient.js. The implementation is shader-heavy and
// already verified against the existing visual; @ts-nocheck above keeps the
// TS compiler off our back rather than annotating ~1100 lines of GL state.
//
// Phase 7 perf tweaks vs. the original:
//   SIM_RESOLUTION      128 -> 96    (fluid grid; ~40% cheaper)
//   DYE_RESOLUTION      512 -> 320   (dye field; cheaper texture binds)
//   PRESSURE_ITERATIONS 20 -> 12     (Jacobi steps; visually nearly identical)
// Run loop drops to 30 FPS via TARGET_INTERVAL_MS.
//
// Public API (unchanged):
//   new AmbientRenderer(canvas)
//   attach(analyser, getVideoEl)
//   start() / stop() / resize() / destroy()

/* ======================================================================== */
/*  TUNING CONSTANTS                                                        */
/* ======================================================================== */

const SIM_RESOLUTION       = 96;     // perf: 128 -> 96
const DYE_RESOLUTION       = 320;    // perf: 512 -> 320
const PRESSURE_ITERATIONS  = 12;     // perf: 20 -> 12
const VELOCITY_DISSIPATION = 0.98;
const DYE_DISSIPATION      = 0.985;
const CURL_STRENGTH        = 20;
const SPLAT_RADIUS         = 0.012;
const FORCE_SCALE          = 6000;
const AUDIO_SMOOTH         = 0.18;
const VIDEO_SAMPLE_SIZE    = 16;
const VIDEO_SMOOTH         = 0.12;
const TARGET_INTERVAL_MS   = 1000 / 30; // perf: cap at 30 FPS

/* ======================================================================== */
/*  GLSL SHADER SOURCES                                                     */
/* ======================================================================== */

const VERT_FULLSCREEN = `#version 300 es
precision highp float;
out vec2 vUv;
void main() {
    // Fullscreen triangle: vertices (-1,-1), (3,-1), (-1,3)
    float x = -1.0 + float((gl_VertexID & 1) << 2);
    float y = -1.0 + float((gl_VertexID & 2) << 1);
    gl_Position = vec4(x, y, 0.0, 1.0);
    vUv = gl_Position.xy * 0.5 + 0.5;
}
`;

const FRAG_ADVECTION = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 uTexelSize;
uniform float uDt;
uniform float uDissipation;
in vec2 vUv;
out vec4 fragColor;
void main() {
    vec2 coord = vUv - uDt * texture(uVelocity, vUv).xy * uTexelSize;
    fragColor = uDissipation * texture(uSource, coord);
}
`;

const FRAG_DIVERGENCE = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uVelocity;
uniform vec2 uTexelSize;
in vec2 vUv;
out vec4 fragColor;
void main() {
    float L = texture(uVelocity, vUv - vec2(uTexelSize.x, 0.0)).x;
    float R = texture(uVelocity, vUv + vec2(uTexelSize.x, 0.0)).x;
    float B = texture(uVelocity, vUv - vec2(0.0, uTexelSize.y)).y;
    float T = texture(uVelocity, vUv + vec2(0.0, uTexelSize.y)).y;
    float div = 0.5 * (R - L + T - B);
    fragColor = vec4(div, 0.0, 0.0, 1.0);
}
`;

const FRAG_PRESSURE = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
uniform vec2 uTexelSize;
in vec2 vUv;
out vec4 fragColor;
void main() {
    float pL = texture(uPressure, vUv - vec2(uTexelSize.x, 0.0)).x;
    float pR = texture(uPressure, vUv + vec2(uTexelSize.x, 0.0)).x;
    float pB = texture(uPressure, vUv - vec2(0.0, uTexelSize.y)).x;
    float pT = texture(uPressure, vUv + vec2(0.0, uTexelSize.y)).x;
    float div = texture(uDivergence, vUv).x;
    fragColor = vec4((pL + pR + pB + pT - div) * 0.25, 0.0, 0.0, 1.0);
}
`;

const FRAG_GRADIENT_SUB = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
uniform vec2 uTexelSize;
in vec2 vUv;
out vec4 fragColor;
void main() {
    float pL = texture(uPressure, vUv - vec2(uTexelSize.x, 0.0)).x;
    float pR = texture(uPressure, vUv + vec2(uTexelSize.x, 0.0)).x;
    float pB = texture(uPressure, vUv - vec2(0.0, uTexelSize.y)).x;
    float pT = texture(uPressure, vUv + vec2(0.0, uTexelSize.y)).x;
    vec2 vel = texture(uVelocity, vUv).xy;
    vel -= 0.5 * vec2(pR - pL, pT - pB);
    fragColor = vec4(vel, 0.0, 1.0);
}
`;

const FRAG_CURL = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uVelocity;
uniform vec2 uTexelSize;
in vec2 vUv;
out vec4 fragColor;
void main() {
    float L = texture(uVelocity, vUv - vec2(uTexelSize.x, 0.0)).y;
    float R = texture(uVelocity, vUv + vec2(uTexelSize.x, 0.0)).y;
    float B = texture(uVelocity, vUv - vec2(0.0, uTexelSize.y)).x;
    float T = texture(uVelocity, vUv + vec2(0.0, uTexelSize.y)).x;
    float curl = R - L - T + B;
    fragColor = vec4(0.5 * curl, 0.0, 0.0, 1.0);
}
`;

const FRAG_VORTICITY = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform vec2 uTexelSize;
uniform float uCurlStrength;
uniform float uDt;
in vec2 vUv;
out vec4 fragColor;
void main() {
    float cL = texture(uCurl, vUv - vec2(uTexelSize.x, 0.0)).x;
    float cR = texture(uCurl, vUv + vec2(uTexelSize.x, 0.0)).x;
    float cB = texture(uCurl, vUv - vec2(0.0, uTexelSize.y)).x;
    float cT = texture(uCurl, vUv + vec2(0.0, uTexelSize.y)).x;
    float cC = texture(uCurl, vUv).x;

    vec2 force = 0.5 * vec2(abs(cT) - abs(cB), abs(cR) - abs(cL));
    float len = length(force) + 1e-5;
    force = force / len * uCurlStrength * cC;

    vec2 vel = texture(uVelocity, vUv).xy + force * uDt;
    fragColor = vec4(vel, 0.0, 1.0);
}
`;

const FRAG_SPLAT = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uSource;
uniform vec2 uSplatPos;
uniform vec3 uSplatColor;
uniform float uSplatRadius;
uniform float uAspect;
in vec2 vUv;
out vec4 fragColor;
void main() {
    vec2 d = vUv - uSplatPos;
    d.x *= uAspect;
    float w = exp(-dot(d, d) / uSplatRadius);
    vec3 base = texture(uSource, vUv).rgb;
    fragColor = vec4(base + uSplatColor * w, 1.0);
}
`;

const FRAG_CLEAR_PRESSURE = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uPressure;
uniform float uValue;
in vec2 vUv;
out vec4 fragColor;
void main() {
    fragColor = vec4(uValue * texture(uPressure, vUv).x, 0.0, 0.0, 1.0);
}
`;

const FRAG_RENDER = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uDye;
in vec2 vUv;
out vec4 fragColor;
void main() {
    vec3 dye = texture(uDye, vUv).rgb;
    // Tone-map to prevent harsh clipping
    dye = dye / (1.0 + dye);
    // Alpha from brightness so canvas is transparent where there's no dye
    float a = max(dye.r, max(dye.g, dye.b));
    fragColor = vec4(dye, a);
}
`;

/* ======================================================================== */
/*  WEBGL UTILITIES                                                         */
/* ======================================================================== */

function compileShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(shader);
        gl.deleteShader(shader);
        throw new Error("Shader compilation failed: " + info);
    }
    return shader;
}

function createProgram(gl, vertSrc, fragSrc) {
    const vs = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        const info = gl.getProgramInfoLog(prog);
        gl.deleteProgram(prog);
        throw new Error("Program link failed: " + info);
    }
    // Cache uniform locations
    const uniforms = {};
    const count = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < count; i++) {
        const info = gl.getActiveUniform(prog, i);
        uniforms[info.name] = gl.getUniformLocation(prog, info.name);
    }
    return { program: prog, uniforms };
}

/**
 * Create a double-buffered framebuffer object pair for ping-pong rendering.
 * Returns { read, write, swap(), width, height }.
 * Each side has { fbo, texture }.
 */
function createDoubleFBO(gl, width, height, internalFormat, format, type, filter) {
    const a = createFBO(gl, width, height, internalFormat, format, type, filter);
    const b = createFBO(gl, width, height, internalFormat, format, type, filter);
    return {
        width,
        height,
        read: a,
        write: b,
        swap() { const t = this.read; this.read = this.write; this.write = t; },
    };
}

function createFBO(gl, width, height, internalFormat, format, type, filter) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, null);

    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.viewport(0, 0, width, height);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error("Framebuffer incomplete: " + status);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { fbo, texture, width, height };
}

function destroyFBO(gl, fboObj) {
    if (!fboObj) return;
    if (fboObj.texture) gl.deleteTexture(fboObj.texture);
    if (fboObj.fbo) gl.deleteFramebuffer(fboObj.fbo);
}

function destroyDoubleFBO(gl, dFBO) {
    if (!dFBO) return;
    destroyFBO(gl, dFBO.read);
    destroyFBO(gl, dFBO.write);
}

/* ======================================================================== */
/*  FLUID SIMULATION (LOW-LEVEL GPU SOLVER)                                 */
/* ======================================================================== */

class FluidSim {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = null;
        this.programs = {};
        this.velocity = null;
        this.pressure = null;
        this.divergenceFBO = null;
        this.curlFBO = null;
        this.dye = null;
        this.simWidth = 0;
        this.simHeight = 0;
        this.dyeWidth = 0;
        this.dyeHeight = 0;
        this._videoTexture = null;
        this._lost = false;

        this._initGL();
    }

    _initGL() {
        const gl = this.canvas.getContext("webgl2", {
            alpha: true,
            depth: false,
            stencil: false,
            antialias: false,
            preserveDrawingBuffer: false,
            premultipliedAlpha: false,
        });
        if (!gl) throw new Error("WebGL2 not available");
        this.gl = gl;

        // Handle context loss/restore
        this.canvas.addEventListener("webglcontextlost", (e) => {
            e.preventDefault();
            this._lost = true;
        });
        this.canvas.addEventListener("webglcontextrestored", () => {
            this._lost = false;
            this._initGL();
            this._initFramebuffers();
        });

        // Required extension for float render targets
        const extFloat = gl.getExtension("EXT_color_buffer_float");
        if (!extFloat) {
            // Try half-float fallback
            gl.getExtension("EXT_color_buffer_half_float");
        }
        gl.getExtension("OES_texture_float_linear");
        gl.getExtension("OES_texture_half_float_linear");

        // Determine best supported float format
        this._floatInternal = gl.RGBA16F;
        this._floatType = gl.HALF_FLOAT;
        this._rInternal = gl.R16F;
        this._rgInternal = gl.RG16F;

        // Test if float rendering works, otherwise try half-float
        if (!this._testRenderTarget(gl, gl.RG16F, gl.RG, gl.HALF_FLOAT)) {
            if (!this._testRenderTarget(gl, gl.RG32F, gl.RG, gl.FLOAT)) {
                throw new Error("No float render target support");
            }
            this._floatInternal = gl.RGBA32F;
            this._floatType = gl.FLOAT;
            this._rInternal = gl.R32F;
            this._rgInternal = gl.RG32F;
        }

        // Compile all shader programs
        this.programs.advection   = createProgram(gl, VERT_FULLSCREEN, FRAG_ADVECTION);
        this.programs.divergence  = createProgram(gl, VERT_FULLSCREEN, FRAG_DIVERGENCE);
        this.programs.pressure    = createProgram(gl, VERT_FULLSCREEN, FRAG_PRESSURE);
        this.programs.gradientSub = createProgram(gl, VERT_FULLSCREEN, FRAG_GRADIENT_SUB);
        this.programs.curl        = createProgram(gl, VERT_FULLSCREEN, FRAG_CURL);
        this.programs.vorticity   = createProgram(gl, VERT_FULLSCREEN, FRAG_VORTICITY);
        this.programs.splat       = createProgram(gl, VERT_FULLSCREEN, FRAG_SPLAT);
        this.programs.clearPressure = createProgram(gl, VERT_FULLSCREEN, FRAG_CLEAR_PRESSURE);
        this.programs.render      = createProgram(gl, VERT_FULLSCREEN, FRAG_RENDER);

        // Create a dummy VAO (required by WebGL2 for draw calls)
        this._vao = gl.createVertexArray();

        // Create reusable video upload texture
        this._videoTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this._videoTexture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        // Clear to transparent so the canvas doesn't show as white/opaque
        // before the simulation starts.
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
    }

    _testRenderTarget(gl, internalFormat, format, type) {
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 4, 4, 0, format, type, null);

        const fbo = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
        const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.deleteFramebuffer(fbo);
        gl.deleteTexture(tex);
        return ok;
    }

    _initFramebuffers() {
        const gl = this.gl;
        if (!gl) return;

        // Clean up any existing FBOs
        destroyDoubleFBO(gl, this.velocity);
        destroyDoubleFBO(gl, this.pressure);
        destroyDoubleFBO(gl, this.dye);
        destroyFBO(gl, this.divergenceFBO);
        destroyFBO(gl, this.curlFBO);

        const aspect = this.canvas.width / this.canvas.height;

        // Simulation fields at SIM_RESOLUTION
        this.simWidth  = Math.max(1, Math.round(SIM_RESOLUTION * aspect));
        this.simHeight = SIM_RESOLUTION;

        // Dye field at DYE_RESOLUTION
        this.dyeWidth  = Math.max(1, Math.round(DYE_RESOLUTION * aspect));
        this.dyeHeight = DYE_RESOLUTION;

        const sW = this.simWidth;
        const sH = this.simHeight;
        const dW = this.dyeWidth;
        const dH = this.dyeHeight;

        const lin = gl.LINEAR;
        const near = gl.NEAREST;
        const hf = this._floatType;

        this.velocity     = createDoubleFBO(gl, sW, sH, this._rgInternal, gl.RG, hf, lin);
        this.pressure     = createDoubleFBO(gl, sW, sH, this._rInternal, gl.RED, hf, near);
        this.divergenceFBO = createFBO(gl, sW, sH, this._rInternal, gl.RED, hf, near);
        this.curlFBO      = createFBO(gl, sW, sH, this._rInternal, gl.RED, hf, near);
        this.dye          = createDoubleFBO(gl, dW, dH, this._floatInternal, gl.RGBA, hf, lin);
    }

    resize(w, h) {
        if (w <= 0 || h <= 0) return;
        if (w === this.canvas.width && h === this.canvas.height && this.dye) return;
        this.canvas.width = w;
        this.canvas.height = h;
        this._initFramebuffers();
    }

    /**
     * Upload the current video frame as a texture for edge sampling on the CPU.
     * Returns { pixels, width, height } or null if not available.
     */
    sampleVideoEdges(videoEl, sampleCanvas, sampleCtx, sampleSize) {
        if (!videoEl || videoEl.paused || !videoEl.videoWidth) return null;
        try {
            sampleCtx.drawImage(videoEl, 0, 0, sampleSize, sampleSize);
        } catch {
            return null;
        }
        return sampleCtx.getImageData(0, 0, sampleSize, sampleSize);
    }

    /**
     * Run one full simulation step.
     * splats: Array of { x, y, dx, dy, r, g, b, radius }
     *   x,y in [0,1] UV space, dx/dy are force components,
     *   r/g/b are dye color, radius is Gaussian radius.
     */
    step(dt, splats) {
        const gl = this.gl;
        if (!gl || this._lost || !this.dye) return;

        gl.bindVertexArray(this._vao);
        gl.disable(gl.BLEND);

        // 1. Advect velocity
        this._advect(this.velocity, this.velocity, dt, VELOCITY_DISSIPATION);

        // 2. Inject forces via splats on velocity field
        for (const s of splats) {
            this._splat(
                this.velocity,
                s.x, s.y,
                s.dx, s.dy, 0.0,
                s.radius,
                this.simWidth / this.simHeight
            );
        }

        // 3. Vorticity confinement
        this._computeCurl();
        this._applyVorticity(dt);

        // 4. Divergence
        this._computeDivergence();

        // 5. Clear pressure (slight damping to help convergence)
        this._clearPressure(0.8);

        // 6. Pressure solve (Jacobi iterations)
        for (let i = 0; i < PRESSURE_ITERATIONS; i++) {
            this._pressureIteration();
        }

        // 7. Gradient subtraction (project velocity to be divergence-free)
        this._gradientSubtract();

        // 8. Advect dye
        this._advectDye(dt);

        // 9. Inject dye via splats
        for (const s of splats) {
            this._splat(
                this.dye,
                s.x, s.y,
                s.r, s.g, s.b,
                s.radius * 1.5, // dye radius slightly larger than force radius
                this.dyeWidth / this.dyeHeight
            );
        }
    }

    /**
     * Render the dye field to the screen.
     */
    render() {
        const gl = this.gl;
        if (!gl || this._lost || !this.dye) return;

        gl.bindVertexArray(this._vao);

        const prog = this.programs.render;
        gl.useProgram(prog.program);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.dye.read.texture);
        gl.uniform1i(prog.uniforms.uDye, 0);

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    // -- internal passes ---------------------------------------------------

    _advect(target, velocitySource, dt, dissipation) {
        const gl = this.gl;
        const prog = this.programs.advection;
        gl.useProgram(prog.program);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, velocitySource.read.texture);
        gl.uniform1i(prog.uniforms.uVelocity, 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, target.read.texture);
        gl.uniform1i(prog.uniforms.uSource, 1);

        gl.uniform2f(prog.uniforms.uTexelSize, 1.0 / target.width, 1.0 / target.height);
        gl.uniform1f(prog.uniforms.uDt, dt);
        gl.uniform1f(prog.uniforms.uDissipation, dissipation);

        this._drawToFBO(target.write, target.width, target.height);
        target.swap();
    }

    _advectDye(dt) {
        const gl = this.gl;
        const prog = this.programs.advection;
        gl.useProgram(prog.program);

        // Velocity is at sim resolution, dye at dye resolution.
        // We sample velocity using the dye UVs (linear filtering handles the mismatch).
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
        gl.uniform1i(prog.uniforms.uVelocity, 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.dye.read.texture);
        gl.uniform1i(prog.uniforms.uSource, 1);

        gl.uniform2f(prog.uniforms.uTexelSize, 1.0 / this.dyeWidth, 1.0 / this.dyeHeight);
        gl.uniform1f(prog.uniforms.uDt, dt);
        gl.uniform1f(prog.uniforms.uDissipation, DYE_DISSIPATION);

        this._drawToFBO(this.dye.write, this.dyeWidth, this.dyeHeight);
        this.dye.swap();
    }

    _computeDivergence() {
        const gl = this.gl;
        const prog = this.programs.divergence;
        gl.useProgram(prog.program);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
        gl.uniform1i(prog.uniforms.uVelocity, 0);

        gl.uniform2f(prog.uniforms.uTexelSize, 1.0 / this.simWidth, 1.0 / this.simHeight);

        this._drawToFBO(this.divergenceFBO, this.simWidth, this.simHeight);
    }

    _clearPressure(value) {
        const gl = this.gl;
        const prog = this.programs.clearPressure;
        gl.useProgram(prog.program);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.pressure.read.texture);
        gl.uniform1i(prog.uniforms.uPressure, 0);
        gl.uniform1f(prog.uniforms.uValue, value);

        this._drawToFBO(this.pressure.write, this.simWidth, this.simHeight);
        this.pressure.swap();
    }

    _pressureIteration() {
        const gl = this.gl;
        const prog = this.programs.pressure;
        gl.useProgram(prog.program);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.pressure.read.texture);
        gl.uniform1i(prog.uniforms.uPressure, 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.divergenceFBO.texture);
        gl.uniform1i(prog.uniforms.uDivergence, 1);

        gl.uniform2f(prog.uniforms.uTexelSize, 1.0 / this.simWidth, 1.0 / this.simHeight);

        this._drawToFBO(this.pressure.write, this.simWidth, this.simHeight);
        this.pressure.swap();
    }

    _gradientSubtract() {
        const gl = this.gl;
        const prog = this.programs.gradientSub;
        gl.useProgram(prog.program);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.pressure.read.texture);
        gl.uniform1i(prog.uniforms.uPressure, 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
        gl.uniform1i(prog.uniforms.uVelocity, 1);

        gl.uniform2f(prog.uniforms.uTexelSize, 1.0 / this.simWidth, 1.0 / this.simHeight);

        this._drawToFBO(this.velocity.write, this.simWidth, this.simHeight);
        this.velocity.swap();
    }

    _computeCurl() {
        const gl = this.gl;
        const prog = this.programs.curl;
        gl.useProgram(prog.program);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
        gl.uniform1i(prog.uniforms.uVelocity, 0);

        gl.uniform2f(prog.uniforms.uTexelSize, 1.0 / this.simWidth, 1.0 / this.simHeight);

        this._drawToFBO(this.curlFBO, this.simWidth, this.simHeight);
    }

    _applyVorticity(dt) {
        const gl = this.gl;
        const prog = this.programs.vorticity;
        gl.useProgram(prog.program);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
        gl.uniform1i(prog.uniforms.uVelocity, 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.curlFBO.texture);
        gl.uniform1i(prog.uniforms.uCurl, 1);

        gl.uniform2f(prog.uniforms.uTexelSize, 1.0 / this.simWidth, 1.0 / this.simHeight);
        gl.uniform1f(prog.uniforms.uCurlStrength, CURL_STRENGTH);
        gl.uniform1f(prog.uniforms.uDt, dt);

        this._drawToFBO(this.velocity.write, this.simWidth, this.simHeight);
        this.velocity.swap();
    }

    /**
     * Add a Gaussian splat to a double-buffered field.
     * color is (r, g, b) -- for velocity fields, only (r, g) are used.
     */
    _splat(target, x, y, cr, cg, cb, radius, aspect) {
        const gl = this.gl;
        const prog = this.programs.splat;
        gl.useProgram(prog.program);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, target.read.texture);
        gl.uniform1i(prog.uniforms.uSource, 0);

        gl.uniform2f(prog.uniforms.uSplatPos, x, y);
        gl.uniform3f(prog.uniforms.uSplatColor, cr, cg, cb);
        gl.uniform1f(prog.uniforms.uSplatRadius, radius);
        gl.uniform1f(prog.uniforms.uAspect, aspect);

        this._drawToFBO(target.write, target.width, target.height);
        target.swap();
    }

    _drawToFBO(fbo, w, h) {
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.fbo);
        gl.viewport(0, 0, w, h);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    destroy() {
        const gl = this.gl;
        if (!gl) return;

        destroyDoubleFBO(gl, this.velocity);
        destroyDoubleFBO(gl, this.pressure);
        destroyDoubleFBO(gl, this.dye);
        destroyFBO(gl, this.divergenceFBO);
        destroyFBO(gl, this.curlFBO);

        if (this._videoTexture) gl.deleteTexture(this._videoTexture);
        if (this._vao) gl.deleteVertexArray(this._vao);

        for (const key in this.programs) {
            gl.deleteProgram(this.programs[key].program);
        }

        this.gl = null;
    }
}

/* ======================================================================== */
/*  AMBIENT LIGHT (PUBLIC API)                                              */
/* ======================================================================== */

export class AmbientLight {
    constructor(canvas) {
        this.canvas = canvas;
        this._analyser = null;
        this._freqData = null;
        this._getVideoEl = null;
        this._running = false;
        this._raf = null;
        this._lastTime = 0;
        this._sim = null;
        this._webgl2 = false;

        // Audio band accumulators (smoothed, 0-1)
        this._bands = { sub: 0, bass: 0, mid: 0, high: 0 };

        // Edge color accumulators (smoothed, 0-1)
        this._edges = {
            left:   [0, 0, 0],
            right:  [0, 0, 0],
            top:    [0, 0, 0],
            bottom: [0, 0, 0],
        };

        // Precompute edge pixel indices for the sampling grid
        const S = VIDEO_SAMPLE_SIZE;
        this._sampleSize = S;
        this._sampCvs = document.createElement("canvas");
        this._sampCvs.width = S;
        this._sampCvs.height = S;
        this._sampCtx = this._sampCvs.getContext("2d", { willReadFrequently: true });

        this._edgeIdx = {};
        for (const [side, fn] of [
            ["left",   (i) => [0, i, 1, i]],
            ["right",  (i) => [S - 1, i, S - 2, i]],
            ["top",    (i) => [i, 0, i, 1]],
            ["bottom", (i) => [i, S - 1, i, S - 2]],
        ]) {
            const flat = [];
            for (let i = 0; i < S; i++) flat.push(...fn(i));
            this._edgeIdx[side] = flat;
        }

        // Try to initialise WebGL2 fluid sim
        try {
            this._sim = new FluidSim(canvas);
            this._webgl2 = true;
        } catch (e) {
            console.warn("AmbientLight: WebGL2 fluid sim unavailable, running as no-op.", e);
            this._webgl2 = false;
        }

        this._onResize = () => this.resize();
        window.addEventListener("resize", this._onResize);
    }

    attach(analyser, getVideoEl) {
        this._analyser = analyser;
        this._freqData = new Uint8Array(analyser.frequencyBinCount);
        this._getVideoEl = getVideoEl;
    }

    start() {
        this._running = true;
        this.resize();
        this._lastTime = performance.now();
        this._tick();
    }

    stop() {
        this._running = false;
        if (this._raf != null) {
            cancelAnimationFrame(this._raf);
            this._raf = null;
        }
        // Clear the canvas
        if (this._sim && this._sim.gl) {
            const gl = this._sim.gl;
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.viewport(0, 0, this.canvas.width, this.canvas.height);
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);
        }
    }

    resize() {
        const w = this.canvas.clientWidth;
        const h = this.canvas.clientHeight;
        if (!w || !h) return;
        const dpr = Math.min(devicePixelRatio || 1, 2);
        const pw = Math.floor(w * dpr);
        const ph = Math.floor(h * dpr);
        if (this._sim && this._webgl2) {
            this._sim.resize(pw, ph);
        }
    }

    destroy() {
        this.stop();
        window.removeEventListener("resize", this._onResize);
        if (this._sim) {
            this._sim.destroy();
            this._sim = null;
        }
    }

    // -- render loop -------------------------------------------------------

    _tick() {
        if (!this._running) return;

        const now = performance.now();
        // Phase 7 perf: target 30 FPS. RAF still runs at the display rate
        // but we skip the simulation step until enough time has elapsed.
        const sinceLast = now - this._lastTime;
        if (sinceLast < TARGET_INTERVAL_MS) {
            this._raf = requestAnimationFrame(() => this._tick());
            return;
        }
        // Clamp dt to avoid huge jumps (e.g. tab was backgrounded)
        const dt = Math.min(sinceLast / 1000, 0.05);
        this._lastTime = now;

        if (this._webgl2 && this._sim && !this._sim._lost) {
            this._readAudio();
            this._readVideo();
            const splats = this._buildSplats();
            this._sim.step(dt, splats);
            this._sim.render();
        }

        this._raf = requestAnimationFrame(() => this._tick());
    }

    // -- audio frequency bands ---------------------------------------------

    _readAudio() {
        if (!this._analyser) return;
        this._analyser.getByteFrequencyData(this._freqData);
        const d = this._freqData;
        const n = d.length;

        const avg = (lo, hi) => {
            let s = 0;
            const end = Math.min(hi, n);
            for (let i = lo; i < end; i++) s += d[i];
            return end > lo ? s / (end - lo) / 255 : 0;
        };

        const t = AUDIO_SMOOTH;
        this._bands.sub  += (avg(0,   3)   - this._bands.sub)  * t;
        this._bands.bass += (avg(3,   11)  - this._bands.bass) * t;
        this._bands.mid  += (avg(11,  86)  - this._bands.mid)  * t;
        this._bands.high += (avg(86, 342)  - this._bands.high) * t;
    }

    // -- video edge color sampling -----------------------------------------

    _readVideo() {
        const el = this._getVideoEl?.();
        if (!el || el.paused || !el.videoWidth) return;

        const S = this._sampleSize;
        try {
            this._sampCtx.drawImage(el, 0, 0, S, S);
        } catch {
            return;
        }
        const px = this._sampCtx.getImageData(0, 0, S, S).data;
        const ct = VIDEO_SMOOTH;

        for (const side of ["left", "right", "top", "bottom"]) {
            const idx = this._edgeIdx[side];
            const count = idx.length / 2;
            let r = 0, g = 0, b = 0;
            for (let j = 0; j < idx.length; j += 2) {
                const p = (idx[j + 1] * S + idx[j]) * 4;
                r += px[p]; g += px[p + 1]; b += px[p + 2];
            }
            r /= count; g /= count; b /= count;
            // Normalize to [0, 1]
            r /= 255; g /= 255; b /= 255;
            const e = this._edges[side];
            e[0] += (r - e[0]) * ct;
            e[1] += (g - e[1]) * ct;
            e[2] += (b - e[2]) * ct;
        }
    }

    // -- build splat list from audio + video state -------------------------

    _buildSplats() {
        const splats = [];
        const { sub, bass, mid, high } = this._bands;

        // If there's no meaningful audio, emit faint idle splats so the
        // simulation doesn't go completely still.
        const totalEnergy = sub + bass + mid + high;

        // Video rect in UV space -- the video is centered in the viewport,
        // constrained to the middle column.  We approximate its position as
        // roughly the center 40% horizontally, full height.  Exact bounds
        // aren't critical; they define where edge dye appears.
        const vL = 0.30; // left edge of video in UV
        const vR = 0.70; // right edge
        const vT = 0.15; // top edge (note: UV y=0 is bottom in GL, but
        const vB = 0.85; // our shaders have vUv.y=0 at bottom)

        // Each edge produces splat(s): force pushes outward, dye = edge color.

        // --- Sub-bass: large radial splats from each edge ------------------
        const subForce = sub * FORCE_SCALE;
        const subRadius = SPLAT_RADIUS + sub * 0.15;

        if (sub > 0.01) {
            // Left edge: multiple splat points distributed vertically
            for (let i = 0; i < 3; i++) {
                const t = (i + 0.5) / 3;
                const y = vT + t * (vB - vT);
                const c = this._edges.left;
                splats.push({
                    x: vL, y,
                    dx: -subForce * 0.5, dy: (t - 0.5) * subForce * 0.15,
                    r: c[0] * sub * 0.8, g: c[1] * sub * 0.8, b: c[2] * sub * 0.8,
                    radius: subRadius,
                });
            }
            // Right edge
            for (let i = 0; i < 3; i++) {
                const t = (i + 0.5) / 3;
                const y = vT + t * (vB - vT);
                const c = this._edges.right;
                splats.push({
                    x: vR, y,
                    dx: subForce * 0.5, dy: (t - 0.5) * subForce * 0.15,
                    r: c[0] * sub * 0.8, g: c[1] * sub * 0.8, b: c[2] * sub * 0.8,
                    radius: subRadius,
                });
            }
            // Top edge
            {
                const c = this._edges.top;
                splats.push({
                    x: 0.5, y: vT,
                    dx: 0, dy: -subForce * 0.4,
                    r: c[0] * sub * 0.7, g: c[1] * sub * 0.7, b: c[2] * sub * 0.7,
                    radius: subRadius * 1.2,
                });
            }
            // Bottom edge
            {
                const c = this._edges.bottom;
                splats.push({
                    x: 0.5, y: vB,
                    dx: 0, dy: subForce * 0.4,
                    r: c[0] * sub * 0.7, g: c[1] * sub * 0.7, b: c[2] * sub * 0.7,
                    radius: subRadius * 1.2,
                });
            }
        }

        // --- Bass: medium splats from edges --------------------------------
        const bassForce = bass * FORCE_SCALE * 0.7;
        const bassRadius = SPLAT_RADIUS + bass * 0.08;

        if (bass > 0.01) {
            for (let i = 0; i < 5; i++) {
                const t = (i + 0.5) / 5;
                const y = vT + t * (vB - vT);
                const cL = this._edges.left;
                const cR = this._edges.right;
                splats.push({
                    x: vL, y,
                    dx: -bassForce * 0.4, dy: (Math.random() - 0.5) * bassForce * 0.1,
                    r: cL[0] * bass * 0.6, g: cL[1] * bass * 0.6, b: cL[2] * bass * 0.6,
                    radius: bassRadius,
                });
                splats.push({
                    x: vR, y,
                    dx: bassForce * 0.4, dy: (Math.random() - 0.5) * bassForce * 0.1,
                    r: cR[0] * bass * 0.6, g: cR[1] * bass * 0.6, b: cR[2] * bass * 0.6,
                    radius: bassRadius,
                });
            }
            // Top and bottom bass splats
            for (let i = 0; i < 3; i++) {
                const t = (i + 0.5) / 3;
                const x = vL + t * (vR - vL);
                const cT = this._edges.top;
                const cB = this._edges.bottom;
                splats.push({
                    x, y: vT,
                    dx: (Math.random() - 0.5) * bassForce * 0.1, dy: -bassForce * 0.3,
                    r: cT[0] * bass * 0.5, g: cT[1] * bass * 0.5, b: cT[2] * bass * 0.5,
                    radius: bassRadius * 0.9,
                });
                splats.push({
                    x, y: vB,
                    dx: (Math.random() - 0.5) * bassForce * 0.1, dy: bassForce * 0.3,
                    r: cB[0] * bass * 0.5, g: cB[1] * bass * 0.5, b: cB[2] * bass * 0.5,
                    radius: bassRadius * 0.9,
                });
            }
        }

        // --- Mid: smaller perturbations from all edges --------------------
        const midForce = mid * FORCE_SCALE * 0.35;
        const midRadius = SPLAT_RADIUS * 0.6 + mid * 0.04;

        if (mid > 0.02) {
            const phase = performance.now() * 0.001;
            for (let i = 0; i < 4; i++) {
                const t = (i + 0.5) / 4;
                const osc = Math.sin(phase * 2.3 + i * 1.7) * 0.02;

                // Left / right
                const cL = this._edges.left;
                const cR = this._edges.right;
                const y = vT + t * (vB - vT) + osc;
                splats.push({
                    x: vL, y,
                    dx: -midForce * 0.25, dy: Math.cos(phase + i) * midForce * 0.08,
                    r: cL[0] * mid * 0.4, g: cL[1] * mid * 0.4, b: cL[2] * mid * 0.4,
                    radius: midRadius,
                });
                splats.push({
                    x: vR, y,
                    dx: midForce * 0.25, dy: Math.cos(phase + i) * midForce * 0.08,
                    r: cR[0] * mid * 0.4, g: cR[1] * mid * 0.4, b: cR[2] * mid * 0.4,
                    radius: midRadius,
                });
            }
        }

        // --- High: shimmer -- modulate existing dye brightness by adding
        //     tiny splats with the edge color at random positions near edges.
        if (high > 0.03) {
            const shimmerForce = high * FORCE_SCALE * 0.15;
            const shimmerRadius = SPLAT_RADIUS * 0.3;
            const sides = [
                { edge: "left",   x: vL, dir: [-1, 0] },
                { edge: "right",  x: vR, dir: [1, 0] },
            ];
            for (const s of sides) {
                for (let i = 0; i < 2; i++) {
                    const y = vT + Math.random() * (vB - vT);
                    const c = this._edges[s.edge];
                    splats.push({
                        x: s.x + (Math.random() - 0.5) * 0.03,
                        y,
                        dx: s.dir[0] * shimmerForce * 0.2,
                        dy: (Math.random() - 0.5) * shimmerForce * 0.15,
                        r: c[0] * high * 0.25,
                        g: c[1] * high * 0.25,
                        b: c[2] * high * 0.25,
                        radius: shimmerRadius,
                    });
                }
            }
        }

        // --- Idle: very faint splats to keep gentle motion when silent ----
        if (totalEnergy < 0.05) {
            const t = performance.now() * 0.0003;
            const idleForce = 200;
            const idleColor = 0.02;
            for (const [edge, x, dirX, dirY] of [
                ["left",  vL, -1,  0],
                ["right", vR,  1,  0],
            ]) {
                const c = this._edges[edge];
                const y = 0.5 + Math.sin(t + (edge === "left" ? 0 : Math.PI)) * 0.15;
                splats.push({
                    x, y,
                    dx: dirX * idleForce, dy: dirY * idleForce + Math.sin(t * 1.3) * 80,
                    r: Math.max(c[0], 0.05) * idleColor,
                    g: Math.max(c[1], 0.05) * idleColor,
                    b: Math.max(c[2], 0.05) * idleColor,
                    radius: SPLAT_RADIUS * 2,
                });
            }
        }

        return splats;
    }
}

// Public re-export so callers can use the more descriptive name. The
// historical class name (AmbientLight) refers to the visual effect; the
// renderer term matches the rest of engine/render.
export { AmbientLight as AmbientRenderer };

