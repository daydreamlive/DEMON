/* DEMON hero animation
   Six thin neon curves continuously flowing rightward,
   each its own phase, frequency, color. Evokes the realtime
   curve readouts in the engine UI.
*/
(function () {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const canvas = document.getElementById('hero-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const dpr = window.devicePixelRatio || 1;
  let w = 0, h = 0;

  function resize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    w = rect.width;
    h = rect.height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize, { passive: true });

  const curves = [
    { color: '#ff3c1f', amp: 38, freq: 0.0026, phase: 0,    speed: 0.0009, yOffset: 0.18, weight: 0.55 },
    { color: '#ff7a00', amp: 52, freq: 0.0019, phase: 1.2,  speed: 0.0011, yOffset: 0.26, weight: 0.5  },
    { color: '#d4ff00', amp: 28, freq: 0.0034, phase: 2.4,  speed: 0.0014, yOffset: 0.36, weight: 0.45 },
    { color: '#00f0e0', amp: 64, freq: 0.0014, phase: 3.6,  speed: 0.0008, yOffset: 0.50, weight: 0.5  },
    { color: '#ff2db4', amp: 44, freq: 0.0022, phase: 0.8,  speed: 0.0010, yOffset: 0.66, weight: 0.45 },
    { color: '#ff7a8a', amp: 32, freq: 0.0028, phase: 1.9,  speed: 0.0013, yOffset: 0.80, weight: 0.4  },
  ];

  let raf = 0;
  let t0 = performance.now();

  function frame(now) {
    const dt = now - t0;
    ctx.clearRect(0, 0, w, h);

    for (const c of curves) {
      ctx.beginPath();
      ctx.lineWidth = c.weight;
      ctx.strokeStyle = c.color;
      ctx.globalAlpha = 0.65;
      ctx.shadowColor = c.color;
      ctx.shadowBlur = 6;

      const baseY = h * c.yOffset;
      const step = 4;
      let drewFirst = false;

      for (let x = 0; x <= w; x += step) {
        // sum of two sines for organic shape
        const y =
          baseY
          + Math.sin(x * c.freq + c.phase + dt * c.speed) * c.amp
          + Math.sin(x * c.freq * 2.13 - c.phase * 0.7 + dt * c.speed * 0.6) * c.amp * 0.35;

        if (!drewFirst) { ctx.moveTo(x, y); drewFirst = true; }
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    ctx.globalAlpha = 1;
    ctx.shadowBlur = 0;

    raf = requestAnimationFrame(frame);
  }

  raf = requestAnimationFrame(frame);

  // pause when hidden to save battery
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) cancelAnimationFrame(raf);
    else raf = requestAnimationFrame(frame);
  });
})();
