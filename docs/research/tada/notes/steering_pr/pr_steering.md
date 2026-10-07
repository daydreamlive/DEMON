# Generic activation steering with a second SA3 pedal

## Summary

- Add a family-independent steering slot behind the model adapter, with eager hooks and TensorRT input support. Preserve ACE steering behavior and let SA3 use the same slot.
- Load safetensors steering packs for the active family and checkpoint, register each pack as a `steer_*` knob, and combine pack vectors with built-in axes.
- Add the SA3 TensorRT DiT steering input, engine selection and binding, plus eager/TRT parity checks.
- Add pack discovery and live-stream sanity tools for generating and checking steering vectors.
- Expose steering controls in the static SA3 demo from the session knob manifest.
- Document the steering slot, pack format, discovery workflow and SA3 results.
- Give the SA3 demo a separate STEER pedal (291a2fd8). It uses the existing rotary knob interaction and parameter commit path, and appears only for manifests with steering knobs.
- Center partial knob rows on the STEER pedal (cf243595): up to 4 per row, 3 at <=620px, 2 at <=380px.
- Make both pedals smaller and viewport-scaled (83f5b77b): one CSS variable (`--pedal-unit`, a clamp() on vw) sets the root font size and every pedal length is in rem, so the two pedals scale together; they stack below 800px.

## Test evidence

- `node --check demos/sa3/sa3.js`: passed.
- `rg -n 'steerSlider|renderSteer|id="steer"' demos/sa3`: no matches.
- `.venv/Scripts/python.exe -m pytest tests/unit/test_steering_seam.py tests/unit/test_steering_packs.py tests/unit/test_sa3_steering_engine.py tests/unit/test_static_site.py -q`: 42 passed.
- `git diff --check`: passed.

## Screenshots

TODO: Add live screenshots of both pedals at desktop and phone widths after visual review.
