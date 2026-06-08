"""Magenta RealTime 2 backend family.

A :class:`~acestep.streaming.generator_backend.GeneratorBackend`
implementor for Google's Magenta RT 2 (recurrentgemma backbone,
SpectroStream codec) — the first token/AR family behind the seam and
the first sidecar-hosted one. See ``backend.py`` for the in-process
client and ``scripts/mrt2_sidecar.py`` for the generation loop that
runs next to the JAX model (WSL venv or Linux pod; JAX has no CUDA on
native Windows).
"""
