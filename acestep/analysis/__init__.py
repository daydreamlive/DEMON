"""Listening models over the canvas: the analysis seam.

The generator seam writes the canvas; everything here only reads audio
and returns items (notes now; beats, chords later):

- :mod:`.engine`: the engine / request / result / submitter contract.
- :mod:`.registry`: one ``AnalyzerSpec`` per analyzer.
- :mod:`.host`: the process-wide host that owns loaded engines.
- :mod:`.notes_engine`: the ``notes`` analyzer (MuScriptor).
- :mod:`.midi_transcribe`: the plugin's one-shot, served by the host.

Nothing here touches generation, and every module is torch-free at
import, so a pod without the optional analysis deps still boots.
"""
