"""Analysis-side consumers of the streaming session's event bus.

Components here observe the session (via ``session.bus``) and never
touch the runner thread, the backend, or the engine. See
``midi_live.py`` for the live audio->MIDI transcriber spike.
"""
