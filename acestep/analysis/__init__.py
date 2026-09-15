"""Analysis passes over audio the session already has (transcription, ...).

Nothing here touches generation; modules are imported lazily by the WS
adapter so a pod without the optional analysis deps still boots.
"""
