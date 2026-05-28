"""Transport-agnostic streaming session API for ACE-Step.

Modules in this package are transport-agnostic: no JSON, no
WebSockets, no wire bytes. Transport adapters (``ws_adapter`` in the
web demo, the MCP control bus, a future VST plugin) translate between
the wire and the typed API surface and subscribe to a typed event bus.
"""
