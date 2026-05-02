# LoRA weights — moved

LoRAs no longer live in this directory. Drop your `.safetensors` files
under the project's models directory:

```
$ACESTEP_MODELS_DIR/loras/
# defaults to ~/.daydream-scope/models/demon/loras/ when the env var
# is unset (see acestep/paths.py::loras_dir).
```

The realtime demo's TRT engine scans that directory at startup via
`TRTLoRAManager.register_library()` and publishes the catalog to the
client. Each `.safetensors` becomes one library entry whose id is the
filename stem, which is what the operator UI's Library panel toggles
on/off.

Keep paths layout-portable: the server reads
`acestep.paths.loras_dir()`, which respects `ACESTEP_MODELS_DIR`. Sync
LoRAs out-of-band (rsync, container layer, etc.) just like checkpoints
and TRT engines:

```bash
# local -> remote 5090
rsync -avz ~/.daydream-scope/models/demon/loras/ \
    user@5090-host:/path/to/.daydream-scope/models/demon/loras/
```

If you have a non-catalog `.safetensors` somewhere else on disk, the
server also accepts ad-hoc paths via the `lora_paths` field in the
WebSocket init config (server registers them on the fly). The native
thin client and the legacy full demo use this path; the web client
goes through the catalog.
