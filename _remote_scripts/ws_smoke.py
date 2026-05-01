"""Minimal WebSocket handshake smoke test against a deployment.

Usage: python ws_smoke.py <host:port>
"""

import asyncio
import json
import sys

import websockets


async def main(host_port: str):
    url = f"ws://{host_port}/"
    print(f"connecting to {url}")
    async with websockets.connect(url, open_timeout=15, max_size=50 * 1024 * 1024) as ws:
        print("connected; sending handshake-only config (no audio upload)")
        await ws.send(
            json.dumps(
                {
                    "sde": False,
                    "lora": False,
                    "depth": 2,
                    "vae_window": 6,
                    "crop": 0,
                    "steps": 2,
                    "prompt": "handshake-smoke-test",
                    "lora_path": None,
                    "fast_vae": False,
                }
            )
        )
        await asyncio.sleep(0.3)
        print("handshake ok")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "109.72.59.164:31409"
    sys.exit(asyncio.run(main(target)))
