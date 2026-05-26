import { defaultWsUrl } from "@/engine/podUrl";
import { float16ArrayToFloat32 } from "@/engine/protocol";
import { getApiKey } from "@/engine/rtmgConfig";
import type { DecodedFixture, DecodedStemAssets } from "@/engine/audio/loadFixture";
import type { LegoLayerConfig } from "@/types/lego";

function resolveWsUrl(serverWsUrl: string | null): string {
  let url = serverWsUrl ?? defaultWsUrl();
  const apiKey = getApiKey();
  if (apiKey) {
    const sep = url.includes("?") ? "&" : "?";
    url = `${url}${sep}apiKey=${encodeURIComponent(apiKey)}`;
  }
  return url;
}

export async function generateLegoStemsPreflight(opts: {
  wsUrl: string | null;
  fixtureName: string;
  decoded: DecodedFixture;
  layers: LegoLayerConfig[];
  onStatus?: (track: string, status: string, error?: string) => void;
}): Promise<DecodedStemAssets> {
  const layers = opts.layers
    .map((layer) => ({ track: layer.track, prompt: layer.prompt.trim() }))
    .filter((layer) => layer.prompt.length > 0);
  if (layers.length === 0) return {};

  return new Promise((resolve, reject) => {
    const ws = new WebSocket(resolveWsUrl(opts.wsUrl));
    ws.binaryType = "arraybuffer";
    let pending:
      | {
          channels: number;
          frames: number;
          sampleRate: number;
          layers: string[];
          buffers: Partial<Record<string, Float32Array>>;
        }
      | null = null;

    const fail = (err: Error) => {
      try {
        ws.close();
      } catch {}
      reject(err);
    };

    ws.onopen = () => {
      ws.send(JSON.stringify({
        type: "lego_preflight",
        fixture_name: opts.fixtureName,
        layers,
        model: "acestep-v15-base",
        seed: 1528,
        steps: 50,
        shift: 1.0,
        cfg_scale: 7.0,
      }));
      const { interleaved, channels } = opts.decoded;
      const samples = interleaved.length / channels;
      const hdr = new ArrayBuffer(8);
      const dv = new DataView(hdr);
      dv.setUint32(0, channels, true);
      dv.setUint32(4, samples, true);
      const combined = new Uint8Array(hdr.byteLength + interleaved.byteLength);
      combined.set(new Uint8Array(hdr), 0);
      combined.set(new Uint8Array(interleaved.buffer), hdr.byteLength);
      ws.send(combined);
    };

    ws.onmessage = (ev) => {
      if (pending && ev.data instanceof ArrayBuffer) {
        const layer = pending.layers[Object.keys(pending.buffers).length];
        if (layer) {
          pending.buffers[layer] = float16ArrayToFloat32(new Uint16Array(ev.data));
        }
        if (pending.layers.every((name) => pending?.buffers[name])) {
          const stems: DecodedStemAssets = {};
          for (const name of pending.layers) {
            const interleaved = pending.buffers[name];
            if (!interleaved) continue;
            stems[name] = {
              interleaved,
              channels: pending.channels,
              frames: pending.frames,
              sampleRate: pending.sampleRate,
            };
          }
          pending = null;
          try {
            ws.close();
          } catch {}
          resolve(stems);
        }
        return;
      }

      if (typeof ev.data !== "string") return;
      let msg: { type?: string; [k: string]: unknown };
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === "lego_status") {
        opts.onStatus?.(
          String(msg.track || ""),
          String(msg.status || ""),
          typeof msg.error === "string" ? msg.error : undefined,
        );
      } else if (msg.type === "lego_assets") {
        pending = {
          channels: Number(msg.channels || 2),
          frames: Number(msg.frames || 0),
          sampleRate: Number(msg.sample_rate || 48000),
          layers: Array.isArray(msg.layers) ? msg.layers.map(String) : [],
          buffers: {},
        };
      } else if (msg.type === "error") {
        fail(new Error(String(msg.message || msg.code || "LEGO generation failed")));
      }
    };

    ws.onerror = () => {
      fail(new Error("LEGO websocket connection failed"));
    };
    ws.onclose = (e) => {
      if (pending) {
        reject(new Error(e.reason || "LEGO generation closed before assets arrived"));
      }
    };
  });
}
