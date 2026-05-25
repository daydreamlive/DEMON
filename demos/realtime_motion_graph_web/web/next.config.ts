import type { NextConfig } from "next";

// During development, Next runs on :6660 and the Python engine defaults to :1318.
// The UI's engine-URL builder fetches /api/* and /fixtures/* (etc.) at
// the same origin; rewrites bridge those to the engine URL provided by run.py.
//
// In production, you'd typically `next build && next start` and run the
// engine on a different port (or front Next behind nginx that does the
// routing). For local dev this is the one-line fix.
const backendUrl = (
  process.env.NEXT_PUBLIC_POD_BASE_URL ?? "http://127.0.0.1:1318"
).replace(/\/$/, "");

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${backendUrl}/api/:path*` },
      { source: "/fixtures/:path*", destination: `${backendUrl}/fixtures/:path*` },
      { source: "/loras/:path*", destination: `${backendUrl}/loras/:path*` },
      { source: "/videos/:path*", destination: `${backendUrl}/videos/:path*` },
    ];
  },
};

export default nextConfig;
