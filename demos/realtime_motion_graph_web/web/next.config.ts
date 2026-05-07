import type { NextConfig } from "next";

// During development, Next runs on :3000 and the Python engine on :8765.
// The UI's engine-URL builder fetches /api/* and /fixtures/* (etc.) at
// the same origin; rewrites bridge those to the engine. Same trick the
// vanilla static demo gets for free since it's served by the engine
// directly.
//
// In production, you'd typically `next build && next start` and run the
// engine on a different port (or front Next behind nginx that does the
// routing). For local dev this is the one-line fix.
const nextConfig: NextConfig = {
  async rewrites() {
    return [
      { source: "/api/:path*", destination: "http://localhost:8765/api/:path*" },
      { source: "/fixtures/:path*", destination: "http://localhost:8765/fixtures/:path*" },
      { source: "/loras/:path*", destination: "http://localhost:8765/loras/:path*" },
      { source: "/videos/:path*", destination: "http://localhost:8765/videos/:path*" },
    ];
  },
};

export default nextConfig;
