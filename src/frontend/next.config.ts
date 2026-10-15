import type { NextConfig } from "next";

// Server-only destination: never expose a laptop IP in browser API requests.
const backendOrigin = (process.env.SCAN_BACKEND_ORIGIN || "http://127.0.0.1:8000").replace(/\/+$/, "");
// Next's dev asset allowlist takes frontend hostnames, without schemes or ports.
const devHosts = (process.env.SCAN_DEV_HOSTS ?? "")
  .split(",")
  .map((host) => host.trim())
  .filter(Boolean);

const nextConfig: NextConfig = {
  allowedDevOrigins: devHosts,
  experimental: { proxyTimeout: 600_000 },
  async rewrites() {
    return [{ source: "/api/scan/:path*", destination: `${backendOrigin}/:path*` }];
  },
};

export default nextConfig;
