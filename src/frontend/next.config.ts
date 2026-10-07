import type { NextConfig } from "next";

// Phone development uses the same laptop hostname for the frontend and API.
// Next's dev asset allowlist takes hostnames, without schemes or ports.
const apiHostname = process.env.NEXT_PUBLIC_API_BASE
  ? new URL(process.env.NEXT_PUBLIC_API_BASE).hostname
  : undefined;
const devHosts = (process.env.SCAN_DEV_HOSTS ?? "")
  .split(",")
  .map((host) => host.trim())
  .filter(Boolean);

const nextConfig: NextConfig = {
  allowedDevOrigins: apiHostname ? [...devHosts, apiHostname] : devHosts,
};

export default nextConfig;
