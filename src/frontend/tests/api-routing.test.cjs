const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");

function load(file, env) {
  const exports = {};
  const source = fs.readFileSync(path.join(__dirname, "..", file), "utf8");
  const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText;
  vm.runInNewContext(js, { exports, process: { env } });
  return exports;
}

test("stale public IP settings cannot change browser API routing", () => {
  const api = load("lib/api.ts", { NEXT_PUBLIC_API_BASE: "https://192.168.4.71:8000" });
  assert.equal(api.API_BASE, "/api/scan");
});

test("all API and artifact paths proxy to loopback by default", async () => {
  const config = load("next.config.ts", {}).default;
  const [rule] = await config.rewrites();
  assert.equal(rule.source, "/api/scan/:path*");
  assert.equal(rule.destination, "http://127.0.0.1:8000/:path*");
  assert.ok(config.experimental.proxyTimeout >= 600000);
});

test("server destination and frontend development hosts can be configured separately", async () => {
  const config = load("next.config.ts", {
    SCAN_BACKEND_ORIGIN: "http://127.0.0.1:9000/",
    SCAN_DEV_HOSTS: "scan.local, localhost",
  }).default;
  assert.equal((await config.rewrites())[0].destination, "http://127.0.0.1:9000/:path*");
  assert.equal(Array.from(config.allowedDevOrigins).join(","), "scan.local,localhost");
});
