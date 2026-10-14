const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { File } = require("node:buffer");
const ts = require("typescript");

function load(file, extra = {}) {
  const exports = {};
  const source = fs.readFileSync(path.join(__dirname, "..", "lib", file), "utf8");
  const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText;
  vm.runInNewContext(js, { exports, process: { env: {} }, FormData, Blob, ...extra });
  return exports;
}

test("goal geometry requires ordered convex normalized corners", () => {
  const { validGoalCorners } = load("goalSetup.ts");
  const points = [[.2, .2], [.8, .2], [.8, .6], [.2, .6]];
  assert.equal(validGoalCorners(points), true);
  assert.equal(validGoalCorners(points.toReversed()), false);
  assert.equal(validGoalCorners([points[0], points[2], points[1], points[3]]), false);
  assert.equal(validGoalCorners([[NaN, 0], ...points.slice(1)]), false);
  assert.equal(validGoalCorners(points.slice(1)), false);
});

test("analysis sends explicit skip or confirmed corners plus the matching photo", async () => {
  let body;
  const api = load("api.ts", { fetch: async (url, options) => {
    assert.match(url, /\/analyze$/);
    body = options.body;
    return { ok: true, json: async () => ({ analysis_id: "test", summary: {}, artifacts: {} }) };
  } });
  const video = new File(["clip"], "rep.mp4", { type: "video/mp4" });
  await api.analyzeVideo(video);
  assert.equal(body.get("goal_setup"), "null");
  assert.equal(body.get("goal_reference"), null);
  const corners = [[.2, .2], [.8, .2], [.8, .6], [.2, .6]];
  await api.analyzeVideo(video, undefined, "right", { corners, reference: new Blob(["reference"], { type: "image/jpeg" }) });
  assert.deepEqual(JSON.parse(body.get("goal_setup")), { confirmed: true, corners });
  assert.equal(await body.get("goal_reference").text(), "reference");
  assert.equal(body.get("athlete_start_side"), "right");
});

test("goal setup additions preserve policy silence", () => {
  const { spokenCue } = load("api.ts");
  assert.equal(spokenCue({ coaching_report: { spoken_cue: null }, summary: { prototype_feedback: "Do not substitute this cue" } }), "");
});
