const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");

function setup(t, unlock = async () => {}) {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const exports = {};
  const source = fs.readFileSync(path.join(__dirname, "../lib/soundCheck.ts"), "utf8");
  vm.runInNewContext(ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText,
    { exports, setTimeout: (...args) => setTimeout(...args), clearTimeout: (timer) => clearTimeout(timer) });
  const calls = [];
  let speech;
  const check = exports.createSoundCheck({ unlock,
    pulse: () => calls.push("pulse"),
    speak: (text, options) => { calls.push(text); speech = options; },
    stopTone: () => calls.push("stop-tone"), stopVoice: () => calls.push("stop-voice"),
    status: (text, active) => calls.push({ text, active }),
  });
  return { check, calls, speech: () => speech };
}

test("sequence finishes the three pulses before speaking a sample", async (t) => {
  const { check, calls, speech } = setup(t);
  await check.start("sequence");
  t.mock.timers.tick(1200);
  assert.equal(calls.filter((c) => c === "pulse").length, 3);
  assert.equal(speech(), undefined);
  t.mock.timers.tick(800);
  assert.ok(speech());
  speech().onEnd();
  assert.equal(calls.at(-1).active, false);
});

test("stop cancels delayed pulses and the queued voice", async (t) => {
  const { check, calls, speech } = setup(t);
  await check.start("sequence");
  check.stop();
  const count = calls.length;
  t.mock.timers.tick(30000);
  assert.equal(calls.length, count);
  assert.equal(speech(), undefined);
});

test("cancelling while audio unlock is pending prevents late playback", async (t) => {
  let resolve;
  const { check, calls } = setup(t, () => new Promise((done) => { resolve = done; }));
  const pending = check.start("beacon");
  check.stop(); resolve(); await pending;
  assert.equal(calls.includes("pulse"), false);
});

test("voice errors and missing completion events release the controls", async (t) => {
  const { check, calls, speech } = setup(t);
  await check.start("voice");
  speech().onError();
  assert.equal(calls.at(-1).active, false);
  await check.start("voice");
  t.mock.timers.tick(15000);
  assert.equal(calls.at(-1).active, false);
  assert.match(calls.at(-1).text, /did not finish/);
});

test("failed audio unlock reports failure without playing", async (t) => {
  const { check, calls } = setup(t, async () => { throw new Error("blocked"); });
  await check.start("beacon");
  assert.equal(calls.includes("pulse"), false);
  assert.equal(calls.at(-1).active, false);
  assert.match(calls.at(-1).text, /could not start/);
});
