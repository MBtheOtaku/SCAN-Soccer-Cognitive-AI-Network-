// Run with the installed TypeScript compiler, without a device or new packages.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const base = path.resolve(__dirname, '..');
const tests = [];
const test = (name, run) => tests.push({ name, run });
const plain = value => JSON.parse(JSON.stringify(value));
const jobId = '1234567890abcdef1234567890abcdef';
const origin = 'https://scan.example';
const route = `/experimental-tracking/${jobId}`;
const source = ts.transpileModule(fs.readFileSync(path.join(base, 'lib/experimental-tracking.ts'), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;

function harness(responses = []) {
  const calls = [], forms = [];
  class LocalFile { constructor(uri) { this.uri = uri; this.exists = true; this.size = 123; } }
  class Multipart { constructor() { this.fields = new Map(); forms.push(this); } append(name, value, fileName) { this.fields.set(name, { value, fileName }); } }
  const module = { exports: {} };
  vm.runInNewContext(source, {
    module, exports: module.exports, URL, AbortController, setTimeout, clearTimeout, console,
    FormData: Multipart, process: { env: { EXPO_PUBLIC_SCAN_API_URL: origin } },
    require: name => {
      if (name === 'expo-file-system') return { File: LocalFile };
      if (name === 'expo/fetch') return { fetch: async (url, options) => {
        calls.push({ url, options });
        assert.equal(options.signal.aborted, false);
        const data = responses.shift();
        if (data === undefined) throw new Error('Unexpected request');
        return { ok: true, status: 200, json: async () => data };
      } };
      throw new Error(`Unexpected import: ${name}`);
    },
  });
  return { api: module.exports, calls, forms };
}
const { api } = harness();
const preview = () => ({ tracking_id: jobId, status: 'ready', preview_url: route + '/preview',
  width: 1080, height: 1920, preview_width: 360, preview_height: 640,
  frame_index: 2, time_s: 2 / 30, fps: 30, duration_s: 5 });
const status = (phase = 'running') => ({ tracking_id: jobId, status: phase, processed_frames: 20, total_frames: 150,
  result_url: phase === 'completed' ? route + '/result' : null, error: null });
const point = (frame, state = 'observed_candidate') => ({ frame, time_s: frame / 30, state,
  center: state === 'observed_candidate' ? { x: .2, y: .4 } : null,
  prediction: state === 'predicted' ? { x: .25, y: .35 } : null, identity_verified: false, reason: 'Unverified identity.' });
const result = () => ({ tracking_id: jobId, status: 'completed', width: 1080, height: 1920, fps: 30,
  start_frame: 2, seed: { x: .5, y: .8 }, points: [point(2), point(3, 'predicted'), point(4, 'lost')] });

test('HTTPS origin excludes paths, credentials, queries and fragments', () => {
  assert.equal(api.trackingOrigin(' https://scan.example:8443/ '), 'https://scan.example:8443');
  for (const bad of ['http://scan.example', 'https://user:pass@scan.example', 'https://scan.example/api',
    'https://scan.example/?key=secret', 'https://scan.example/#token', '']) {
    assert.throws(() => api.trackingOrigin(bad));
  }
});

test('server artifacts are restricted to the expected job and resource', () => {
  assert.equal(api.trackingArtifactPath(route + '/preview', jobId, 'preview'), route + '/preview');
  for (const bad of ['https://evil.example/preview', '//evil.example/preview', route + '/result',
    route + '/preview?token=x', '/experimental-tracking/../preview', '/analyze', route + '%2fpreview']) {
    assert.throws(() => api.trackingArtifactPath(bad, jobId, 'preview'));
  }
  assert.throws(() => api.trackingArtifactPath('/anything', '../escape', 'preview'));
});

test('contain mapping rejects letterbox margins and maps real image coordinates', () => {
  const rect = api.containRect(300, 300, 100, 200);
  assert.deepEqual(plain(rect), { x: 75, y: 0, width: 150, height: 300 });
  assert.deepEqual(plain(api.tapToSeed(150, 150, rect)), { x: .5, y: .5 });
  assert.deepEqual(plain(api.tapToSeed(75, 0, rect)), { x: 0, y: 0 });
  assert.equal(api.tapToSeed(74.9, 150, rect), null);
  assert.equal(api.tapToSeed(225, 150, rect), null);
  assert.equal(api.tapToSeed(100, 300, rect), null);
  assert.equal(api.tapToSeed(NaN, 0, rect), null);
  assert.equal(api.containRect(0, 300, 100, 200), null);
  const landscape = api.containRect(300, 400, 1920, 1080);
  assert.equal(api.tapToSeed(100, 100, landscape), null);
});

test('preview verifies oriented aspect and exact initial-frame metadata', () => {
  assert.deepEqual(plain(api.parseTrackingPreview(preview())), preview());
  assert.equal(api.previewImageUrl(api.parseTrackingPreview(preview())), origin + route + '/preview');
  for (const change of [{ width: 1920, height: 1080 }, { fps: 0 }, { time_s: 2 }, { frame_index: -1 },
    { tracking_id: 'not-a-job' }, { preview_url: 'https://evil.example/image.jpg' }]) {
    assert.throws(() => api.parseTrackingPreview({ ...preview(), ...change }));
  }
});

test('status cannot swap jobs, exceed progress or attach an external result', () => {
  assert.deepEqual(plain(api.parseTrackingStatus(status(), jobId)), status());
  for (const change of [{ tracking_id: 'a'.repeat(32) }, { processed_frames: 151 }, { total_frames: -1 },
    { status: 'surprise' }, { result_url: 'https://evil.example/result' }, { status: 'completed', result_url: null }]) {
    assert.throws(() => api.parseTrackingStatus({ ...status(), ...change }, jobId));
  }
});

test('result validates ordered frames and never promotes seed or identity to observation', () => {
  assert.deepEqual(plain(api.parseTrackingResult(result(), jobId)), result());
  for (const change of [{ seed: { x: 1, y: .5 } }, { points: [point(3), point(2)] },
    { points: [point(2), point(2)] }, { points: [{ ...point(2), identity_verified: true }] },
    { points: [{ ...point(2), center: { x: -1, y: .5 } }] }, { points: [{ ...point(2), time_s: 4 }] },
    { points: [point(0)] }]) {
    assert.throws(() => api.parseTrackingResult({ ...result(), ...change }, jobId));
  }
  const lost = { ...point(4, 'lost'), center: { x: .5, y: .5 }, prediction: { x: .5, y: .5 } };
  const parsed = api.parseTrackingResult({ ...result(), points: [lost] }, jobId);
  assert.equal(parsed.points[0].center, null);
  assert.equal(parsed.points[0].prediction, null);
  assert.equal(api.pointAtTime([], 2 / 30, 30), null);
});

test('frame lookup does not snap to a later point or carry across a missing frame', () => {
  const sparse = [point(30), point(32), point(33, 'lost')];
  assert.equal(api.pointAtTime(sparse, 1, 30).frame, 30);
  assert.equal(api.pointAtTime(sparse, 1.04, 30), null);
  assert.equal(api.pointAtTime(sparse, 1.1, 30).state, 'lost');
  assert.equal(api.pointAtTime(sparse, 1.15, 30), null);
  assert.equal(api.pointAtTime(sparse, -1, 30), null);
  assert.equal(api.pointAtTime([{ ...point(30), time_s: 1.01 }], 1, 30), null);
});

test('presentation timestamps align a 30fps timeline even when metadata reports 23fps', () => {
  const timed = { ...result(), fps: 22.994379151762903, timing_mode: 'presentation_timestamps',
    points: Array.from({ length: 150 }, (_, frame) => ({ ...point(frame), time_s: frame / 30,
      end_time_s: (frame + 1) / 30 })), start_frame: 0 };
  const parsed = api.parseTrackingResult(timed, jobId);
  assert.equal(api.pointAtTime(parsed.points, 2.64, parsed.fps).frame, 79);
  assert.equal(api.pointAtTime(parsed.points, 4.99, parsed.fps).frame, 149);
  assert.equal(api.pointAtTime(parsed.points, 5, parsed.fps), null);
  assert.equal(api.pointAtTime(parsed.points, 1 / 30, parsed.fps).frame, 1);
  const sparse = parsed.points.filter(p => p.frame !== 80);
  assert.equal(api.pointAtTime(sparse, 80.5 / 30, parsed.fps), null);
  const lost = api.parseTrackingResult({ ...timed, points: timed.points.map(p => p.frame === 80
    ? { ...p, state: 'lost', center: { x: .2, y: .4 } } : p) }, jobId);
  assert.equal(api.pointAtTime(lost.points, 80.5 / 30, lost.fps).state, 'lost');
  assert.equal(api.pointAtTime(lost.points, 80.5 / 30, lost.fps).center, null);
  for (const change of [{ timing_mode: 'guess' },
    { points: [{ ...timed.points[0], end_time_s: undefined }] },
    { points: [{ ...timed.points[0], end_time_s: 0 }] },
    { points: [{ ...timed.points[0], end_time_s: NaN }] },
    { points: [{ ...timed.points[0], end_time_s: .1 }, timed.points[1]] },
    { points: [timed.points[0], { ...timed.points[1], time_s: 0 }] }]) {
    assert.throws(() => api.parseTrackingResult({ ...timed, ...change }, jobId));
  }
});

test('preview upload uses SDK58 File multipart only and never calls coaching', async () => {
  const h = harness([preview()]);
  const rep = { name: 'rep-123-test.mov', uri: 'file:///documents/rep-123-test.mov' };
  await h.api.prepareTracking(rep, new AbortController().signal);
  assert.equal(h.calls[0].url, origin + '/experimental-tracking/preview');
  assert.equal(h.calls[0].options.method, 'POST');
  assert.equal(h.forms.length, 1);
  assert.equal(h.forms[0].fields.size, 1);
  assert.equal(h.forms[0].fields.get('video').value.uri, rep.uri);
  assert.equal(h.forms[0].fields.get('video').fileName, rep.name);
  assert.equal(h.calls[0].options.headers['Content-Type'], undefined);
});

test('start sends only normalized seed and reconnect fetches existing job without resubmission', async () => {
  const h = harness([status('queued'), status('completed'), result()]);
  const signal = new AbortController().signal;
  await h.api.startTracking(jobId, { x: .5, y: .8 }, signal);
  assert.deepEqual(JSON.parse(h.calls[0].options.body), { seed: { x: .5, y: .8 } });
  assert.equal(h.calls[0].options.headers['Content-Type'], 'application/json');
  const received = [];
  const completed = await h.api.waitForTracking(jobId, signal, data => received.push(data.status));
  assert.equal(completed.status, 'completed');
  assert.deepEqual(received, ['completed']);
  assert.deepEqual(h.calls.map(call => call.url), [origin + route + '/start', origin + route, origin + route + '/result']);
  assert.equal(h.forms.length, 0);
});

test('stop waiting aborts polling without sending cancellation or a new upload', async () => {
  const h = harness([status()]), controller = new AbortController();
  await assert.rejects(h.api.waitForTracking(jobId, controller.signal, () => controller.abort()), /Stopped waiting/);
  assert.equal(h.calls.length, 1);
  assert.equal(h.calls[0].url, origin + route);
  await assert.rejects(h.api.startTracking(jobId, { x: 1, y: 0 }, new AbortController().signal), /Tap the ball/);
});

test('recovery is explicit, progress is separate, and failed recovery retains basic observations', async () => {
  const h = harness([status('queued')]);
  await h.api.startTracking(jobId, { x: .5, y: .8 }, new AbortController().signal, true);
  assert.deepEqual(JSON.parse(h.calls[0].options.body), { seed: { x: .5, y: .8 }, recover_misses: true });
  const prepared = api.parseTrackingPreview({ ...preview(), recovery_available: true });
  assert.equal(prepared.recovery_available, true);
  const progress = api.parseTrackingStatus({ ...status(), phase: 'recovery', recovery_processed_windows: 1, recovery_total_windows: 4 }, jobId);
  assert.equal(progress.phase, 'recovery');
  assert.equal(progress.recovery_processed_windows, 1);
  assert.throws(() => api.parseTrackingStatus({ ...status(), phase: 'recovery', recovery_processed_windows: 5, recovery_total_windows: 4 }, jobId));
  const data = { ...result(), processing: { model_load_s: 1, baseline_s: 20, recovery_s: 30, total_s: 51 },
    recovery: { requested: true, status: 'unavailable', added_observations: 0, warning: 'Basic tracking is available.' } };
  const parsed = api.parseTrackingResult(data, jobId);
  assert.deepEqual(plain(parsed.points), data.points);
  assert.equal(parsed.processing.total_s, 51);
  assert.equal(parsed.recovery.status, 'unavailable');
  assert.throws(() => api.parseTrackingResult({ ...data, processing: { ...data.processing, total_s: NaN } }, jobId));
  const lost = { ...point(4, 'lost'), recovered: true };
  assert.equal(api.parseTrackingResult({ ...result(), points: [lost] }, jobId).points[0].recovered, false);
});

test('explicit cancellation targets only the existing job and does not upload or analyze', async () => {
  const h = harness([status('cancelled')]);
  const cancelled = await h.api.cancelTracking(jobId, new AbortController().signal);
  assert.equal(cancelled.status, 'cancelled');
  assert.equal(h.calls[0].url, origin + route + '/cancel');
  assert.equal(h.calls[0].options.method, 'POST');
  assert.equal(h.calls.length, 1);
  assert.equal(h.forms.length, 0);
});

(async () => {
  let passed = 0;
  for (const { name, run } of tests) {
    try { await run(); console.log(`PASS ${name}`); passed++; }
    catch (error) { console.error(`FAIL ${name}`); console.error(error); process.exitCode = 1; }
  }
  console.log(`${passed}/${tests.length} experimental-tracking checks passed.`);
})();
