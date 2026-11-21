// Dependency-free checks using the project's installed TypeScript compiler.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const base = path.resolve(__dirname, '..');
const tests = [];
const test = (name, run) => tests.push({ name, run });
const plain = value => JSON.parse(JSON.stringify(value));

function load(file, modules = {}, extra = {}) {
  const module = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(path.join(base, file), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  vm.runInNewContext(code, { module, exports: module.exports, require: name => {
    if (Object.hasOwn(modules, name)) return modules[name];
    throw new Error(`Unexpected dependency: ${name}`);
  }, console, URL, AbortController, setTimeout, clearTimeout, ...extra }, { filename: file });
  return module.exports;
}

const intentions = load('lib/training-intention.ts');
const feedback = load('lib/intention-feedback.ts', { './training-intention': intentions });
const general = { shot_type: 'unspecified', target_zone: null, reported_outcome: null };
const curl = { shot_type: 'curl', target_zone: 'upper_left', reported_outcome: null };

function storage() {
  const files = new Map(), events = [];
  let failMetadata = false;
  const api = {
    documentDirectory: 'file:///documents/', cacheDirectory: 'file:///cache/',
    makeDirectoryAsync: async () => {},
    getInfoAsync: async uri => ({ exists: files.has(uri), isDirectory: false, size: Buffer.byteLength(files.get(uri) || '') }),
    readAsStringAsync: async uri => { if (!files.has(uri)) throw new Error('Missing'); return files.get(uri); },
    writeAsStringAsync: async (uri, data) => { files.set(uri, data); events.push(['write', uri]); },
    readDirectoryAsync: async uri => [...files.keys()].filter(key => key.startsWith(uri)).map(key => key.slice(uri.length)),
    copyAsync: async ({ from, to }) => { files.set(to, files.get(from)); events.push(['copy', to]); },
    moveAsync: async ({ from, to }) => {
      if (failMetadata && to.endsWith('.title.json')) throw new Error('Simulated storage interruption');
      files.set(to, files.get(from)); files.delete(from); events.push(['move', to]);
    },
    deleteAsync: async uri => { files.delete(uri); events.push(['delete', uri]); },
  };
  return { files, events, api, interruptMetadata: value => { failMetadata = value; },
    reps: load('lib/reps.ts', { 'expo-file-system/legacy': api, './training-intention': intentions }) };
}

test('old and malformed optional intention fields resolve to general', () => {
  assert.deepEqual(plain(intentions.normalizeIntention()), general);
  assert.deepEqual(plain(intentions.normalizeIntention({ shot_type: 'invented', target_zone: '../video', reported_outcome: true })), general);
  assert.equal(intentions.intentionKey({ ...curl, ignored: 'value' }), intentions.intentionKey(curl));
});

test('legacy and corrupt sidecars never hide existing video files', async () => {
  const h = storage(), root = 'file:///documents/scan-reps/';
  h.files.set(root + 'rep-123-abcd.mov', 'video');
  h.files.set(root + 'rep-123-abcd.mov.title.json', '{bad json');
  const result = await h.reps.listReps();
  assert.equal(result.length, 1);
  assert.deepEqual(plain(result[0].trainingIntention), general);
  assert.equal(h.files.get(result[0].uri), 'video');
});

test('metadata failure preserves video, recovery journal, cached original and recording intention', async () => {
  const h = storage(), root = 'file:///documents/scan-reps/';
  h.files.set('file:///cache/camera/clip.mov', 'original-video-bytes');
  h.interruptMetadata(true);
  await assert.rejects(h.reps.saveRep('file:///cache/camera/clip.mov', curl), /storage interruption/);
  const journal = JSON.parse(h.files.get(root + 'pending.json'));
  assert.deepEqual(journal.trainingIntention, curl);
  assert.equal(h.files.get(root + journal.name), 'original-video-bytes');
  assert.equal(h.files.get('file:///cache/camera/clip.mov'), 'original-video-bytes');
  h.files.set(root + journal.name + '.title.json', JSON.stringify({ title: 'Keep title', notes: 'Keep notes' }));
  h.interruptMetadata(false);
  await h.reps.recoverSave();
  assert.equal(h.files.has(root + 'pending.json'), false);
  assert.equal(h.files.has('file:///cache/camera/clip.mov'), false);
  const result = (await h.reps.listReps())[0];
  assert.deepEqual(plain(result.trainingIntention), curl);
  assert.equal(result.title, 'Keep title'); assert.equal(result.notes, 'Keep notes');
  assert.equal(h.events.filter(([kind]) => kind === 'copy').length, 1);
  const metadataMoved = h.events.findIndex(([kind, uri]) => kind === 'move' && uri.endsWith('.title.json'));
  const originalDeleted = h.events.findIndex(([kind, uri]) => kind === 'delete' && uri === 'file:///cache/camera/clip.mov');
  assert.ok(metadataMoved >= 0 && originalDeleted > metadataMoved);
});

test('renaming and clearing notes preserve per-clip intention and original video', async () => {
  const h = storage(), root = 'file:///documents/scan-reps/', name = 'rep-123-abcd.mov';
  h.files.set(root + name, 'original');
  await h.reps.saveRepDetails(name, 'First title', 'Notes', { ...curl, reported_outcome: 'goal' });
  await h.reps.renameRep(name, 'Renamed');
  await h.reps.saveRepDetails(name, '', '');
  const rep = (await h.reps.listReps())[0];
  assert.equal(rep.title, undefined); assert.equal(rep.notes, undefined);
  assert.deepEqual(plain(rep.trainingIntention), { ...curl, reported_outcome: 'goal' });
  assert.equal(h.files.get(root + name), 'original');
  await h.reps.saveRepDetails(name, 'New title', 'Retained notes', general);
  assert.deepEqual(plain((await h.reps.listReps())[0].trainingIntention), general);
});

test('old recovery journals receive general intention without deleting video', async () => {
  const h = storage(), root = 'file:///documents/scan-reps/';
  const job = { name: 'rep-123-abcd.mov', cachePath: 'clip.mov', bytes: 3 };
  h.files.set(root + 'pending.json', JSON.stringify(job)); h.files.set('file:///cache/clip.mov', 'abc');
  await h.reps.recoverSave();
  assert.deepEqual(plain((await h.reps.listReps())[0].trainingIntention), general);
  assert.equal(h.files.get(root + job.name), 'abc');
});

test('invalid recovery paths remain untouched and cannot reach arbitrary files', async () => {
  const h = storage(), root = 'file:///documents/scan-reps/';
  h.files.set(root + 'pending.json', JSON.stringify({ name: 'rep-123-abcd.mov', cachePath: '../outside.mov', bytes: 3 }));
  await assert.rejects(h.reps.recoverSave(), /storage recovery/);
  assert.ok(h.files.has(root + 'pending.json'));
  assert.deepEqual(h.events, []);
});

function backend(echoOverride) {
  const forms = [];
  class FakeForm {
    constructor() { this.fields = new Map(); forms.push(this); }
    append(key, value) { this.fields.set(key, value); }
  }
  const api = load('lib/backend.ts', {
    'expo-file-system/legacy': { getInfoAsync: async () => ({ exists: true, isDirectory: false, size: 42 }) },
    'expo-file-system': { File: class File { constructor(uri) { this.uri = uri; } } },
    'expo/fetch': { fetch: async (url, options) => ({ ok: true, json: async () => ({
      analysis_id: 'analysis-123', summary: {}, artifacts: {},
      training_intention: echoOverride ?? JSON.parse(options.body.fields.get('training_intention')),
    }) }) }, './training-intention': intentions, './intention-feedback': feedback,
  }, { FormData: FakeForm, process: { env: { EXPO_PUBLIC_SCAN_API_URL: 'https://scan.example' } } });
  return { api, forms };
}

test('upload snapshots exact clip intention before asynchronous file inspection', async () => {
  const h = backend(), rep = { name: 'rep-123-abcd.mov', uri: 'file:///rep.mov', trainingIntention: { ...curl } };
  const task = h.api.analyseRep(rep, new AbortController().signal);
  rep.trainingIntention.shot_type = 'power';
  const result = await task;
  assert.deepEqual(JSON.parse(h.forms[0].fields.get('training_intention')), curl);
  assert.deepEqual(plain(result.trainingIntention), curl);
  assert.equal(h.forms[0].fields.get('goal_setup'), 'null');
});

test('responses for another training intention cannot be displayed as the current analysis', async () => {
  const h = backend(general);
  await assert.rejects(h.api.analyseRep({ name: 'rep-123-abcd.mov', uri: 'file:///rep.mov', trainingIntention: curl },
    new AbortController().signal), /did not confirm/);
});

test('feedback does not display invalid numeric measures or verified player claims', () => {
  const result = feedback.parseIntentionFeedback({ shot_type: 'power', focus: 'Power', summary: 'Evidence is incomplete.',
    metrics: [{ key: 'speed', label: 'Speed', status: 'available', value: Infinity, unit: 'm/s' }],
    reference: { note: 'No matched baseline', sources: [{ title: 'Bad', url: 'javascript:alert(1)' }] },
    reported_outcome: { value: 'goal', source: 'player_reported', verified: true } });
  assert.equal(result.metrics[0].status, 'unavailable'); assert.equal(result.metrics[0].value, null);
  assert.equal(result.reported_outcome, null); assert.equal(result.reference.sources.length, 0);
});

test('research cohort records require their protocol and limitation and never create a user rating', () => {
  const cohort = { mean: 83.18, sd: 6.61, unit: 'km/h', sample_size: 17, cohort: 'Study population',
    protocol: 'Radar, fastest of five attempts', statistic: 'Cohort mean and standard deviation',
    limitation: 'Not directly comparable with video transit averages.' };
  const result = feedback.parseIntentionFeedback({ shot_type: 'power', focus: 'Power', summary: 'Unavailable',
    reference: { profiles: [cohort, { ...cohort, limitation: '' }] } });
  assert.equal(result.reference.profiles.length, 1);
  assert.equal(result.reference.status, 'not_comparable');
  assert.equal(Object.hasOwn(result, 'score'), false);
  assert.equal(result.reference.profiles[0].limitation, cohort.limitation);
});

(async () => {
  for (const { name, run } of tests) { await run(); console.log(`PASS ${name}`); }
  console.log(`${tests.length} training-intention checks passed.`);
})().catch(error => { console.error(error); process.exitCode = 1; });
