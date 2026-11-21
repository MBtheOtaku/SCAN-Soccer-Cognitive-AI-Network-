const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const base = path.resolve(__dirname, '..');
const moduleOutput = { exports: {} };
const code = ts.transpileModule(fs.readFileSync(path.join(base, 'lib/capture-audio.ts'), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
vm.runInNewContext(code, { module: moduleOutput, exports: moduleOutput.exports });
const { captureAudio } = moduleOutput.exports;
const plain = value => JSON.parse(JSON.stringify(value));

// Missing, denied and granted microphone states must never choose an implicit silent fallback.
assert.deepEqual(plain(captureAudio('sound', null)), { canRecord: false, mute: true, label: 'MIC NEEDED' });
assert.deepEqual(plain(captureAudio('sound', { granted: false })), { canRecord: false, mute: true, label: 'MIC NEEDED' });
assert.deepEqual(plain(captureAudio('sound', { granted: true })), { canRecord: true, mute: false, label: 'SOUND ON' });
assert.deepEqual(plain(captureAudio('video-only', { granted: false })), { canRecord: true, mute: true, label: 'VIDEO ONLY' });
assert.deepEqual(plain(captureAudio('video-only', { granted: true })), { canRecord: true, mute: true, label: 'VIDEO ONLY' });

const config = JSON.parse(fs.readFileSync(path.join(base, 'app.json'), 'utf8'));
const cameraPlugin = config.plugins.find(plugin => Array.isArray(plugin) && plugin[0] === 'expo-camera');
assert.equal(cameraPlugin[1].recordAudioAndroid, true);
assert.ok(cameraPlugin[1].microphonePermission.includes('sound'));
console.log('7 capture-audio permission/config checks passed.');
