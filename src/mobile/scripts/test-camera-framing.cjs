const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const output = { exports: {} };
const source = path.resolve(__dirname, '../lib/camera-framing.ts');
vm.runInNewContext(ts.transpileModule(fs.readFileSync(source, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText, { module: output, exports: output.exports });
const { cameraPreviewSize, clampCameraZoom, framingLenses, standardLens, ultraWideLens } = output.exports;

// Every profile records 16:9; a portrait preview must preserve 9:16 on both
// small phones and tall phones, including the expanded view.
for (const [width, height] of [[320, 568], [390, 844], [430, 932], [768, 1024]]) {
  const compact = cameraPreviewSize(width, height, false);
  const expanded = cameraPreviewSize(width, height, true);
  assert.ok(Math.abs(compact.width / compact.height - 9 / 16) < 1e-10);
  assert.ok(Math.abs(expanded.width / expanded.height - 9 / 16) < 1e-10);
  assert.ok(compact.width <= width - 48 && expanded.width <= width - 48 + 1e-10);
  assert.ok(expanded.height >= compact.height);
}
assert.equal(cameraPreviewSize(390, 844, true).width, 342);
assert.ok(cameraPreviewSize(390, 844, false).height > (390 - 48) * 3 / 4);
assert.equal(clampCameraZoom(-.1), 0);
assert.equal(clampCameraZoom(1.1), 1);
assert.equal(clampCameraZoom(NaN), 0);
let zoom = 0;
for (let index = 0; index < 20; index++) zoom = clampCameraZoom(zoom + .05);
assert.equal(zoom, 1);
for (let index = 0; index < 20; index++) zoom = clampCameraZoom(zoom - .05);
assert.equal(zoom, 0);

// Never offer an unreported lens or infer support from localized display names.
const plain = value => JSON.parse(JSON.stringify(value));
assert.deepEqual(plain(framingLenses([])), [{ id: '', label: 'Default' }]);
assert.deepEqual(plain(framingLenses([{ deviceType: 'unsupported', localizedName: 'Ultra Wide' }])), [{ id: '', label: 'Default' }]);
const choices = framingLenses([{ deviceType: ultraWideLens }, { deviceType: standardLens }, { deviceType: ultraWideLens }]);
assert.deepEqual(plain(choices), [{ id: '', label: 'Default' }, { id: standardLens, label: 'Standard' }, { id: ultraWideLens, label: 'Ultra wide' }]);
console.log('Camera framing checks passed: portrait geometry, device zoom bounds and hardware-gated lens choices.');
