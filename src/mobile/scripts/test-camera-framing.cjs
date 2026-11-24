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
const { cameraPreviewSize, cameraZoomTarget, cameraZoomPosition, clampCameraZoom, framingLenses, standardLens, ultraWideLens, widestFramingLens } = output.exports;

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
// Open at the true minimum on a reported ultra-wide lens, including phones
// that expose ultra-wide without a separate standard lens in the inventory.
assert.equal(widestFramingLens([]), undefined);
assert.equal(widestFramingLens([{ deviceType: standardLens }]), undefined);
assert.equal(widestFramingLens([{ deviceType: 'unsupported', localizedName: 'Ultra Wide' }]), undefined);
assert.equal(widestFramingLens([{ deviceType: ultraWideLens }]), ultraWideLens);
assert.equal(widestFramingLens([{ deviceType: standardLens }, { deviceType: ultraWideLens }]), ultraWideLens);
assert.equal(cameraZoomTarget(0), 0);
assert.equal(cameraZoomTarget(1), 1);
assert.equal(cameraZoomPosition(0), 0);
assert.equal(cameraZoomPosition(1), 1);
// Half the bar uses only one eighth of native zoom range, leaving substantially
// more control near minimum. Small changes must not round to a dead zone.
assert.ok(cameraZoomTarget(.5) < .15);
assert.ok(cameraZoomTarget(.01) > 0);
let previous = 0;
for (let index = 0; index <= 100; index++) {
  const position = index / 100;
  const target = cameraZoomTarget(position);
  assert.ok(target >= previous && target <= 1);
  assert.ok(Math.abs(cameraZoomPosition(target) - position) < 1e-10);
  previous = target;
}
for (const invalid of [NaN, Infinity, -Infinity, -1]) {
  assert.equal(cameraZoomTarget(invalid), 0);
  assert.equal(cameraZoomPosition(invalid), 0);
}
assert.equal(cameraZoomTarget(2), 1);
console.log('Camera framing checks passed: portrait geometry, reported wide-lens default and precise wide-end slider range.');
