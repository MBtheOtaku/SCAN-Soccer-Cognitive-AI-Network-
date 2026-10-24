/* global __dirname */
// Metro's CI mode skips its filesystem watcher. Keep this opt-in: source changes
// require a server restart, but file-sync events cannot crash the watcher.
const { spawn } = require('node:child_process');
const { networkInterfaces } = require('node:os');
const path = require('node:path');

const projectRoot = path.resolve(__dirname, '..');
const cli = path.join(path.dirname(require.resolve('expo/package.json')), 'bin', 'cli');
const extra = process.argv.slice(2);
const portFlag = extra.indexOf('--port');
const port = portFlag >= 0 ? Number(extra[portFlag + 1]) : 8081;
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Use a valid --port number.');

console.log('\nSCAN stable preview — automatic reload is disabled.');
console.log('After editing code: Ctrl+C, then run npm run start:stable again.');
console.log('After Metro says it is ready, open the Wi-Fi address below in Expo Go:');
for (const [name, addresses] of Object.entries(networkInterfaces())) {
  if (/wsl|hyper-v|vethernet|docker|loopback/i.test(name)) continue;
  for (const address of addresses ?? []) {
    if (address.family === 'IPv4' && !address.internal) {
      console.log(`  ${name}: exp://${address.address}:${port}`);
    }
  }
}
console.log('Your phone and laptop must still be able to reach each other.\n');

const server = spawn(process.execPath, [cli, 'start', '--go', ...extra], {
  cwd: projectRoot, stdio: 'inherit', windowsHide: true, env: { ...process.env, CI: '1' },
});
server.on('error', error => { console.error(error.message); process.exitCode = 1; });
server.on('exit', code => { process.exitCode = code ?? 0; });
process.on('SIGINT', () => server.kill('SIGINT'));
process.on('SIGTERM', () => server.kill('SIGTERM'));
