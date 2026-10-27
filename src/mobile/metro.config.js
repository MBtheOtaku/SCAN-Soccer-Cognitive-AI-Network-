const path = require('node:path');
const fs = require('node:fs');
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);
const existing = config.resolver.blockList;

// Dependency repair backups are not app inputs. Match both Windows and POSIX paths.
config.resolver.blockList = [
  ...(Array.isArray(existing) ? existing : existing ? [existing] : []),
  /(?:^|[\\/])\.expo[\\/](?:sdk\d+[^\\/]*|metro-file-map)(?:[\\/]|$)/,
];

// Keep the watcher index separate from stale shared caches after dependency changes.
config.fileMapCacheDirectory = path.join(__dirname, '.expo', 'metro-file-map');
fs.mkdirSync(config.fileMapCacheDirectory, { recursive: true });

module.exports = config;
