import * as FS from 'expo-file-system/legacy';

export type Rep = { name: string; uri: string };
type Job = { name: string; cachePath: string; bytes: number };

function root() {
  if (!FS.documentDirectory) throw new Error('Local storage is unavailable.');
  return FS.documentDirectory + 'scan-reps/';
}

async function prepare() {
  await FS.makeDirectoryAsync(root(), { intermediates: true });
}

async function size(uri: string) {
  const info = await FS.getInfoAsync(uri);
  return info.exists && !info.isDirectory ? info.size : 0;
}

export async function hasPendingSave() {
  await prepare();
  return (await FS.getInfoAsync(root() + 'pending.json')).exists;
}

// Retry an interrupted copy before accepting another recording.
export async function recoverSave() {
  await prepare();
  const journal = root() + 'pending.json';
  if (!(await FS.getInfoAsync(journal)).exists) return;
  const job: Job = JSON.parse(await FS.readAsStringAsync(journal));
  if (!job || !/^rep-\d+-[a-z0-9]+\.(mov|mp4)$/.test(job.name) ||
      typeof job.cachePath !== 'string' || !job.cachePath ||
      job.cachePath.includes('..') || job.cachePath.startsWith('/') ||
      job.cachePath.includes(':') || job.cachePath.includes('\\') ||
      !Number.isFinite(job.bytes) || job.bytes <= 0 || !FS.cacheDirectory) {
    throw new Error('The pending recording needs storage recovery. Keep the app installed.');
  }
  const target = root() + job.name;
  const staging = target + '.partial';
  if (await size(target) !== job.bytes) {
    if (await size(staging) !== job.bytes) {
      const source = FS.cacheDirectory + job.cachePath;
      if (await size(source) !== job.bytes) {
        throw new Error('The unfinished recording is unavailable. Previously saved reps are still kept.');
      }
      await FS.deleteAsync(staging, { idempotent: true });
      await FS.copyAsync({ from: source, to: staging });
      if (await size(staging) !== job.bytes) throw new Error('Save incomplete. Free storage, then retry.');
    }
    await FS.moveAsync({ from: staging, to: target });
  }
  await FS.deleteAsync(journal, { idempotent: true });
  // The verified Documents copy must exist before removing the cached source.
  await FS.deleteAsync(FS.cacheDirectory + job.cachePath, { idempotent: true }).catch(() => {});
}

export async function saveRep(uri: string) {
  await prepare();
  if (await hasPendingSave()) throw new Error('Retry the previous save first.');
  const bytes = await size(uri);
  if (!bytes || !FS.cacheDirectory || !uri.startsWith(FS.cacheDirectory)) {
    throw new Error('The camera did not return a usable local recording.');
  }
  const ext = uri.toLowerCase().endsWith('.mov') ? 'mov' : 'mp4';
  const job: Job = {
    name: `rep-${Date.now()}-${Math.random().toString(36).slice(2, 10)}.${ext}`,
    cachePath: uri.slice(FS.cacheDirectory.length), bytes,
  };
  const temporaryJournal = root() + 'pending.tmp';
  await FS.writeAsStringAsync(temporaryJournal, JSON.stringify(job));
  await FS.moveAsync({ from: temporaryJournal, to: root() + 'pending.json' });
  await recoverSave();
}

export async function listReps(): Promise<Rep[]> {
  await prepare();
  const names = (await FS.readDirectoryAsync(root()))
    .filter(name => /^rep-\d+-[a-z0-9]+\.(mov|mp4)$/.test(name)).sort().reverse();
  return names.map(name => ({ name, uri: root() + name }));
}
