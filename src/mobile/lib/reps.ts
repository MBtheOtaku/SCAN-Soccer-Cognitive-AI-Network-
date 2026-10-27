import * as FS from 'expo-file-system/legacy';

export type RepDetails = { title?: string; notes?: string };
export type Rep = { name: string; uri: string } & RepDetails;
export const REP_TITLE_LIMIT = 80;
export const REP_NOTES_LIMIT = 500;
const repNamePattern = /^rep-\d+-[a-z0-9]+\.(mov|mp4)$/;
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
    .filter(name => repNamePattern.test(name)).sort().reverse();
  return Promise.all(names.map(async name => ({ name, uri: root() + name, ...await readDetails(name) })));
}

async function readDetails(name: string): Promise<RepDetails> {
  // Optional metadata must never prevent an existing video from being listed.
  try {
    const metadata = JSON.parse(await FS.readAsStringAsync(root() + name + '.title.json'));
    const text = (value: unknown, limit: number) => typeof value === 'string' && value.trim().length <= limit
      ? value.trim() || undefined : undefined;
    return { title: text(metadata.title, REP_TITLE_LIMIT), notes: text(metadata.notes, REP_NOTES_LIMIT) };
  } catch { return {}; }
}

export async function renameRep(name: string, value: string): Promise<string | undefined> {
  if (!repNamePattern.test(name)) throw new Error('Choose a saved rep to rename.');
  const previous = await readDetails(name);
  return (await saveRepDetails(name, value, previous.notes || '')).title;
}

export async function saveRepDetails(name: string, titleValue: string, notesValue: string): Promise<RepDetails> {
  if (!repNamePattern.test(name)) throw new Error('Choose a saved rep to rename.');
  const title = titleValue.trim();
  const notes = notesValue.trim();
  if (title.length > REP_TITLE_LIMIT) throw new Error(`Use ${REP_TITLE_LIMIT} characters or fewer.`);
  if (notes.length > REP_NOTES_LIMIT) throw new Error(`Keep notes to ${REP_NOTES_LIMIT} characters or fewer.`);
  if (!await size(root() + name)) throw new Error('This saved rep is no longer available.');
  const metadata = root() + name + '.title.json';
  if (!title && !notes) {
    await FS.deleteAsync(metadata, { idempotent: true });
    return {};
  }
  // Only metadata is replaced. Video names/URIs and their timestamps stay intact.
  const temporary = metadata + '.tmp';
  await FS.writeAsStringAsync(temporary, JSON.stringify({ title, notes }));
  await FS.moveAsync({ from: temporary, to: metadata });
  return { title: title || undefined, notes: notes || undefined };
}
