import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { hasPendingSave, listReps, recoverSave, renameRep, saveRep, saveRepDetails, type Rep } from './reps';
import { normalizeIntention, type TrainingIntention } from './training-intention';

type Store = {
  reps: Rep[]; loading: boolean; saving: boolean; pending: boolean; error: string | null;
  refresh: () => Promise<void>; save: (uri: string, intention?: TrainingIntention) => Promise<void>; retry: () => Promise<void>;
  rename: (name: string, title: string) => Promise<void>;
  updateDetails: (name: string, title: string, notes: string, intention?: TrainingIntention) => Promise<void>;
};
const Context = createContext<Store | null>(null);
const describe = (error: unknown) => error instanceof Error ? error.message : 'Could not access your saved reps.';

export function RepProvider({ children }: { children: ReactNode }) {
  const [reps, setReps] = useState<Rep[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const unsaved = useRef<{ uri: string; intention: TrainingIntention } | null>(null);
  const locked = useRef(false);
  const started = useRef(false);
  const refresh = useCallback(async () => {
    const [saved, waiting] = await Promise.all([listReps(), hasPendingSave()]);
    setReps(saved); setPending(waiting || Boolean(unsaved.current));
  }, []);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    void (async () => {
      try { await recoverSave(); } catch (e) { setError(describe(e)); }
      try { await refresh(); } catch (e) { setError(describe(e)); setPending(true); }
      finally { setLoading(false); }
    })();
  }, [refresh]);

  async function persist(uri?: string, intention?: TrainingIntention) {
    if (locked.current || loading) return;
    locked.current = true; setSaving(true); setError(null);
    if (uri) unsaved.current = { uri, intention: normalizeIntention(intention) };
    try {
      if (await hasPendingSave()) await recoverSave();
      else if (unsaved.current) await saveRep(unsaved.current.uri, unsaved.current.intention);
      unsaved.current = null;
      await refresh();
    } catch (e) { setPending(true); setError(describe(e)); throw e; }
    finally { locked.current = false; setSaving(false); }
  }
  async function rename(name: string, value: string) {
    const title = await renameRep(name, value);
    setReps(current => current.map(rep => rep.name === name ? { ...rep, title } : rep));
  }
  async function updateDetails(name: string, title: string, notes: string, intention?: TrainingIntention) {
    const details = await saveRepDetails(name, title, notes, intention);
    setReps(current => current.map(rep => rep.name === name ? { ...rep, ...details } : rep));
  }
  return <Context.Provider value={{ reps, loading, saving, pending, error, refresh, rename, updateDetails,
    save: (uri, intention) => persist(uri, intention), retry: () => persist() }}>{children}</Context.Provider>;
}

export function useReps() {
  const store = useContext(Context);
  if (!store) throw new Error('Saved reps must be used inside RepProvider.');
  return store;
}

export function repDate(name: string) {
  return new Date(Number(name.split('-')[1]));
}
