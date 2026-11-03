import { useEffect, useRef, useState } from 'react';
import { Text, View } from 'react-native';
import { Action, BallProgress } from './controls';
import { analyseRep, checkBackend, type AnalysisResult } from '../lib/backend';
import type { Rep } from '../lib/reps';
import { ui } from '../lib/theme';

export function RepAnalysis({ rep }: { rep: Rep }) {
  const [phase, setPhase] = useState<'idle' | 'checking' | 'analysing'>('idle');
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const pending = useRef<AbortController | null>(null);
  const mounted = useRef(false);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; pending.current?.abort(); };
  }, []);

  async function run(analyse: boolean) {
    if (pending.current) return;
    const controller = new AbortController();
    pending.current = controller;
    setError(null); setStatus(null); setPhase('checking');
    if (analyse) setResult(null);
    try {
      await checkBackend(controller.signal);
      if (controller.signal.aborted || !mounted.current) return;
      if (analyse) {
        setPhase('analysing');
        const response = await analyseRep(rep, controller.signal);
        if (!controller.signal.aborted && mounted.current) setResult(response);
      } else setStatus('Connected to SCAN.');
    } catch (e) {
      if (mounted.current) setError(e instanceof Error ? e.message : 'The request failed. Your clip is safe.');
    } finally {
      pending.current = null;
      if (mounted.current) setPhase('idle');
    }
  }

  const busy = phase !== 'idle';
  return <View style={ui.card}>
    <Text style={ui.badge}>BACKEND ANALYSIS</Text>
    <Text style={ui.body}>Analyse this saved video on your laptop. Keep SCAN open while waiting.</Text>
    <Text style={ui.small}>Goal calibration is not connected to native uploads yet, so shot outcome remains unknown. Opponent-pressure review is off for this first version.</Text>
    <Action title="Check backend connection" secondary disabled={busy}
      onPress={() => { void run(false); }} />
    <Action title={result ? 'Analyse again' : error ? 'Retry analysis' : 'Analyse rep'} disabled={busy}
      onPress={() => { void run(true); }} />
    {busy && <>
      <BallProgress label={phase === 'checking' ? 'Checking backend…' : 'Sending video and analysing…'} />
      <Action title="Stop waiting" secondary onPress={() => pending.current?.abort()} />
    </>}
    {status && <Text accessibilityLiveRegion="polite" style={ui.body}>{status}</Text>}
    {error && <Text accessibilityRole="alert" style={ui.error}>{error}</Text>}
    {result && <View style={{ gap: 16 }}>
      <Text style={ui.heading}>Session report</Text>
      <Text style={ui.small}>Analysis ID: {result.id}</Text>
      {result.sections.map(section => <View key={section.label} style={{ gap: 6 }}>
        <Text style={ui.heading}>{section.label}</Text>
        <Text style={ui.body}>{section.text}</Text>
      </View>)}
      {!result.sections.length && <Text style={ui.body}>{result.fallback || 'Analysis completed, but no written coaching report was returned.'}</Text>}
    </View>}
    <Text style={ui.small}>Your original stays on this phone. Stop waiting or leaving this screen does not stop a job already running on the server.</Text>
  </View>;
}