import type {} from 'expo/types';
import * as FS from 'expo-file-system/legacy';
import { File } from 'expo-file-system';
import { fetch } from 'expo/fetch';
import type { Rep } from './reps';

export type ReportSection = { label: string; text: string };
export type AnalysisResult = {
  id: string;
  sections: ReportSection[];
  fallback: string | null;
};

function object(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown> : null;
}

function text(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function origin() {
  const setting = process.env.EXPO_PUBLIC_SCAN_API_URL?.trim();
  if (!setting) throw new Error('Set EXPO_PUBLIC_SCAN_API_URL in .env.local and restart Metro.');
  const url = new URL(setting);
  if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash ||
      (url.pathname !== '/' && url.pathname !== '')) {
    throw new Error('Use the HTTPS backend origin without a path or credentials.');
  }
  return url.origin;
}

async function request(
  path: string, options: RequestInit, signal: AbortSignal, timeoutMs: number,
): Promise<Record<string, unknown>> {
  const controller = new AbortController();
  const cancel = () => controller.abort();
  signal.addEventListener('abort', cancel, { once: true });
  if (signal.aborted) controller.abort();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  try {
    const response = await fetch(origin() + path, {
      ...options, signal: controller.signal,
      headers: { Accept: 'application/json' },
    });
    const data = object(await response.json().catch(() => null));
    if (!response.ok) {
      const detail = text(data?.detail);
      throw new Error(detail ? detail.slice(0, 400) : `SCAN returned HTTP ${response.status}.`);
    }
    if (!data) throw new Error('Expected SCAN JSON. Check the backend address and port.');
    return data;
  } catch (error) {
    if (timedOut) throw new Error('Request timed out. Your clip is safe; the server may still finish processing it.');
    if (signal.aborted) throw new Error('Stopped waiting. Your clip is safe; the server may still finish processing it.');
    if (error instanceof TypeError && /network|fetch/i.test(error.message)) {
      throw new Error('Cannot reach SCAN. Check Tailscale, internet access and the laptop backend.');
    }
    throw error;
  } finally {
    clearTimeout(timer);
    signal.removeEventListener('abort', cancel);
  }
}

export async function checkBackend(signal: AbortSignal) {
  const data = await request('/health', {}, signal, 15_000);
  if (data.status !== 'healthy' || data.service !== 'SCAN API') {
    throw new Error('This address did not return the SCAN backend health response.');
  }
}

export async function analyseRep(rep: Rep, signal: AbortSignal): Promise<AnalysisResult> {
  const info = await FS.getInfoAsync(rep.uri);
  if (!info.exists || info.isDirectory || info.size <= 0) {
    throw new Error('This local recording is unavailable. Try playing it first.');
  }
  const form = new FormData();
  // SDK 58's Expo fetch needs a File/Blob, not a legacy { uri, name, type } object.
  form.append('video', new File(rep.uri), rep.name);
  form.append('goal_setup', 'null');
  // Let fetch supply the multipart boundary; do not set Content-Type yourself.
  const data = await request('/analyze', { method: 'POST', body: form }, signal, 600_000);
  const id = text(data.analysis_id);
  const summary = object(data.summary);
  if (!id || !summary || !object(data.artifacts)) {
    throw new Error('SCAN returned an incomplete analysis response. Your local clip is unchanged.');
  }
  const report = object(data.coaching_report);
  const fields = [
    ['overall_assessment', 'Assessment'],
    ['what_went_well', 'What went well'],
    ['primary_focus', 'Focus'],
    ['outcome_analysis', 'Outcome'],
    ['next_rep_recommendation', 'Next rep'],
    ['confidence_note', 'Confidence and limitations'],
  ] as const;
  const sections: ReportSection[] = [];
  for (const [key, label] of fields) {
    const value = text(report?.[key]);
    if (value) sections.push({ label, text: value });
  }
  return {
    id, sections,
    fallback: sections.length ? null : text(summary.prototype_feedback),
  };
}
