import { shotTypes, type ShotType } from './training-intention';

export type IntentionMetric = {
  key: string; label: string; status: 'available' | 'unavailable'; value: number | null;
  unit: string | null; formula: string | null; reason: string | null;
};
export type IntentionFeedback = {
  shot_type: ShotType; focus: string; summary: string; metrics: IntentionMetric[];
  reference: { status: 'not_comparable'; note: string; sources: { title: string; url: string }[];
    profiles: { mean: number; sd: number; unit: string; sample_size: number; cohort: string;
      protocol: string; statistic: string; limitation: string }[] };
  reported_outcome: { value: string; source: 'player_reported'; verified: false } | null;
};

const object = (value: unknown): Record<string, unknown> | null => value !== null && typeof value === 'object' && !Array.isArray(value)
  ? value as Record<string, unknown> : null;
const text = (value: unknown): string | null => typeof value === 'string' && value.trim() ? value.trim() : null;

export function parseIntentionFeedback(value: unknown): IntentionFeedback | null {
  const data = object(value);
  if (!data || !shotTypes.includes(data.shot_type as ShotType) || !text(data.focus) || !text(data.summary)) return null;
  const metrics: IntentionMetric[] = [];
  for (const raw of Array.isArray(data.metrics) ? data.metrics : []) {
    const metric = object(raw);
    if (!metric || !text(metric.key) || !text(metric.label)) continue;
    const available = metric.status === 'available' && typeof metric.value === 'number' && Number.isFinite(metric.value);
    metrics.push({ key: metric.key as string, label: metric.label as string, status: available ? 'available' : 'unavailable',
      value: available ? metric.value as number : null, unit: text(metric.unit), formula: text(metric.formula),
      reason: text(metric.reason) || (available ? null : 'The required evidence is not available for this clip.') });
  }
  const reference = object(data.reference);
  const sources: { title: string; url: string }[] = [];
  for (const raw of Array.isArray(reference?.sources) ? reference.sources : []) {
    const source = object(raw);
    const title = text(source?.title), url = text(source?.url);
    if (title && url && /^https:\/\//i.test(url)) sources.push({ title, url });
  }
  const reported = object(data.reported_outcome);
  const profiles: IntentionFeedback['reference']['profiles'] = [];
  for (const raw of Array.isArray(reference?.profiles) ? reference.profiles : []) {
    const profile = object(raw);
    if (!profile || typeof profile.mean !== 'number' || !Number.isFinite(profile.mean) ||
        typeof profile.sd !== 'number' || !Number.isFinite(profile.sd) || profile.sd < 0 ||
        typeof profile.sample_size !== 'number' || !Number.isSafeInteger(profile.sample_size) || profile.sample_size <= 0 ||
        !text(profile.unit) || !text(profile.cohort) || !text(profile.protocol) || !text(profile.statistic) || !text(profile.limitation)) continue;
    profiles.push({ mean: profile.mean, sd: profile.sd, unit: profile.unit as string, sample_size: profile.sample_size,
      cohort: profile.cohort as string, protocol: profile.protocol as string, statistic: profile.statistic as string,
      limitation: profile.limitation as string });
  }
  return {
    shot_type: data.shot_type as ShotType, focus: data.focus as string, summary: data.summary as string, metrics,
    reference: { status: 'not_comparable', note: text(reference?.note) || 'This clip has no validated professional comparison.', sources, profiles },
    reported_outcome: text(reported?.value) && reported?.source === 'player_reported' && reported?.verified === false
      ? { value: reported.value as string, source: 'player_reported', verified: false } : null,
  };
}
