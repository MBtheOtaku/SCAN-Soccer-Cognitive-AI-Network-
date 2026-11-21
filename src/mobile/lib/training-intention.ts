export const shotTypes = ['unspecified', 'power', 'curl', 'placement'] as const;
export const targetZones = ['upper_left', 'upper_center', 'upper_right', 'middle_left', 'center', 'middle_right',
  'lower_left', 'lower_center', 'lower_right'] as const;
export const reportedOutcomes = ['goal', 'miss', 'woodwork', 'saved'] as const;

export type ShotType = typeof shotTypes[number];
export type TargetZone = typeof targetZones[number];
export type ReportedOutcome = typeof reportedOutcomes[number];
export type TrainingIntention = {
  shot_type: ShotType;
  target_zone: TargetZone | null;
  reported_outcome: ReportedOutcome | null;
};

export const shotLabels: Record<ShotType, string> = { unspecified: 'General', power: 'Power', curl: 'Curl', placement: 'Placement' };
export const targetLabels: Record<TargetZone, string> = {
  upper_left: 'Top left', upper_center: 'Top centre', upper_right: 'Top right',
  middle_left: 'Middle left', center: 'Centre', middle_right: 'Middle right',
  lower_left: 'Bottom left', lower_center: 'Bottom centre', lower_right: 'Bottom right',
};
export const outcomeLabels: Record<ReportedOutcome, string> = { goal: 'Goal', miss: 'Miss', woodwork: 'Woodwork', saved: 'Saved' };

function member<T extends string>(values: readonly T[], value: unknown): value is T {
  return typeof value === 'string' && values.includes(value as T);
}

// Old recordings have no intention metadata. Malformed optional fields cannot hide a saved video.
export function normalizeIntention(value?: unknown): TrainingIntention {
  const data = value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown> : {};
  return {
    shot_type: member(shotTypes, data.shot_type) ? data.shot_type : 'unspecified',
    target_zone: member(targetZones, data.target_zone) ? data.target_zone : null,
    reported_outcome: member(reportedOutcomes, data.reported_outcome) ? data.reported_outcome : null,
  };
}

export function intentionKey(value?: unknown): string {
  return JSON.stringify(normalizeIntention(value));
}

export function intentionLabel(value?: unknown): string {
  const intention = normalizeIntention(value);
  return shotLabels[intention.shot_type] + (intention.target_zone ? ` · ${targetLabels[intention.target_zone]}` : '');
}
