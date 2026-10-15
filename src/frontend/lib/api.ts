import type { AnalysisResponse, AthleteSide, GoalSetupValue } from "./analysis";

// Browser requests follow the page's origin, including on a phone or hotspot.
// Next forwards these requests to the laptop's local backend.
export const API_BASE = "/api/scan";

export async function analyzeVideo(video: File, signal?: AbortSignal, athleteSide: AthleteSide = "", goal: GoalSetupValue | null = null): Promise<AnalysisResponse> {
  const form = new FormData();
  form.append("video", video);
  if (athleteSide) form.append("athlete_start_side", athleteSide);
  form.append("goal_setup", goal ? JSON.stringify({ confirmed: true, corners: goal.corners }) : "null");
  if (goal) form.append("goal_reference", goal.reference, "goal-reference.jpg");
  const response = await fetch(`${API_BASE}/analyze`, { method: "POST", body: form, signal });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(typeof data?.detail === "string" ? data.detail : `SCAN analysis failed (${response.status}).`);
  }
  if (!data?.analysis_id || !data?.summary || !data?.artifacts) {
    throw new Error("SCAN returned an incomplete analysis response. Please try again.");
  }
  return data as AnalysisResponse;
}

export function spokenCue(result: AnalysisResponse): string {
  // The controller selects one supported pressure cue after the rep, without
  // queuing another technique cue over it. Existing behaviour remains otherwise.
  if (result.pressure_report?.policy.should_speak && result.pressure_report.spoken_cue) {
    return result.pressure_report.spoken_cue;
  }
  // An explicit null/empty cue in an existing report means policy chose silence.
  if (result.coaching_report) return result.coaching_report.spoken_cue ?? "";
  return result.summary.prototype_feedback?.replace(/^Prototype cue:\s*/i, "") ?? "";
}
