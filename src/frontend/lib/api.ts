import type { AnalysisResponse } from "./analysis";

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000").replace(/\/+$/, "");

export async function analyzeVideo(video: File, signal?: AbortSignal): Promise<AnalysisResponse> {
  const form = new FormData();
  form.append("video", video);
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
  // An explicit null/empty cue in an existing report means policy chose silence.
  if (result.coaching_report) return result.coaching_report.spoken_cue ?? "";
  return result.summary.prototype_feedback?.replace(/^Prototype cue:\s*/i, "") ?? "";
}
