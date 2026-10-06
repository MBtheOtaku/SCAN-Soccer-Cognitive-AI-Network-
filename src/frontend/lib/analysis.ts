type ScanSummary = {
  total_frames: number;
  valid_pose_frames: number;
  pose_coverage: number;
  probable_striking_leg: string | null;
  probable_strike_frame: number | null;
  probable_strike_time_s: number | null;
  strike_peak_relative_ankle_speed_px_s: number | null;
  strike_torso_lean_deg: number | null;
  strike_left_knee_angle_deg: number | null;
  strike_right_knee_angle_deg: number | null;
  prototype_feedback_code: string | null;
  prototype_feedback: string;
};

interface CoachingReport {
  overall_assessment: string;
  what_went_well: string;
  primary_focus: string | null;
  outcome_analysis: string;
  next_rep_recommendation: string;
  spoken_cue: string | null;
  confidence_note: string | null;
}

export interface AnalysisResponse {
  analysis_id: string;
  filename: string;
  summary: ScanSummary;
  coaching_report: CoachingReport | null;
  artifacts: {
    annotated_video: string;
    metrics_csv: string;
    shot_summary: string;
  };
}

