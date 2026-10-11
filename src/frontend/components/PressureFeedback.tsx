import type { AthleteSide, PressureReport } from "../lib/analysis";

export function PressureSetup({ value, onChange, disabled = false }: {
  value: AthleteSide; onChange: (value: AthleteSide) => void; disabled?: boolean;
}) {
  return <label className="pressure-setup">
    <span>Optional opponent-pressure review · athlete position at clip start</span>
    <select value={value} disabled={disabled} onChange={(event) => onChange(event.target.value as AthleteSide)}>
      <option value="">Off — technique analysis only</option>
      <option value="left">Athlete on camera-left</option>
      <option value="right">Athlete on camera-right</option>
    </select>
    <small>Use a stationary camera with exactly two players separated at the start. Recheck this selection for every rep. Left/right refers to the camera image.</small>
  </label>;
}

export default function PressureFeedback({ report }: { report?: PressureReport | null }) {
  if (!report) return null;
  return <section className="feedback-panel pressure-feedback">
    <div className="panel-label">OPPONENT PRESSURE · POST-REP REVIEW</div>
    <p>{report.assessment}</p>
    <p><strong>Next rep: </strong>{report.next_rep_recommendation}</p>
    <p className="pressure-note">{report.confidence_note}</p>
  </section>;
}
