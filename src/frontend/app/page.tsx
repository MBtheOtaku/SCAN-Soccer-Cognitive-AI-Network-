"use client";

import { ChangeEvent, useEffect, useRef, useState } from "react";
import {
  initializeSpeech,
  speak,
  stopSpeaking,
} from "../lib/speech";

const API_BASE = "http://127.0.0.1:8000";
const DEFAULT_ROI = "540,280,1180,650";

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
  prototype_feedback_code: string;
  prototype_feedback: string;
};

type AnalysisResponse = {
  analysis_id: string;
  filename: string;
  summary: ScanSummary;
  artifacts: {
    annotated_video: string;
    metrics_csv: string;
    shot_summary: string;
  };
};

const processingStages = [
  "VIDEO INGESTED",
  "PLAYER DETECTED",
  "TRACKING POSE",
  "IDENTIFYING STRIKE",
  "ANALYZING TECHNIQUE",
  "GENERATING FEEDBACK",
];

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeStage, setActiveStage] = useState(0);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [voiceEnabled, setVoiceEnabled] = useState(true);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const hasWelcomed = useRef(false);

  useEffect(() => {
    initializeSpeech();

    return () => {
      stopSpeaking();
    };
  }, []);

  const handleFirstInteraction = () => {
    if (!voiceEnabled || hasWelcomed.current) {
      return;
    }

    hasWelcomed.current = true;

    speak(
      "SCAN online. Ready for training analysis."
    );
  };

  const handleFileChange = (
    event: ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0] ?? null;

    setSelectedFile(file);
    setResult(null);
    setError(null);

    if (file && voiceEnabled) {
      speak(
        "Video acquired. Ready to calibrate."
      );
    }
  };

  const openFilePicker = () => {
    if (!isAnalyzing) {
      fileInputRef.current?.click();
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setResult(null);
    setError(null);
    setActiveStage(0);

    if (voiceEnabled) {
      speak(
        "Calibration complete. Beginning session analysis."
      );
    }

    /*
      Temporary UI progression.

      Right now FastAPI returns only when analysis is complete,
      so these stages are visual rather than real backend events.

      Later we can replace this with WebSockets / server events.
    */
    const stageTimer = window.setInterval(() => {
      setActiveStage((current) =>
        Math.min(current + 1, processingStages.length - 1)
      );
    }, 900);

    try {
      const formData = new FormData();

      formData.append("video", selectedFile);
      formData.append("roi", DEFAULT_ROI);

      const response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ?? "SCAN analysis failed."
        );
      }

      setActiveStage(processingStages.length - 1);

      // Brief pause so the final processing stage is visible.
      await new Promise((resolve) =>
        window.setTimeout(resolve, 600)
      );

      const analysisResult = data as AnalysisResponse;

      setResult(analysisResult);

      const spokenFeedback =
        analysisResult.summary.prototype_feedback.replace(
          /^Prototype cue:\s*/i,
          ""
        );

      if (voiceEnabled) {
        speak(
          `Analysis complete. ${spokenFeedback}`
        );
      }
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred.");
      }
    } finally {
      window.clearInterval(stageTimer);
      setIsAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setSelectedFile(null);
    setResult(null);
    setError(null);
    setActiveStage(0);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <main className="scan-page" onPointerDown={handleFirstInteraction}>
      {/* Background pitch */}
      <div className="pitch-background" aria-hidden="true">
        <div className="pitch-half-line" />
        <div className="pitch-center-circle" />
        <div className="pitch-center-dot" />
      </div>

      <div className="scan-line" aria-hidden="true" />

      {/* Header */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">S</div>

          <div>
            <div className="brand-name">SCAN</div>
            <div className="brand-subtitle">
              Soccer Cognitive AI Network
            </div>
          </div>
        </div>

        <div className="header-controls">
        <button
          className="voice-toggle"
          onClick={(event) => {
            event.stopPropagation();

            setVoiceEnabled((current) => {
              const next = !current;

              if (!next) {
                stopSpeaking();
              }

              return next;
            });
          }}
        >
          {voiceEnabled ? "◉ VOICE ON" : "○ VOICE OFF"}
        </button>

        
        <div className="system-status">
          <span className="status-dot" />
          SYSTEM ONLINE
        </div>
      </div>
      </header>

      {/* PROCESSING SCREEN */}
      {isAnalyzing && (
        <section className="processing-screen">
          <div className="processing-ball">
            <div className="processing-ball-inner" />
          </div>

          <div className="processing-eyebrow">
            SCAN ANALYSIS ENGINE
          </div>

          <h2>ANALYZING SESSION</h2>

          <p>{selectedFile?.name}</p>

          <div className="processing-stages">
            {processingStages.map((stage, index) => {
              const complete = index < activeStage;
              const active = index === activeStage;

              return (
                <div
                  key={stage}
                  className={`processing-stage ${
                    complete ? "stage-complete" : ""
                  } ${active ? "stage-active" : ""}`}
                >
                  <span className="stage-indicator">
                    {complete ? "✓" : active ? "●" : "○"}
                  </span>

                  <span>{stage}</span>
                </div>
              );
            })}
          </div>

          <div className="processing-bar">
            <div
              className="processing-bar-fill"
              style={{
                width: `${
                  ((activeStage + 1) /
                    processingStages.length) *
                  100
                }%`,
              }}
            />
          </div>
        </section>
      )}

      {/* RESULTS SCREEN */}
      {!isAnalyzing && result && (
        <section className="results-page">
          <div className="results-heading">
            <div>
              <div className="eyebrow">
                ANALYSIS COMPLETE
              </div>

              <h1>SESSION REPORT</h1>

              <p>
                Analysis ID: {result.analysis_id}
              </p>
            </div>

            <button
              className="secondary-button"
              onClick={resetAnalysis}
            >
              ANALYZE ANOTHER VIDEO
            </button>
          </div>

          <div className="results-grid">
            {/* Annotated video */}
            <div className="result-panel video-panel">
              <div className="panel-label">
                COMPUTER VISION OUTPUT
              </div>

              <video
                className="result-video"
                controls
                playsInline
                src={`${API_BASE}${result.artifacts.annotated_video}`}
              />
            </div>

            {/* Summary */}
            <div className="result-panel metrics-panel">
              <div className="panel-label">
                REP SUMMARY
              </div>

              <div className="primary-stat">
                <span>POSE COVERAGE</span>
                <strong>
                  {Math.round(
                    result.summary.pose_coverage * 100
                  )}
                  %
                </strong>
              </div>

              <div className="metric-list">
                <Metric
                  label="STRIKING LEG"
                  value={
                    result.summary.probable_striking_leg
                      ?.toUpperCase() ?? "—"
                  }
                />

                <Metric
                  label="STRIKE TIME"
                  value={
                    result.summary.probable_strike_time_s !==
                    null
                      ? `${result.summary.probable_strike_time_s.toFixed(
                          2
                        )} s`
                      : "—"
                  }
                />

                <Metric
                  label="STRIKE FRAME"
                  value={
                    result.summary.probable_strike_frame ??
                    "—"
                  }
                />

                <Metric
                  label="TORSO LEAN"
                  value={
                    result.summary.strike_torso_lean_deg !==
                    null
                      ? `${result.summary.strike_torso_lean_deg.toFixed(
                          1
                        )}°`
                      : "—"
                  }
                />

                <Metric
                  label="LEFT KNEE"
                  value={
                    result.summary
                      .strike_left_knee_angle_deg !== null
                      ? `${result.summary.strike_left_knee_angle_deg.toFixed(
                          1
                        )}°`
                      : "—"
                  }
                />

                <Metric
                  label="RIGHT KNEE"
                  value={
                    result.summary
                      .strike_right_knee_angle_deg !== null
                      ? `${result.summary.strike_right_knee_angle_deg.toFixed(
                          1
                        )}°`
                      : "—"
                  }
                />
              </div>
            </div>
          </div>

          {/* Feedback */}
          <div className="feedback-panel">
            <div className="feedback-header">
              <span className="feedback-icon">S</span>

              <div>
                <div className="panel-label">
                  SCAN FEEDBACK
                </div>

                <div className="feedback-type">
                  {result.summary.prototype_feedback_code
                    .replaceAll("_", " ")
                    .toUpperCase()}
                </div>
              </div>
            </div>

            <p>
              {result.summary.prototype_feedback}
            </p>

            <div className="prototype-warning">
              PROTOTYPE OUTPUT — TECHNICAL THRESHOLDS ARE
              NOT YET VALIDATED SOCCER BIOMECHANICS
            </div>
          </div>
        </section>
      )}

      {/* LANDING SCREEN */}
      {!isAnalyzing && !result && (
        <section className="hero">
          <div className="eyebrow">
            AI-POWERED TRAINING INTELLIGENCE
          </div>

          <h1>
            TURN EVERY REP
            <span> INTO INFORMATION.</span>
          </h1>

          <p className="hero-description">
            Upload a soccer training clip and let SCAN
            analyze movement, technique, and player state
            through computer vision.
          </p>

          <div
            className={`upload-panel ${
              selectedFile
                ? "upload-panel-selected"
                : ""
            }`}
            onClick={openFilePicker}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/quicktime,.mov,.mp4"
              onChange={handleFileChange}
              hidden
            />

            <div className="upload-icon">
              {selectedFile ? "✓" : "+"}
            </div>

            {selectedFile ? (
              <>
                <div className="upload-title">
                  VIDEO READY
                </div>

                <div className="file-name">
                  {selectedFile.name}
                </div>

                <div className="upload-hint">
                  Click to choose a different video
                </div>
              </>
            ) : (
              <>
                <div className="upload-title">
                  DROP TRAINING VIDEO
                </div>

                <div className="upload-hint">
                  or click to browse
                </div>

                <div className="file-types">
                  MP4 / MOV
                </div>
              </>
            )}
          </div>

          <button
            className="analyze-button"
            disabled={!selectedFile}
            onClick={handleAnalyze}
          >
            <span>ANALYZE SESSION</span>
            <span className="button-arrow">→</span>
          </button>

          {error && (
            <div className="analysis-error">
              {error}
            </div>
          )}

          <div className="pipeline-preview">
            <span>POSE TRACKING</span>
            <div className="pipeline-line" />
            <span>STRIKE DETECTION</span>
            <div className="pipeline-line" />
            <span>ADAPTIVE FEEDBACK</span>
          </div>
        </section>
      )}

      <footer className="footer">
        <span>SCAN // PROTOTYPE BUILD</span>
        <span>
          COMPUTER VISION · COGNITIVE AI · FOOTBALL
        </span>
      </footer>
    </main>
  );
}

function Metric({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="metric-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}