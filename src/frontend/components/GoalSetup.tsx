"use client";
import { PointerEvent, RefObject, useEffect, useRef, useState } from "react";
import { API_BASE } from "../lib/api";
import type { GoalPoint, GoalSetupValue } from "../lib/analysis";
import { validGoalCorners } from "../lib/goalSetup";

async function snapshot(video?: HTMLVideoElement | null): Promise<Blob> {
    if (!video?.videoWidth || video.readyState < 2) throw new Error("Enable the camera and wait for the preview before finding the goal.");
    const scale = Math.min(1, 1280 / video.videoWidth);
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Image capture is unavailable in this browser.");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let visible = 0, samples = 0;
    for (let i = 0; i < pixels.length; i += 64) {
      samples++;
      if (Math.max(pixels[i], pixels[i + 1], pixels[i + 2]) > 12) visible++;
    }
    if (visible / samples < .01) throw new Error("The camera frame is blank. Wait for a visible preview, then retry FIND GOAL.");
    return await new Promise<Blob>((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Could not capture a reference photo.")), "image/jpeg", .9));
}

export default function GoalSetup({ file, videoRef, disabled = false, onChange }: {
  file?: File; videoRef?: RefObject<HTMLVideoElement | null>; disabled?: boolean;
  onChange: (goal: GoalSetupValue | null) => void;
}) {
  const [photo, setPhoto] = useState<{ url: string; blob: Blob } | null>(null);
  const [corners, setCorners] = useState<GoalPoint[]>([]);
  const [candidates, setCandidates] = useState<GoalPoint[][]>([]);
  const [busy, setBusy] = useState(false);
  const [manual, setManual] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [message, setMessage] = useState("");
  const dragging = useRef<number | null>(null);
  const generation = useRef(0);
  const request = useRef<AbortController | null>(null);
  useEffect(() => () => { generation.current++; request.current?.abort(); }, []);
  useEffect(() => () => { if (photo) URL.revokeObjectURL(photo.url); }, [photo]);

  function change(points: GoalPoint[]) { setCorners(points); setConfirmed(false); onChange(null); }

  async function findGoal() {
    const token = ++generation.current;
    request.current?.abort();
    setBusy(true); setMessage(""); setPhoto(null); setCandidates([]); change([]); setManual(false);
    const controller = new AbortController();
    request.current = controller;
    try {
      const form = new FormData();
      if (file) {
        form.append("video", file);
      } else {
        const blob = await snapshot(videoRef?.current);
        if (token !== generation.current) return;
        setPhoto({ blob, url: URL.createObjectURL(blob) });
        form.append("frame", blob, "goal-reference.jpg");
      }
      const timeout = window.setTimeout(() => controller.abort(), file ? 60000 : 15000);
      try {
        const response = await fetch(`${API_BASE}${file ? "/goals/preview" : "/goals/propose"}`, { method: "POST", body: form, signal: controller.signal });
        const data = await response.json().catch(() => null);
        if (!response.ok) throw new Error(typeof data?.detail === "string" ? data.detail : "Automatic goal proposal failed. Mark the corners manually below.");
        if (token !== generation.current) return;
        if (file) {
          if (typeof data?.image_base64 !== "string") throw new Error("No reference image was returned. Restart the backend and retry FIND GOAL.");
          const bytes = Uint8Array.from(atob(data.image_base64), (character) => character.charCodeAt(0));
          const blob = new Blob([bytes], { type: "image/jpeg" });
          setPhoto({ blob, url: URL.createObjectURL(blob) });
        }
        const proposals: GoalPoint[][] = (data?.candidates ?? []).map((c: { corners: GoalPoint[] }) => c.corners).filter(validGoalCorners);
        setCandidates(proposals);
        if (proposals.length) { change(proposals[0]); setMessage("Check all four corners. Drag a numbered handle to adjust, then confirm the goal."); }
        else setMessage("No supported goal outline found. Use MARK CORNERS, then tap top-left, top-right, bottom-right and bottom-left.");
      } finally { window.clearTimeout(timeout); }
    } catch (err) {
      if (token === generation.current) setMessage(controller.signal.aborted ? "Goal preview timed out. Retry FIND GOAL, or skip goal setup for technique analysis." : err instanceof TypeError ? "Cannot reach goal setup. Check the backend connection and retry; a captured photo can still be marked manually." : err instanceof Error ? err.message : "Goal setup failed. Retry or mark the corners manually.");
    } finally { if (token === generation.current) setBusy(false); }
  }

  function point(event: PointerEvent<SVGSVGElement>): GoalPoint {
    const box = event.currentTarget.getBoundingClientRect();
    return [Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)), Math.max(0, Math.min(1, (event.clientY - box.top) / box.height))];
  }

  return <section className="goal-setup">
    <div className="panel-label">GOAL SETUP · CONFIRM ONCE PER CAMERA SETUP</div>
    <button className="secondary-button" disabled={disabled || busy} onClick={findGoal}>{busy ? "FINDING GOAL…" : photo ? "RECAPTURE / FIND GOAL" : "FIND GOAL"}</button>
    <p>{confirmed ? "Goal confirmed for this setup. Reconfirm if you move the camera." : "Goal not confirmed. Analysis can continue, with shot outcome left unknown."}</p>
    {photo && <>
      <div className="goal-setup-image">
        {/* A captured reference frame must retain its natural aspect ratio. */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={photo.url} alt="Goal calibration reference" />
        <svg viewBox="0 0 1 1" preserveAspectRatio="none" role="img" aria-label="Goal corners, numbered top-left clockwise" style={{ touchAction: "none" }}
          onPointerDown={(event) => {
            if (disabled || busy) return;
            if (manual && corners.length < 4 && dragging.current === null) change([...corners, point(event)]);
          }}
          onPointerMove={(event) => {
            if (dragging.current !== null && !disabled && !busy) change(corners.map((p, i) => i === dragging.current ? point(event) : p));
          }}
          onPointerUp={() => { dragging.current = null; }} onPointerCancel={() => { dragging.current = null; }}>
          {corners.length > 1 && <polygon points={corners.map((p) => p.join(",")).join(" ")} fill="rgba(119,255,146,.12)" stroke="#77ff92" strokeWidth=".004" />}
          {corners.map(([x, y], i) => <g key={i} onPointerDown={(event) => { if (disabled || busy) return; event.stopPropagation(); dragging.current = i; event.currentTarget.setPointerCapture(event.pointerId); }}>
            <circle cx={x} cy={y} r=".022" fill="transparent" stroke="transparent" strokeWidth="26" vectorEffect="non-scaling-stroke" pointerEvents="all" />
            <circle cx={x} cy={y} r=".022" fill="#77ff92" />
            <text x={x} y={y + .008} textAnchor="middle" fontSize=".027" fill="#071009" pointerEvents="none">{i + 1}</text>
          </g>)}
        </svg>
      </div>
      {candidates.length > 1 && <label>Goal candidate <select disabled={disabled || busy} defaultValue="0" onChange={(event) => { change(candidates[Number(event.target.value)]); setManual(false); }}>
        {candidates.map((_, i) => <option key={i} value={i}>{i + 1}</option>)}
      </select></label>}
      <div className="live-controls">
        <button className="secondary-button" disabled={disabled || busy} onClick={() => { change([]); setManual(true); setMessage("Tap 1: top-left, 2: top-right, 3: bottom-right, 4: bottom-left. Drag handles to adjust."); }}>MARK CORNERS</button>
        <button className="secondary-button" disabled={disabled || busy || !validGoalCorners(corners)} onClick={() => { onChange({ corners, reference: photo.blob }); setConfirmed(true); setMessage("Confirmed. These corners and this photo will accompany this setup's reps."); }}>CONFIRM GOAL</button>
        <button className="secondary-button" disabled={disabled || busy} onClick={() => { onChange(null); setConfirmed(false); setMessage("Goal setup skipped. Technique analysis remains available."); }}>SKIP GOAL</button>
      </div>
    </>}
    {message && <p role="status">{message}</p>}
    <p className="pressure-note">Automatic outlines are provisional. Confirm the goal mouth, not the surrounding net or fence. 1 TL · 2 TR · 3 BR · 4 BL.</p>
  </section>;
}
