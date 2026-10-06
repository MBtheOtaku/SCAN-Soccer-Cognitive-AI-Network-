"use client";

import { useEffect, useRef, useState } from "react";
import { analyzeVideo, spokenCue } from "../lib/api";
import type { AnalysisResponse } from "../lib/analysis";
import { speak, stopSpeaking } from "../lib/speech";

type Phase = "READY" | "RECORDING" | "ANALYZING" | "REP COMPLETE";
const REP_MS = 4500;
const MIME_TYPES = ["video/mp4;codecs=avc1.42E01E", "video/mp4", "video/webm;codecs=vp8", "video/webm;codecs=vp9", "video/webm"];

export default function LiveSession({ voiceEnabled }: { voiceEnabled: boolean }) {
  const [open, setOpen] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [phase, setPhase] = useState<Phase>("READY");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [reps, setReps] = useState(0);
  const preview = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const request = useRef<AbortController | null>(null);
  const generation = useRef(0);
  const busy = useRef(false);
  const voice = useRef(voiceEnabled);

  useEffect(() => { voice.current = voiceEnabled; }, [voiceEnabled]);

  function release() {
    generation.current += 1;
    if (timer.current) clearTimeout(timer.current);
    request.current?.abort();
    const current = recorder.current;
    if (current) {
      current.onstop = null;
      current.onerror = null;
      if (current.state !== "inactive") current.stop();
    }
    stream.current?.getTracks().forEach((track) => { track.onended = null; track.stop(); });
    stream.current = null;
    recorder.current = null;
    busy.current = false;
    stopSpeaking();
  }

  useEffect(() => () => { release(); }, []);

  async function enableCamera() {
    if (busy.current) return;
    busy.current = true;
    setConnecting(true);
    setError(null);
    const token = ++generation.current;
    try {
      if (!window.isSecureContext) throw new Error("Camera access requires trusted HTTPS on your phone (or localhost on this laptop). See docs/live-session.md.");
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access is unavailable in this browser.");
      if (typeof MediaRecorder === "undefined") throw new Error("Video recording is unsupported. Try a current Safari or Chrome browser.");
      const acquired = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } } });
      if (token !== generation.current) { acquired.getTracks().forEach((track) => track.stop()); return; }
      stream.current = acquired;
      acquired.getVideoTracks().forEach((track) => {
        track.onended = () => {
          release();
          setCameraReady(false);
          setPhase("READY");
          setError("Camera disconnected. Enable the camera again to continue.");
        };
      });
      if (preview.current) {
        preview.current.srcObject = acquired;
        await preview.current.play();
      }
      if (token === generation.current) setCameraReady(true);
    } catch (err) {
      if (token !== generation.current) return;
      setConnecting(false);
      release();
      setCameraReady(false);
      const name = err instanceof DOMException ? err.name : "";
      setError(name === "NotAllowedError" ? "Camera permission denied. Allow camera access in browser settings, then retry." : name === "NotFoundError" ? "No camera found. Connect a camera and retry." : name === "NotReadableError" ? "Camera is busy or unavailable. Close other camera apps and retry." : err instanceof Error ? err.message : "Could not start the camera.");
    } finally {
      if (token === generation.current) { busy.current = false; setConnecting(false); }
    }
  }

  // This entry point can later be triggered by rep detection without changing analysis.
  function startRep() {
    if (busy.current || !stream.current || !cameraReady) return;
    busy.current = true;
    stopSpeaking();
    setError(null);
    const token = generation.current;
    const chunks: Blob[] = [];
    let failed = false;
    try {
      const mimeType = MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type));
      if (!mimeType) throw new Error("This browser cannot record a supported MP4 or WebM clip. Try another browser.");
      const recording = new MediaRecorder(stream.current, { mimeType });
      recorder.current = recording;
      recording.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
      recording.onerror = () => {
        if (token !== generation.current) return;
        failed = true;
        if (timer.current) clearTimeout(timer.current);
        if (recording.state !== "inactive") recording.stop();
        busy.current = false;
        setPhase("READY");
        setError("Recording failed. Please retry the repetition.");
      };
      recording.onstop = async () => {
        if (failed || token !== generation.current) return;
        if (timer.current) clearTimeout(timer.current);
        setPhase("ANALYZING");
        const controller = new AbortController();
        request.current = controller;
        const timeout = setTimeout(() => controller.abort(), 180_000);
        try {
          const blob = new Blob(chunks, { type: recording.mimeType });
          if (!blob.size) throw new Error("The camera returned an empty clip. Please retry.");
          const extension = recording.mimeType.includes("mp4") ? "mp4" : "webm";
          const analysis = await analyzeVideo(new File([blob], `rep-${Date.now()}.${extension}`, { type: blob.type }), controller.signal);
          if (token !== generation.current) return;
          setResult(analysis);
          setReps((count) => count + 1);
          setPhase("REP COMPLETE");
          const cue = spokenCue(analysis);
          if (voice.current && cue) speak(cue);
          timer.current = setTimeout(() => { if (token === generation.current) { setPhase("READY"); busy.current = false; } }, 1200);
        } catch (err) {
          if (token !== generation.current) return;
          setError(controller.signal.aborted ? "Analysis timed out. Check the backend connection and retry." : err instanceof TypeError ? "Cannot reach the SCAN backend. Check the API URL, HTTPS certificate, network and CORS settings." : err instanceof Error ? err.message : "Analysis failed. Please retry.");
          setPhase("READY");
          busy.current = false;
        } finally { clearTimeout(timeout); }
      };
      recording.start();
      setResult(null);
      setPhase("RECORDING");
      timer.current = setTimeout(() => {
        if (recording.state === "recording") recording.stop();
      }, REP_MS);
    } catch (err) {
      busy.current = false;
      setPhase("READY");
      setError(err instanceof Error ? err.message : "Could not start recording.");
    }
  }

  return (
    <section className="live-session result-panel">
      <button className="secondary-button" aria-expanded={open} onClick={() => {
        if (open) { release(); setCameraReady(false); setConnecting(false); setPhase("READY"); }
        setOpen(!open);
      }}>{open ? "CLOSE LIVE SESSION" : "OPEN LIVE SESSION"}</button>
      {open && <>
        <h2 className="panel-label">LIVE SESSION v0.1</h2>
        <p>Place your phone where your full movement is visible. START REP records 4.5 seconds, then analyses the clip on your laptop. Keep this page visible.</p>
        <video ref={preview} className="live-preview" autoPlay muted playsInline aria-label="Live camera preview" />
        <p role="status" aria-live="polite">{connecting ? "REQUESTING CAMERA" : !cameraReady ? "CAMERA OFF" : phase} · {reps} reps complete</p>
        <div className="live-controls">
          {!cameraReady && <button className="secondary-button" disabled={connecting} onClick={enableCamera}>ENABLE CAMERA</button>}
          <button className="secondary-button" disabled={!cameraReady || phase !== "READY"} onClick={startRep}>START REP</button>
        </div>
        {error && <div className="analysis-error" role="alert">{error}</div>}
        {result && <div className="feedback-panel">
          <div className="panel-label">LAST REP · {result.analysis_id}</div>
          {result.coaching_report ? <>
            <p>{result.coaching_report.overall_assessment}</p>
            <p><strong>What went well: </strong>{result.coaching_report.what_went_well}</p>
            <p><strong>Focus: </strong>{result.coaching_report.primary_focus ?? "No specific correction warranted"}</p>
            <p><strong>Outcome: </strong>{result.coaching_report.outcome_analysis}</p>
            <p><strong>Next rep: </strong>{result.coaching_report.next_rep_recommendation}</p>
            {result.coaching_report.confidence_note && <p>{result.coaching_report.confidence_note}</p>}
          </> : <p>{result.summary.prototype_feedback}</p>}
          {spokenCue(result) && <>
            <p><strong>Spoken cue: </strong>{spokenCue(result)}</p>
            <button className="secondary-button" disabled={!voiceEnabled || phase === "RECORDING"} onClick={() => speak(spokenCue(result))}>REPLAY CUE</button>
          </>}
        </div>}
      </>}
    </section>
  );
}
