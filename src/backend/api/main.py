from dataclasses import asdict
from pathlib import Path
import shutil
import uuid
import json
import os
import base64

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from src.backend.pose_mvp.scan_pose_mvp_v2 import analyze


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "backend"
    / "pose_mvp"
    / "pose_landmarker_full.task"
)

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------

app = FastAPI(
    title="SCAN API",
    description="Backend API for the Soccer Cognitive AI Network",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://localhost:3000",
        "https://127.0.0.1:3000",
    ] + [origin.strip() for origin in os.getenv("SCAN_CORS_ORIGINS", "").split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Makes generated files available at:
# http://127.0.0.1:8000/outputs/...
app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs",
)


# ---------------------------------------------------------
# Basic routes
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "SCAN",
        "full_name": "Soccer Cognitive AI Network",
        "status": "online",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "SCAN API",
    }


# ---------------------------------------------------------
# Video analysis
# ---------------------------------------------------------

@app.post("/goals/preview")
async def goal_video_preview(video: UploadFile = File(...)):
    import cv2
    from src.backend.goal_preview import extract_goal_preview
    from src.backend.goal_proposal import propose_goals
    suffix = Path(video.filename or "").suffix.lower()
    preview_path = INPUT_DIR / f"goal-preview-{uuid.uuid4().hex}{suffix}"
    try:
        if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}:
            raise ValueError("Unsupported video format for goal setup.")
        with preview_path.open("wb") as output:
            await run_in_threadpool(shutil.copyfileobj, video.file, output)
        image, frame_index = await run_in_threadpool(extract_goal_preview, preview_path)
        candidates = await run_in_threadpool(propose_goals, image)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("Could not create the goal reference image.")
        return {"image_base64": base64.b64encode(encoded).decode("ascii"), "frame_index": frame_index,
                "candidates": candidates, "requires_confirmation": True}
    except (ValueError, cv2.error) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        await video.close()
        preview_path.unlink(missing_ok=True)


@app.post("/goals/propose")
async def propose_goal(frame: UploadFile = File(...)):
    from src.backend.goal_setup import decode_reference
    from src.backend.goal_proposal import propose_goals
    try:
        contents = await frame.read(5_000_001)
        image = decode_reference(contents)
        candidates = await run_in_threadpool(propose_goals, image)
        return {"candidates": candidates, "requires_confirmation": True,
                "note": "Prototype goal-frame geometry proposals. Confirm or adjust all corners; scores are not probabilities."}
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        await frame.close()


@app.post("/analyze")
async def analyze_video(
    video: UploadFile = File(...),
    roi: str | None = Form(default=None),
    athlete_start_side: str | None = Form(default=None),
    goal_setup: str | None = Form(default=None),
    goal_reference: UploadFile | None = File(default=None),
):
    """
    Upload a soccer drill video and run the current
    SCAN pose-analysis MVP.

    ROI format:
        x1,y1,x2,y2

    Example for IMG_5702.mov:
        540,280,1180,650
    """

    # Validate optional setup before saving or running the existing pipeline.
    from src.backend.goal_setup import parse_goal_setup, decode_reference
    reference_contents = None
    try:
        calibration = parse_goal_setup(goal_setup if isinstance(goal_setup, str) else None)
        if calibration:
            if goal_reference is None or not hasattr(goal_reference, "read"):
                raise ValueError("A confirmed goal requires its reference photo.")
            reference_contents = await goal_reference.read(5_000_001)
            decode_reference(reference_contents)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid goal setup: {exc}")
    finally:
        if goal_reference is not None and hasattr(goal_reference, "close"):
            await goal_reference.close()
    if not isinstance(athlete_start_side, str):
        athlete_start_side = None
    if athlete_start_side not in {None, "left", "right"}:
        raise HTTPException(status_code=400, detail="athlete_start_side must be left or right.")

    allowed_extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".m4v",
        ".webm",
    }

    suffix = Path(video.filename or "").suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {suffix}",
        )

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Pose model not found: {MODEL_PATH}",
        )

    # Give every analysis its own unique ID.
    analysis_id = uuid.uuid4().hex[:12]

    input_path = INPUT_DIR / f"{analysis_id}{suffix}"

    analysis_output_dir = OUTPUT_DIR / analysis_id
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = (
        analysis_output_dir / "scan_annotated_v2.mp4"
    )

    csv_path = (
        analysis_output_dir / "scan_pose_metrics_v2.csv"
    )

    json_path = (
        analysis_output_dir / "scan_shot_summary.json"
    )

    # -----------------------------------------------------
    # Save uploaded video
    # -----------------------------------------------------

    try:
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save uploaded video: {exc}",
        )

    finally:
        await video.close()

    # -----------------------------------------------------
    # Run SCAN pose analysis
    # -----------------------------------------------------

    try:
        reference_path = None
        if reference_contents:
            reference_path = analysis_output_dir / "goal_reference.jpg"
            reference_path.write_bytes(reference_contents)
        summary = await run_in_threadpool(
            analyze,
            input_path,
            MODEL_PATH,
            annotated_path,
            csv_path,
            json_path,
            roi,
            goal_calibration_override=calibration,
            use_legacy_goal_calibration=False,
            goal_reference_path=reference_path,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SCAN analysis failed: {exc}",
        )

    # -----------------------------------------------------
    # Load full SCAN analysis JSON
    # -----------------------------------------------------

    try:
        analysis_data = json.loads(
            json_path.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load SCAN analysis JSON: {exc}",
        )

    # -----------------------------------------------------
    # API response
    # -----------------------------------------------------

    pressure_report = None
    if athlete_start_side:
        try:
            from src.backend.opponent_pressure import analyze_pressure
            from src.backend.pressure_report import build_pressure_report
            evidence = await run_in_threadpool(
                analyze_pressure, input_path, athlete_start_side, PROJECT_ROOT / "yolo26n.pt"
            )
            pressure_report = build_pressure_report(evidence)
        except Exception:
            # Optional contextual analysis must never break a completed SCAN report.
            pressure_report = {
                "status": "unavailable", "events": [],
                "assessment": "Opponent-pressure assessment is unavailable; technique analysis completed normally.",
                "confidence_note": "No pressure or safe escape direction was established.",
                "next_rep_recommendation": "Retry with both players visible and the local person detector available.",
                "spoken_cue": None,
                "policy": {"decision_type": "silence_default", "should_speak": False},
            }

    return {
        "analysis_id": analysis_id,
        "filename": video.filename,
        "pressure_report": pressure_report,
        "goal_motion": analysis_data.get("goal_motion"),
        "goal_setup": analysis_data.get("goal_setup"),

        "summary": asdict(summary),

        "coaching_report": (
            analysis_data.get(
                "coaching_report"
            )
        ),

        "jev_judgements": (
            analysis_data.get(
                "jev_judgements"
            )
        ),

        "artifacts": {
            "annotated_video": (
                f"/outputs/{analysis_id}/scan_annotated_v2.mp4"
            ),
            "metrics_csv": (
                f"/outputs/{analysis_id}/scan_pose_metrics_v2.csv"
            ),
            "shot_summary": (
                f"/outputs/{analysis_id}/scan_shot_summary.json"
            ),
        },
    }
