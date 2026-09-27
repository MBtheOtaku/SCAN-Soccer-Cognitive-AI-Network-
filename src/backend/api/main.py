from dataclasses import asdict
from pathlib import Path
import shutil
import uuid

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
    ],
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

@app.post("/analyze")
async def analyze_video(
    video: UploadFile = File(...),
    roi: str | None = Form(default=None),
):
    """
    Upload a soccer drill video and run the current
    SCAN pose-analysis MVP.

    ROI format:
        x1,y1,x2,y2

    Example for IMG_5702.mov:
        540,280,1180,650
    """

    allowed_extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".m4v",
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
        summary = await run_in_threadpool(
            analyze,
            input_path,
            MODEL_PATH,
            annotated_path,
            csv_path,
            json_path,
            roi,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SCAN analysis failed: {exc}",
        )

    # -----------------------------------------------------
    # API response
    # -----------------------------------------------------

    return {
        "analysis_id": analysis_id,
        "filename": video.filename,
        "summary": asdict(summary),
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