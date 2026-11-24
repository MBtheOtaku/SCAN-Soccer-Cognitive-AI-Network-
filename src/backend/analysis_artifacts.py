"""Render-only helpers; perception always keeps original video coordinates."""
from dataclasses import replace
from pathlib import Path
import subprocess


def preview_dimensions(width, height, maximum_dimension=None):
    if any(type(v) is not int or v < 2 for v in (width, height)):
        raise ValueError("Expected positive image dimensions")
    if maximum_dimension is None:
        return width, height
    if type(maximum_dimension) is not int or maximum_dimension < 2:
        raise ValueError("Expected integer preview bound of at least two pixels")
    ratio = min(1., maximum_dimension / max(width, height))
    return tuple(max(2, int(v * ratio) // 2 * 2) for v in (width, height))


def detections_for_preview(detections, source_dimensions, render_dimensions):
    sx = render_dimensions[0] / source_dimensions[0]
    sy = render_dimensions[1] / source_dimensions[1]
    return [replace(d, center_x=d.center_x*sx, center_y=d.center_y*sy,
                    bbox_width=d.bbox_width*sx, bbox_height=d.bbox_height*sy)
            for d in detections]


def encode_browser_video(source: Path, destination: Path):
    """One bounded, fast H.264 artifact encode. Originals are never passed out."""
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-c:v", "libx264", "-preset", "veryfast",
        "-threads", "4", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an", str(destination),
    ], check=True)
