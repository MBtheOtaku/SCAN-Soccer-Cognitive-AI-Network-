"""Offline contact annotation and interval scoring; never supplies inference inputs.

Build: python -m scripts.review_contact_timing
Score: python -m scripts.review_contact_timing --labels path/to/contact-labels.json
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "data/reference-review/pose-coverage-20261120/foreground-check/summary.json"
DEFAULT_OUT = ROOT / "data/reference-review/contact-timing-20261120"


def score_interval(estimate, label, total_frames, fps):
    """Signed distance to a reviewer interval, not to its invented midpoint."""
    if not math.isfinite(fps) or fps <= 0 or total_frames <= 0:
        raise ValueError("Invalid video timing")
    if label.get("status") == "unresolved":
        return {"status": "unresolved", "signed_error_frames": None}
    if label.get("status") != "reviewed":
        raise ValueError("Reference must be reviewed or unresolved")
    lo, hi = label.get("contact_start_frame"), label.get("contact_end_frame")
    movement = label.get("first_ball_motion_frame")
    for value in (lo, hi, movement):
        if type(value) is not int or not 0 <= value < total_frames:
            raise ValueError("Reference frames must be integer decoded frame indices")
    if not lo <= hi <= movement:
        raise ValueError("Contact interval must precede or include first ball movement")
    if estimate is None:
        return {"status": "no_estimate", "signed_error_frames": None}
    if type(estimate) is not int or not 0 <= estimate < total_frames:
        raise ValueError("Invalid estimated frame")
    error = estimate - lo if estimate < lo else estimate - hi if estimate > hi else 0
    return {"status": "inside_reference_interval" if error == 0 else "outside_reference_interval",
            "signed_error_frames": error, "signed_error_s": error / fps,
            "absolute_error_s": abs(error) / fps,
            "reference_interval_frames": [lo, hi], "first_ball_motion_frame": movement}


def build(audit, out):
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    cards, manifest = [], []
    for row in audit:
        source = Path(row["source"])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != row["source_sha256"]:
            raise ValueError(f"Source changed: {source.name}")
        clip = row["clip"]
        folder = out / clip
        folder.mkdir(exist_ok=True)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f"Cannot decode {source}")
        count = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # Full image, no detector or annotation overlays; frame access is sequential.
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (round(w * min(1, 960 / h)), min(h, 960)))
                if not cv2.imwrite(str(folder / f"{count:04d}.jpg"), small):
                    raise OSError("Could not write review frame")
                count += 1
        finally:
            cap.release()
        if count != row["decoded_frames"]:
            raise ValueError(f"Decoded frame count changed for {clip}")
        item = {"clip": clip, "source_sha256": digest, "fps": row["fps"],
                "frames": count, "colour": row["colour"],
                "estimate_frame": row["current"]["summary"]["probable_strike_frame"]}
        manifest.append(item)
        cards.append(f'<option value="{len(manifest)-1}">{html.escape(clip)} ({html.escape(row["colour"])})</option>')
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    page = PAGE.replace("__CASES__", json.dumps(manifest).replace("<", "\\u003c")).replace("__OPTIONS__", "".join(cards))
    (out / "index.html").write_text(page, encoding="utf-8")
    print(out / "index.html")


PAGE = '''<!doctype html><meta charset="utf-8"><title>SCAN contact timing audit</title>
<style>body{background:#0b1510;color:#e6f0e8;font:16px system-ui;max-width:1100px;margin:auto;padding:24px}button,select,input{font:inherit;margin:6px;padding:8px}img{height:68vh;max-width:100%;object-fit:contain}#range{width:95%}label{display:inline-block}pre{white-space:pre-wrap}a{color:#8de6b0}</style>
<h1>Contact timing review</h1>
<p>Development audit, not an automatic contact detector. Use the last frame before visible contact and first frame showing separation to bracket uncertain contact. Mark first visible shot-ball movement separately. If obscured, choose unresolved.</p>
<p>Zero-based decoded frames. Times are frame/FPS approximations, not verified presentation timestamps. Labels are evaluation only and never feed app analysis. No ball labels or seeds were used to create these images.</p>
<select id="clip">__OPTIONS__</select><br><img id="image" alt="Unannotated decoded source frame"><br>
<button id="prev">Previous frame</button><button id="next">Next frame</button><button id="peak">Show ankle-speed estimate</button>
<input id="range" type="range" min="0"><p id="time"></p>
<p><label>Contact interval start <input id="start" type="number" min="0"></label><button id="markStart">Use current</button>
<label>Contact interval end <input id="end" type="number" min="0"></label><button id="markEnd">Use current</button>
<label>First ball movement <input id="motion" type="number" min="0"></label><button id="markMotion">Use current</button></p>
<button id="save">Save reviewed interval</button><button id="unresolved">Mark unresolved</button><button id="download">Download reference JSON</button><pre id="status"></pre>
<p>References are kept in memory until downloaded. Download before closing this page. Do not infer confirmed contact from the ankle-speed marker.</p>
<script>
const cases=__CASES__, refs={};let i=0,f=0;
const el=id=>document.getElementById(id);
function draw(){const c=cases[i];f=Math.max(0,Math.min(c.frames-1,f));el('image').src=c.clip+'/'+String(f).padStart(4,'0')+'.jpg';el('range').max=c.frames-1;el('range').value=f;el('time').textContent=`Frame ${f}, approximately ${(f/c.fps).toFixed(3)}s. Ankle-speed estimate: frame ${c.estimate_frame}.`;}
function change(){i=Number(el('clip').value);f=0;const r=refs[cases[i].clip]||{};for(const [id,key] of [['start','contact_start_frame'],['end','contact_end_frame'],['motion','first_ball_motion_frame']])el(id).value=r[key]??'';draw();}
el('clip').onchange=change;el('prev').onclick=()=>{f--;draw()};el('next').onclick=()=>{f++;draw()};el('peak').onclick=()=>{f=cases[i].estimate_frame??0;draw()};el('range').oninput=e=>{f=Number(e.target.value);draw()};
for(const [button,input] of [['markStart','start'],['markEnd','end'],['markMotion','motion']])el(button).onclick=()=>el(input).value=f;
function record(unresolved){const c=cases[i],a=Number(el('start').value),b=Number(el('end').value),m=Number(el('motion').value);if(!unresolved&&(['start','end','motion'].some(id=>el(id).value==='')||![a,b,m].every(Number.isInteger)||a<0||a>b||b>m||m>=c.frames)){el('status').textContent='Choose valid contact start <= end <= first movement within the clip.';return;}refs[c.clip]={clip:c.clip,source_sha256:c.source_sha256,status:unresolved?'unresolved':'reviewed',...(unresolved?{}:{contact_start_frame:a,contact_end_frame:b,first_ball_motion_frame:m})};el('status').textContent=JSON.stringify(refs,null,2);}
el('save').onclick=()=>record(false);el('unresolved').onclick=()=>record(true);
el('download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify({schema_version:1,purpose:'evaluation_only_not_inference',annotations:Object.values(refs)},null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='contact-labels.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};draw();
</script>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if args.labels:
        rows = {r["clip"]: r for r in audit}
        labels = json.loads(args.labels.read_text(encoding="utf-8"))["annotations"]
        results, seen = [], set()
        for label in labels:
            clip = label["clip"]
            if clip in seen:
                raise ValueError(f"Duplicate reference: {clip}")
            seen.add(clip)
            row = rows[clip]
            if row["source_sha256"] != label["source_sha256"]:
                raise ValueError(f"Reference source mismatch: {clip}")
            results.append({"clip": clip, **score_interval(row["current"]["summary"]["probable_strike_frame"], label, row["decoded_frames"], row["fps"])})
        args.out.mkdir(parents=True, exist_ok=True)
        report = {"purpose": "development_contact_timing_evaluation_not_accuracy_validation",
                  "reference_provenance": json.loads(args.labels.read_text(encoding="utf-8")).get("provenance", "reviewer_not_specified"),
                  "unreviewed_clips": sorted(set(rows)-seen), "results": results}
        (args.out / "scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        table = "".join(f'<tr><td>{html.escape(r["clip"])}</td><td>{html.escape(r["status"])}</td>'
                        f'<td>{r.get("signed_error_frames")}</td><td>{r.get("signed_error_s")}</td></tr>' for r in results)
        (args.out / "comparison.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Contact timing comparison</title>'
            '<style>body{font:16px system-ui;background:#0b1510;color:#e6f0e8;padding:24px}td,th{padding:10px;text-align:left}a{color:#8de6b0}</style>'
            '<h1>Provisional contact timing comparison</h1><p>Development clips; interval references are visual estimates, not confirmed contact timing. '
            'Negative error means the ankle peak is early. Zero means inside the reference interval, not exact contact. '
            'No reference labels feed app inference.</p><p>' + html.escape(str(report["reference_provenance"])) + '</p>'
            '<a href="index.html">Inspect every original frame / create your own references</a>'
            '<table><tr><th>Clip</th><th>Result</th><th>Error frames</th><th>Error seconds</th></tr>' + table + '</table>', encoding="utf-8")
        print(json.dumps(report, indent=2))
    else:
        build(audit, args.out)


if __name__ == "__main__":
    main()
