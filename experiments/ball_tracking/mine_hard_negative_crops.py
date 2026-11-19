"""Snapshot a crop dataset with extra training-only, detector-selected distractors.

This is an offline sampling experiment. Review the montage for unlabelled balls
before training; duplicated crops deliberately increase hard-negative weight.
"""

import argparse
import json
from pathlib import Path
import shutil

from scripts.finetune_ball import digest, inventory


def mine(source: Path, weights: Path, output: Path, confidence: float, limit: int):
    import cv2
    from ultralytics import YOLO

    source, weights, output = source.resolve(), weights.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Choose a new output folder; dataset snapshots are immutable.")
    if not 0 < confidence < 1 or limit < 1:
        raise ValueError("Invalid confidence or limit.")
    inventory(source)
    model = YOLO(str(weights))
    if model.task != "detect" or model.names != {0: "ball"}:
        raise ValueError(f"Expected one-class ball weights: {model.names}")
    candidates = []
    for image_path in sorted((source / "images/train").glob("*.jpg")):
        label = source / "labels/train" / (image_path.stem + ".txt")
        if label.read_text(encoding="utf-8").strip():
            continue
        prediction = model.predict(str(image_path), imgsz=640, conf=confidence,
                                   device="cpu", verbose=False)[0]
        if len(prediction.boxes):
            scores = prediction.boxes.conf.cpu().numpy().tolist()
            candidates.append((max(scores), image_path, prediction.boxes.xyxy.cpu().numpy().tolist()))
    candidates.sort(key=lambda item: (-item[0], item[1].name))
    selected = candidates[:limit]
    if not selected:
        print("No baseline false positives on training negative crops; nothing to oversample.")
        return []
    shutil.copytree(source, output, ignore=shutil.ignore_patterns("*.cache"))
    records = []
    thumbs = []
    for score, original, boxes in selected:
        new_image = output / "images/train" / (original.stem + "_hard0.jpg")
        new_label = output / "labels/train" / (new_image.stem + ".txt")
        shutil.copy2(original, new_image)
        new_label.write_text("", encoding="utf-8")
        records.append({"source": original.name, "image": new_image.relative_to(output).as_posix(),
                        "max_confidence": score, "boxes": boxes,
                        "image_sha256": digest(new_image), "label_sha256": digest(new_label)})
        image = cv2.imread(str(original))
        thumb = cv2.resize(image, (192, 192))
        sx, sy = 192 / image.shape[1], 192 / image.shape[0]
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(thumb, (round(x1 * sx), round(y1 * sy)),
                          (round(x2 * sx), round(y2 * sy)), (0, 255, 255), 1)
        cv2.putText(thumb, f"{original.stem[-18:]} {score:.2f}", (3, 184),
                    cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 255, 255), 1)
        thumbs.append(thumb)
    import numpy as np
    columns = min(8, len(thumbs))
    rows = (len(thumbs) + columns - 1) // columns
    montage = np.zeros((rows * 192, columns * 192, 3), dtype=np.uint8)
    for index, thumb in enumerate(thumbs):
        y, x = divmod(index, columns)
        montage[y*192:(y+1)*192, x*192:(x+1)*192] = thumb
    cv2.imwrite(str(output / "hard-negative-review.jpg"), montage)
    (output / "hard-negative-mining.json").write_text(json.dumps({
        "schema": 1, "source": str(source), "weights": str(weights),
        "weights_sha256": digest(weights), "confidence": confidence,
        "limit": limit, "eligible_empty_train_crops": sum(1 for p in (source / "images/train").glob("*.jpg")
                                                  if not (source / "labels/train" / (p.stem + ".txt")).read_text().strip()),
        "false_positive_crops": len(candidates), "selected": records,
        "limitation": "Review montage for other unlabelled balls. The validation split was not mined or altered."
    }, indent=2) + "\n", encoding="utf-8")
    (output / "dataset.yaml").write_text('path: ' + json.dumps(output.as_posix()) +
                                           '\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n',
                                           encoding="utf-8")
    inventory(output)
    print(f"Mined {len(selected)} distractors from {len(candidates)} false-positive crops")
    print(output / "hard-negative-review.jpg")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=48)
    args = parser.parse_args()
    mine(args.source, args.weights, args.output, args.confidence, args.limit)


if __name__ == "__main__":
    main()
