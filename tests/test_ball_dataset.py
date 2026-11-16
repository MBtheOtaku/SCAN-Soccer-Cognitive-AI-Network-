import argparse
import json

import pytest

from scripts import ball_dataset as dataset


def sample():
    manifest = {"clip_id": "13a1022c5009", "source_sha256": "example",
                "frames": [{"frame": 79, "width": 720, "height": 1280,
                            "image": "frames/000079.jpg"}]}
    labels = {"schema": 1, "clip_id": manifest["clip_id"], "source_sha256": "example",
              "annotations": [{"frame": 79, "state": "visible", "box": [260, 750, 280, 780]}]}
    return manifest, labels


def test_sampling_includes_dense_contact_and_uniform_context():
    frames = dataset.sample_indices(185, 10, 75, 85)
    assert set(range(75, 86)) <= set(frames)
    assert 0 in frames and 180 in frames
    assert len(frames) == len(set(frames))


def test_named_camera_recording_gets_stable_content_id(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    source = tmp_path / "data/input/IMG_5703.mov"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"example recording")
    clip_id, path, sha = dataset.resolve_prepare_source(None, source)
    assert clip_id == sha[:12] and path == source and sha == dataset.file_sha256(source)
    with pytest.raises(ValueError, match="either an analysis ID"):
        dataset.resolve_prepare_source("13a1022c5009", source)
    outside = tmp_path / "elsewhere.mov"
    outside.write_bytes(b"example recording")
    with pytest.raises(ValueError, match="data/input"):
        dataset.resolve_prepare_source(None, outside)


@pytest.mark.parametrize("start,end", [(None, 4), (-1, 4), (3, 30), (7, 3)])
def test_invalid_sampling(start, end):
    with pytest.raises(ValueError):
        dataset.sample_indices(20, 10, start, end)


@pytest.mark.parametrize("state,expected", [("absent", 1), ("uncertain", 0), ("unreviewed", 0)])
def test_only_explicit_absence_becomes_negative(state, expected):
    manifest, labels = sample()
    labels["annotations"][0].update(state=state, box=None)
    assert len(dataset.validate_labels(manifest, labels)) == expected


@pytest.mark.parametrize("box", [[0, 1, 0, 3], [0, 0, 721, 10], [0, 0, float('nan'), 5], None])
def test_reject_invalid_box(box):
    manifest, labels = sample()
    labels["annotations"][0]["box"] = box
    with pytest.raises(ValueError):
        dataset.validate_labels(manifest, labels)


def test_reject_duplicate_frame_and_wrong_source():
    manifest, labels = sample()
    labels["source_sha256"] = "other"
    with pytest.raises(ValueError):
        dataset.validate_labels(manifest, labels)
    labels["source_sha256"] = "example"
    labels["annotations"] *= 2
    with pytest.raises(ValueError):
        dataset.validate_labels(manifest, labels)


def test_same_session_cannot_cross_splits():
    with pytest.raises(ValueError, match="Recording group"):
        dataset.validate_splits({"clips": [
            {"clip_id": "13a1022c5009", "group": "night-session", "split": "train"},
            {"clip_id": "e9935750581e", "group": "night-session", "split": "val"}]})


def test_export_normalises_original_coordinates_and_excludes_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "DATA", tmp_path / "annotations")
    manifest, labels = sample()
    for index, state in [(80, "absent"), (81, "uncertain"), (82, "unreviewed")]:
        manifest["frames"].append({"frame": index, "width": 720, "height": 1280,
                                   "image": f"frames/{index:06d}.jpg"})
        labels["annotations"].append({"frame": index, "state": state, "box": None})
    folder = dataset.DATA / manifest["clip_id"]
    (folder / "frames").mkdir(parents=True)
    # Export copies existing encoded bytes; image decoding is preparation's responsibility.
    (folder / "frames/000079.jpg").write_bytes(b"test-image")
    (folder / "frames/000080.jpg").write_bytes(b"negative-image")
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "labels.json").write_text(json.dumps(labels))
    config = tmp_path / "splits.json"
    config.write_text(json.dumps({"clips": [{"clip_id": manifest["clip_id"],
                                            "group": "session-1", "split": "train"}]}))
    output = tmp_path / "export"
    dataset.export_dataset(argparse.Namespace(config=config, output=output))
    values = (output / "labels/train/13a1022c5009_000079.txt").read_text().split()
    assert values[0] == "0"
    assert list(map(float, values[1:])) == pytest.approx([270/720, 765/1280, 20/720, 30/1280])
    assert (output / "images/train/13a1022c5009_000079.jpg").read_bytes() == b"test-image"
    assert (output / "labels/train/13a1022c5009_000080.txt").read_text() == ""
    assert not (output / "images/train/13a1022c5009_000081.jpg").exists()
    assert not (output / "images/train/13a1022c5009_000082.jpg").exists()
    with pytest.raises(ValueError, match="already exists"):
        dataset.export_dataset(argparse.Namespace(config=config, output=output))
