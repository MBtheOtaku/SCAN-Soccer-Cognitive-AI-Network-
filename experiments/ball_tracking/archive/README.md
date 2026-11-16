# Archived ball-tracking experiments

Archived 10 November 2026 to leave one [active workflow](../../../docs/ball-tracking.md).
This archive preserves implementations and failure evidence; it is not production
code or a list of equally active proposals. No inference/tracking behaviour was
intentionally changed during the move; only imports and repository-root paths were
updated. Run modules from the SCAN repository root.

| Group | Preserved implementation | Why it is archived |
|---|---|---|
| `detector_v2/` | Dedicated YOLO adapter, tiled inference, conservative candidate association and checks | Dedicated checkpoint missed true balls or selected turf on inspected phone footage. Association cannot supply missing detections. |
| `wasb/` | Three-frame CPU adapter, preprocessing/heatmap checks | Night probe selected a floodlight; both daylight contact windows had no candidates at threshold 0.5. Full numerical equivalence to the author's CUDA pipeline was not established. |
| `contact_candidates/` | Independent generic YOLO/ankle proximity proposals | Some near-foot proposals selected boots instead of the ball. Proximity did not establish contact. |
| `research.md` | Initial literature survey and primary-source links | Historical recommendations predate the recorded failures above. |

The historical guides in these groups retain original setup narratives and command
snippets for context. **Their old copy/template instructions are superseded.**
Seven `.py.template` files were removed only after comparing normalised text with
the actual implementations and verifying exact equality. Their code survives here.

## Reproduce a probe when needed

Existing models/recordings are still required; they were not moved or deleted.

```powershell
python -m experiments.ball_tracking.archive.detector_v2.review_ball_v2 --help
python -m experiments.ball_tracking.archive.wasb.probe_wasb --help
python -m experiments.ball_tracking.archive.contact_candidates.propose_contacts --help
```

For example, rerun the old WASB contact probe with its original model/source folders:

```powershell
python -m experiments.ball_tracking.archive.wasb.probe_wasb 13a1022c5009 --start-frame 78
```

Archived checks were renamed `checks_*.py` so normal `test_*.py` discovery does not
execute historical model dependencies. They remain directly runnable:

```powershell
python -m pytest experiments/ball_tracking/archive/detector_v2/checks_ball_detector_v2.py experiments/ball_tracking/archive/detector_v2/checks_ball_temporal_v2.py experiments/ball_tracking/archive/wasb/checks_wasb_probe.py -q
```

Generated results remain at their existing ignored `data/reference-review/ball-v2/`,
`wasb/` and `contact-candidates/` paths. No recordings, labels, model weights or
experiment outputs were removed. Production backend and mobile source were not edited.

## Cleanup recovery

A verified pre-cleanup ZIP and source/destination/hash manifest are preserved in:

`data/reference-review/repo-cleanup/20261110T210906919587Z/`

The ZIP includes the current uncommitted versions of moved files, removed duplicate
templates, and the previous root README. It is local/ignored. Inspect or extract
individual entries into a separate recovery folder; do not blindly overwrite newer
working files. Git history is an additional recovery source for previously committed
files, but is not the only backup for uncommitted work.
