# Ball tracking research for SCAN

Research checked 6 November 2026. Sources below are original papers, author repositories and official implementation documentation. Recommendations are engineering proposals for SCAN, not measured results on our videos. No external models were installed or benchmarked during this review.

## What our experiments establish

In clip `13a1022c5009`, independent raw detections include boots and turf markings as balls, and miss the actual ball around the user-reviewed contact interval (frames 79–80). Proximity candidates at frames 76 and 81 circle the boot rather than the ball. The production tracker also starts by choosing the highest-confidence detection before a trajectory exists. A confidently maintained false track can therefore inflate detection coverage.

Three distinct problems need separate evaluation:

1. **Detection:** is the true ball among the candidates?
2. **Association:** do we keep the identity of that ball through time?
3. **Event understanding:** which contact is the final shot?

A tracker cannot turn a boot observation into evidence of a real ball. A contact detector cannot be evaluated fairly against an incorrect ball trajectory. Plausible ankle overlays in this clip do not establish pose accuracy across all clips.

## Existing implementations worth using

### A. Roboflow sports soccer example: practical starting skeleton

The official example uses a dedicated `football-ball-detection.pt` model, overlapping image slices and duplicate suppression before its ball tracker. The repository includes model download references. This is substantially closer to SCAN's task than generic COCO sports-ball inference. [Example pipeline](https://github.com/roboflow/sports/blob/main/examples/soccer/main.py), [model setup](https://github.com/roboflow/sports/blob/main/examples/soccer/setup.sh).

Its `BallTracker` stores recent detection centres and selects the current detection nearest their combined centroid. This is a simple demonstration association rule. **My inference:** false detections can contaminate that history, and a recent-location average can lag after a kick; it is not a complete solution for SCAN. Reuse the detection/slicing skeleton first and benchmark association rather than copying the tracker unquestioningly. [Actual tracker source](https://github.com/roboflow/sports/blob/main/sports/common/ball.py).

### B. SAHI: higher-resolution small-object detection

SAHI processes overlapping crops, maps detections back into full-image coordinates, and merges duplicates. Official Ultralytics documentation describes its integration with YOLO26. [SAHI paper](https://arxiv.org/abs/2202.06934), [official YOLO integration](https://docs.ultralytics.com/guides/sahi-tiled-inference/).

**Proposed experiment:** compare the current detector with and without slicing before changing association. Slicing may recover a small or distant ball, but can also produce additional false positives and more computation. It does not supply the semantic distinction between a boot and a ball. Published improvements on aerial datasets are not evidence of improvement on SCAN.

### C. WASB-SBDT: strongest research baseline to benchmark first

WASB combines high-resolution feature extraction, location-sensitive training and temporal inference. Consecutive frames produce position heatmaps; temporal selection rejects inconsistent locations. The paper evaluates soccer alongside four other sports. It predicts at most one ball location per frame. [BMVC 2023 paper](https://arxiv.org/abs/2311.05237).

The author repository provides evaluation code and a model zoo with **soccer-specific checkpoints**, including WASB, DeepBall, BallSeg and TrackNetV2. The documented setup is older Linux/CUDA software, so use an isolated environment rather than altering SCAN's working backend dependencies. Download links were found; the weight files were not downloaded or tested here. [Code](https://github.com/nttcom/WASB-SBDT), [model zoo](https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md), [setup](https://github.com/nttcom/WASB-SBDT/blob/main/GET_STARTED.md).

**My recommendation:** WASB's soccer model is a more relevant first temporal-model benchmark than assuming a badminton checkpoint will work on football. Soccer research footage still differs from handheld phone recordings, nighttime turf and close training views; performance must be measured locally.

### D. TrackNet family: learn location from a sequence

The original TrackNet learns heatmaps from consecutive images, addressing small, fast and blurred sports objects. [Original paper](https://arxiv.org/abs/1907.03698).

TrackNetV3 adds background information and a trajectory-repair network, with code and downloadable checkpoints. Its primary demonstration is shuttlecock tracking. Predicted repairs must remain predictions in SCAN, especially near hidden foot–ball contact. The repo documents older Python/PyTorch dependencies. [Author code](https://github.com/qaz812345/TrackNetV3).

TrackNetV4 introduces motion-aware fusion. Its official TensorFlow implementation provides training, evaluation and prediction workflows and pretrained-model references for tennis/badminton-oriented experiments. It is a research alternative requiring a domain-transfer check, not a verified soccer replacement. [Project and paper](https://tracknetv4.github.io/), [code](https://github.com/TrackNetV4/TrackNetV4).

**Why relevant:** the network can use temporal image evidence instead of asking only whether one patch looks round. **Limitation:** camera motion and sport/domain differences still matter; temporal models can also make wrong predictions.

### E. FootAndBall: an additional soccer-specific detector

FootAndBall produces ball confidence maps alongside player detections, with code and a pretrained model. It targets high-resolution long-shot soccer footage, uses feature-pyramid information and documents an older PyTorch environment. It is an additional detector baseline if the first two comparisons are inconclusive. [Author repository](https://github.com/jac99/FootAndBall), [paper](https://arxiv.org/abs/1912.05445).

### F. Norfair: reusable association and camera-motion scaffolding

Norfair supports point-based detections, custom matching distances, Kalman-style position estimation and moving-camera examples. Its repository also has a small-object/SAHI demo. Use it if a clearer association framework is preferable to expanding our custom tracker. It still depends on detector quality. [Official code and demos](https://github.com/tryolabs/norfair).

**My inference:** a general multi-object tracker, including ByteTrack, should not be assumed to solve our failure. Wrong initialization and boot false positives require target-specific evidence and evaluation regardless of tracker library.

## Recommended SCAN design

This is a proposed architecture, not an existing implementation or claimed research result:

```text
Original video with frame timing
    -> independent ball candidates / temporal position heatmaps
    -> association with appearance, size and motion checks
    -> observed / tentative / predicted / lost states
    -> possible contact intervals using pose AND ball evidence
    -> shot selection with explicit ambiguity
    -> strike window, goal reasoning, SituatedState and coaching
```

Skeleton logic:

```python
# Pseudocode: detector/tracker adapter interfaces have not been implemented.
for frame, timestamp in video:
    candidates = detector.detect(frame)  # initially full-frame + slices
    camera_transform = camera_motion.estimate(frame)
    prediction = tracker.predict(timestamp, camera_transform)
    matches = associate(candidates, prediction, appearance, expected_size)

    if unique_match_is_supported(matches):
        tracker.observe(matches.best, timestamp)
        evidence = observed_ball(matches.best)
    elif prediction_is_still_useful(prediction):
        evidence = predicted_ball(prediction)  # never label as seen contact
    else:
        tracker.mark_lost()
        evidence = unknown_ball()

    contact_candidates.update(pose, evidence)
```

Important design details:

- A manual initial ball click is useful as an **experimental control**: compare automatic initialization against a correctly seeded tracker. It need not become the final athlete workflow.
- Appearance checks should compare candidate image patches with trusted ball examples and boot/turf negatives. Simple colour or circularity tests alone are not sufficient under blur or changing illumination.
- Retain multiple plausible candidates when evidence is ambiguous. Do not promote an arbitrary winner merely to keep a continuous trail.
- Use camera-motion estimation from background features when needed, excluding moving players and the ball. If estimation fails, widen uncertainty or return unknown. Do not apply a ground-plane homography as a true physical trajectory for an airborne ball.
- Do not require a smooth constant-velocity path through a kick or bounce. Those are real motion changes; preserve uncertainty and allow reacquisition rather than smoothing them away.
- Predictions and trajectory inpainting cannot verify foot contact, goal entry, ball speed or technique.
- Do not hard-exclude the foot area: the real ball is near the foot precisely when we need it most.
- Do not build the trajectory around the old ankle-speed strike estimate. First obtain independent ball evidence, then infer candidate events.
- Model class IDs differ. Our existing COCO filter uses class 32; a dedicated one-class ball model may use 0. Read model metadata and configure the adapter rather than silently keeping the old filter.
- Keep bounding-box centres, normalised coordinates, crop offsets and letterbox scaling explicit. Evaluate in original-image coordinates before comparing with pose.

## Concrete experiment order

### First: compare detectors on identical labelled frames

Use the current three clips as development examples. Label visible ball centres/boxes across approach, contact and early flight; label occluded frames as unobservable, not as guessed centres. Include boots, field markings and background objects as negative examples. Reserve additional whole clips, athletes or sessions for testing; adjacent frames from the same recording should not be randomly split between training and test sets.

Compare:

1. Current generic YOLO, full frame.
2. Same weights with sliced inference, isolating the resolution change.
3. Dedicated soccer-ball weights with the same slicing setup.
4. WASB soccer weights as a separate temporal-model benchmark.

TrackNet is a subsequent alternative if these results justify further training effort. This ordering is my practical recommendation, not a published ranking on SCAN data.

### Second: evaluate identity and tracking

Compare automatic and manually verified initialization. Count correct visible-ball localizations, false boot/turf selections, identity changes, losses and reacquisition delays. Inspect contact windows separately from overall clip performance.

Report precision and recall, centre error normalised by annotated ball size, visible-frame coverage, false localization rate when the ball is absent/occluded, and processing time. Detection rate alone is not success. Declare the spatial tolerance before comparing systems and avoid selecting it to favour one model.

### Third: evaluate contact and shot selection

Only after identity is stable, compare estimated contact intervals against frames 79–80, 117–118 and 136–137. Report interval timing error, missed shots, dribble touches falsely called shots and abstentions. Evaluate first touches, running steps, dribbling and final shots separately.

### Then: adapt to SCAN footage if needed

If pretrained models repeatedly confuse boots and balls, collect labelled phone footage across light, distance, blur and occlusion conditions; fine-tune with hard negatives. The number of labels needed is an empirical question, not a guaranteed fixed count. Training/fine-tuning and GPU/runtime evaluation are separate work from wiring inference into the app.

## Dataset caution

SoccerTrack and SoccerNet are useful soccer resources, but their camera views differ from SCAN's phone clips. Check annotation meaning before using a downloaded trajectory as measured ball truth: the SoccerTrack-v2 dataset card explicitly describes its auxiliary ball track as an event-anchored reconstruction interpolated between anchors. That is unsuitable as per-frame visible-ball ground truth for our localization test. [Dataset card](https://huggingface.co/datasets/atomscott/soccertrack-v2).

## Practical choice

Start with **the Roboflow dedicated-ball/sliced-detection skeleton** and **WASB's soccer checkpoint** in isolated experiments. Compare them against the existing generic detector before rewriting production tracking. Then improve association using whichever detector actually recovers the ball in our clips.

This is a research-backed plan, not proof either model will work. Neither the published accuracy numbers nor a smooth-looking overlay establish dependable contact detection on SCAN footage. Check code, checkpoint and dataset usage terms separately before redistributing assets.
