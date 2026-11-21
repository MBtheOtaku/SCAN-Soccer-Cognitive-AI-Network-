# Training intention and shooting measurements

Each clip can carry a declared intention of power, curl or placement, an optional target zone, and an optional player-reported outcome. General is the default for older recordings. These fields describe the task; perception supplies the evidence used to assess it.

The current app saves this context with the recording, sends it during analysis and shows a separate measurement report. It also supports editing existing clips. Editing context clears the displayed analysis and requires another analysis request. Choosing an intention does not create a measurement or a professional rating.

## Formulas and required evidence

The formulas below are geometric definitions for future validated measurements. They are not validated soccer technique scores. Current uploads supply no calibrated physical flight measurements, so these values remain unavailable.

World coordinates use forward, lateral and upward axes in metres. Velocity uses metres per second. A goal-plane position uses horizontal and upward coordinates in metres. Raw pixel coordinates and a ground-plane homography cannot directly supply the position of an airborne ball in this coordinate system.

| Intention | Measurement | Definition | Evidence required |
| --- | --- | --- | --- |
| Power | Release speed in km/h | `3.6 * sqrt(v_forward² + v_lateral² + v_up²)` | Validated metric 3D velocity at independently verified release timing |
| Power | Flight interval in seconds | `t_end - t_start` | Verified timestamps and flight endpoints |
| Power | Mean endpoint transit speed in km/h | `3.6 * distance_3d(start, end) / elapsed_seconds` | Calibrated 3D endpoints and verified elapsed time; this is the magnitude of mean velocity, not launch speed or mean speed along a curved path |
| Curl | Signed horizontal heading change in degrees | `degrees(atan2(cross(v0_horizontal, v1_horizontal), dot(v0_horizontal, v1_horizontal)))` | Validated horizontal velocity directions in continuous free flight; principal signed difference from -180 to 180 degrees |
| Curl | Signed lateral deviation in metres | `cross(unit(v0_horizontal), end_horizontal - start_horizontal)` | Calibrated positions and initial horizontal heading; positive follows the calibrated lateral axis |
| Placement | Target distance in metres | `sqrt((hit_horizontal - target_horizontal)² + (hit_up - target_up)²)` | Measured goal-plane intersection, calibrated target and evidence that the target was declared before the attempt |

All metric packets require explicit identity, calibration, camera-motion, depth and observation checks. Predicted or extrapolated samples cannot satisfy them. Curl also requires an uninterrupted flight segment, without a bounce or deflection. A measured goal-plane intersection alone does not establish a goal: ball extent, crossing evidence and outcome remain separate.

Dip is a separate future flight descriptor. It does not measure kicking power. Neither endpoint transit speed nor shot travel time measures mechanical power in watts. Projected image curvature does not establish physical curl or ball spin. Asai and colleagues measured curve trajectories using 24 cameras and marked balls; their two participants were collegiate players, not a professional reference cohort. [Asai et al. 2020](https://www.mdpi.com/2504-3900/49/1/88)

## Professional reference context

The power profile includes one descriptive cohort from Ben Brahim et al. Seventeen young male professionals from one Tunisian league team, aged 20.7 ± 1.0 years, produced 83.18 ± 6.61 km/h in the study's 12-hour recovery condition. The protocol used a stationary maximal instep shot toward a 1 × 1 m target, five attempts with one-minute rests, and retained each player's fastest radar measurement. This is a specific cohort statistic, not a professional pass threshold. It is not directly comparable with video endpoint transit speed. [Ben Brahim et al. 2023](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0294867)

Professional comparisons remain disabled. The app exposes the cohort, protocol and source under measurement details without calculating a personal rating. No professional numeric curl or placement threshold is configured.

Speed and target accuracy must remain separate: instructions and target position can change their relationship. The placement references support measuring target error and recording intention, rather than treating every goal as equally precise. [Hunter et al. 2018](https://pubmed.ncbi.nlm.nih.gov/29580692/), [Sørensen et al. 2024](https://www.mdpi.com/2076-3417/14/22/10668)

## API and reasoning boundaries

`POST /analyze` accepts an optional multipart text field named `training_intention`, containing JSON:

```json
{"shot_type":"placement","target_zone":"upper_left","reported_outcome":"goal"}
```

Supported shot types are `unspecified`, `power`, `curl` and `placement`. Targets are the nine zones shown in the app, from the player's view toward the goal. Reported outcomes are `goal`, `miss`, `woodwork` and `saved`, or null. The parser rejects extra fields and client-supplied measurements. The response and saved analysis JSON include `training_intention` and deterministic `intention_feedback`.

SituatedState carries task context into Jev judgements and language synthesis. Player-reported outcomes retain their attribution and never populate machine-observed goal fields. Retrospectively editing a target cannot by itself satisfy the predeclared-target evidence check. The controller and speech policy are unchanged.

The metric evaluator's internal `measurements` argument is reserved for a future server perception adapter. Its quality flags are requirements that adapter must substantiate, not proof created by setting flags to true. No current adapter derives physical speed, curl or target error from the experimental tracker. Deterministic intention reports remain available when external APIs fail.

## Verification

Backend checks cover formula values and units, finite inputs, missing calibration, predicted paths, bounce/deflection exclusions, retrospective targets, request isolation and external API fallback. Mobile checks cover legacy metadata, interrupted saves, context preservation and stale response rejection. Native recording and replay still require a phone smoke test after restarting the backend and Metro.
