# SCAN — Soccer Cognitive AI Network

SCAN is an AI-driven adaptive soccer training coach that decides *what* to say,
*when* to say it, and *how much* detail to give a player during solo shooting
and first-touch drills — based on their real-time technical and cognitive
state, instead of giving constant commentary.

## Why

Constant feedback overloads a player's germane cognitive load and drowns out
the signal that actually matters. SCAN treats coaching as a decision problem:
withhold feedback unless a hazard, a repeated error pattern, or a natural rep
boundary justifies breaking silence.

## Architecture

1. **Feature extraction** (`src/feature_extraction.py`) — CV pipeline turns
   raw drill footage into `RepFeatures` per repetition.
2. **Controller** (`src/controller.py`) — priority-ordered decision loop:
   hazard override → protect germane load → repeated-error summary →
   end-of-rep cue → default silence.
3. **LLM feedback** (`src/llm_feedback.py`) — converts a `Decision` into a
   natural-language (or spoken) coaching cue.
4. **Player state** (`src/player_state.py`) — tracks rep history, error
   counts, and intervention history across a session.

## Status

- [x] Core decision loop (`controller.py`) with unit tests
- [x] Player state tracking
- [ ] Real CV feature extraction (Phase 1)
- [ ] LLM-generated natural-language cues (Phase 2)
- [ ] Within-subjects evaluation study vs. always-on baseline (Phase 3)
- [ ] Demo video + packaging (Phase 4)

**Current result:** 55% reduction in feedback volume vs. an always-on
baseline, with zero missed hazard-type errors (pilot test).

## Setup

```bash
pip install -r requirements.txt
pytest tests/
```

## Roadmap

See project docs for the full phase plan (real footage → controller →
LLM cues → evaluation study → demo packaging). AR/VR (Unity) integration is
scoped as future work, not a build target for this year.
