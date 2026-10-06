# SCAN Development Instructions

SCAN = Soccer Cognitive AI Network.

SCAN is an AI-driven adaptive soccer training system that combines
computer vision, probabilistic judgement, coaching policy, and
language generation to analyse soccer training repetitions.

## Architecture

Video
→ perception
→ SituatedState
→ Jev judgements
→ controller / policy
→ coaching synthesis
→ UI / voice

### Layer responsibilities

**Perception**
- Detects and measures what happened.
- Includes pose, ball tracking, goal tracking, and derived metrics.
- Perception outputs are evidence, not coaching conclusions.

**SituatedState**
- Combines relevant body, task, environment, outcome, and history state.
- This is the structured representation consumed by later reasoning layers.

**Jev**
- Produces probabilistic judgements from structured evidence.
- Jev does not control coaching policy.
- Treat probabilities as uncertainty, not ground truth.

**Controller / Policy**
- Determines whether intervention is appropriate.
- Protects cognitive load and prevents unnecessary feedback.
- Policy decisions must remain separate from language generation.

**Coaching Synthesis**
- Converts justified structured conclusions into natural coaching language.
- The LLM may improve wording and interpretation.
- The LLM must never invent observations or biomechanics.

**UI / Voice**
- Detailed analysis belongs in the post-repetition UI.
- Spoken feedback should remain short and attention-aware.
- Written analysis and spoken cues serve different cognitive purposes.


## Core Rules

- Always spell "judgement" and "judgements" with an e.
- Do not let an LLM invent biomechanics or unsupported soccer observations.
- Do not manufacture criticism when no correction is supported.
- Jev provides probabilistic judgements; it does not determine policy.
- The controller owns whether feedback/intervention should occur.
- Separate technique quality from shot outcome.
- A successful outcome does not prove perfect technique.
- Preserve uncertainty when perception or judgement confidence is weak.
- Detailed feedback belongs in the post-rep UI.
- Spoken feedback should remain short and focused.
- Preserve deterministic fallbacks when external APIs fail.
- External API failure must never break the core analysis pipeline.
- Do not refactor working perception or tracking code unless the task requires it.
- Prefer small, targeted changes over broad rewrites.
- Preserve existing working behaviour unless a task explicitly requires changing it.


## Current Feedback Philosophy

Good coaching does not require a correction after every repetition.

When evidence for a specific correction is weak:
- do not invent one;
- emphasize repeatability, consistency, or additional observation;
- allow silence when appropriate.

The written report may explain.

The spoken cue should direct attention.


## Frontend

The frontend is a Next.js application.

Before changing Next.js code, follow the generated Next.js agent
instructions at the top of this file and consult the appropriate
documentation inside `node_modules/next/dist/docs/`.

Do not rewrite working frontend components merely for stylistic reasons.

Preserve the existing SCAN visual identity unless the task explicitly
asks for a redesign.


## Backend

The backend uses FastAPI.

Important backend components include:
- pose / player perception
- ball tracking
- goal tracking
- SituatedState
- Jev judgements
- controller / policy
- deterministic coaching report generation
- LLM coaching synthesis

Maintain separation between these responsibilities.


## Reliability

SCAN should degrade gracefully.

If:
- Jev fails,
- the OpenAI API fails,
- credentials are unavailable,
- or an external service times out,

use the appropriate deterministic fallback where one exists rather than
breaking the complete analysis request.


## Secrets

Never commit API keys, tokens, credentials, or secrets.

Secrets such as:
- `TYPESAFE_API_KEY`
- `OPENAI_API_KEY`

must remain in environment variables or ignored local environment files.


## Development Behaviour

Before making substantial changes:
1. Inspect the relevant existing implementation.
2. Understand how the change fits into the SCAN architecture.
3. Avoid unrelated refactors.
4. Preserve existing fallbacks and safety constraints.
5. Run the relevant tests, type checks, or build checks after editing.

When changing both backend and frontend API contracts, update and verify
both sides together.