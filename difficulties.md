# Current difficulties

Working notes on ranking quality. Updated 2026-09-18 after switching `sp play`
text queries to the sustained-mood contrast ranker (`query/contrast.py`) and
removing the caption-document pipeline (`--ranker docs`, `sp describe`
kept as readout-only diagnostic). Numbers from the live index
(`.sp-index/index.db`, ~6150 windowed tracks), scripts were scratch.

## Current pipeline

Text query → unit query vector (embedded as `{q}` and `{q} mood.`, averaged) →
per-window margin `cos(window, query) − mean_i cos(window, anchor_i)` over 20
broad mood anchors (`MOOD_ANCHORS`, en+zh) → track score = **median** margin
(coverage = share of positive windows, tiebreak) → cutoff
`max(0.02, min_score × best)`.

## What the contrast ranker fixed

- γ dilemma gone: no evidence weighting, margins compete on one scale.
- Additive baseline is the right correction family: a **unit** centroid
  baseline (first attempt) inflated the broad baseline by `1/‖centroid‖`
  (~2×, ‖centroid‖=0.477 over 20 anchors) and buried soft textures; the
  unnormalized mean of per-anchor affinities is the correct "music in
  general" reference. After the fix, TUNIC tracks surface for
  `dreamy, melancholic` (Sunset Breakfast +0.174 … coverage 1.00).
- Sustained median defeats the peak-window false-positive problem by
  construction (§3 of the old notes no longer applies — windows are scored
  directly, no mean-vec smearing).

## Remaining problems, measured today

### 1. Residual absolute-scale gap: dense vs soft textures (open, core)

MuQ-MuLan gives dense-produced windows much higher text affinity than soft
game-synth/ambient. For `dreamy, melancholic`:

| track | med cos(w,q) | mean-anchor aff | margin |
|---|---|---|---|
| Silent Hill — The Terminal Show | 0.432 | 0.122 | +0.311 |
| TUNIC — To Far Shores | 0.266 | 0.144 | +0.120 |

Baseline affinity is *similar* (0.12 vs 0.14) — the bias is not additive, so
additive subtraction can't cancel it. The gap is model-intrinsic; margin
ranking is honest but TFS sits ~rank 800/6152 (best interpretation: `dreamy`
alone, rank 329, +0.140). Sub-window max/p75 blends don't move it (799–884).

Important: this is *not* a variation problem. To Far Shores is consistent —
uniformly dreamy/wistful across all 18 windows (coverage 1.00, per-window
margins ≈ +0.08…+0.19, no standout passage). Median ≈ its every-window
behavior. Its deficit is purely the lower absolute q-affinity the model
assigns to soft game-synth texture; no aggregation change can recover a
signal the model never produced.

### 2. LLM query adaptation actively harms (unfixed)

`--llm` rewrites already-good comma lists: `'dreamy, melancholic' ->
'dreamy mood, melancholic mood.'`. Measured: TFS margin 0.120 → 0.040, rank
799 → 2012. The added `mood` padding + rewording moves the vector off soft
textures (difficulties §4 style constraint still applies). Raw comma lists
beat every adapted form. Fix candidate: adapt prompt should pass through
comma-attribute inputs unchanged (typos/translation only, never append
"mood"/genre words). Not yet implemented.

### 3. Relative min-score re-introduces texture bias (unfixed)

Default cutoff `max(0.02, 0.6 × best)` with best≈0.31 keeps only tracks
≥0.19 ≈ top 3% — which are exactly the dense/dark high-affinity tracks. A
"dreamy, melancholic" playlist built with defaults excludes every soft
texture. Margin semantics (0 = average-mood fit, +0.10 = clear fit) suggest a
much weaker relative default or a higher absolute floor. Not yet decided.

### 4. Regression correction rejected (measured)

Fitting per-query β and scoring `cos(w,q) − β·mean-anchor-aff`
(Q,B corr = 0.73, β≈1.5 for mood queries) penalizes soft textures *more*
(TFS rank 1349–2451). For texture queries (`calm piano`) β flips sign
(−1.2) and helps (Celeste +0.409). Unstable, query-dependent — rejected.
Additive margin stands.

### 5. Per-track relative scaling still collapses

Same as old §2: any per-track normalization of margins amplifies noise on
uniform tracks (all windows similar margin → z-scores meaningless). Not
pursued.

## Levers not yet tried

| lever | expected effect | cost |
|---|---|---|
| LLM pass-through prompt (§2) | recovers ~0.08 margin on adapted queries | one prompt edit |
| min-score default rethink (§3) | soft textures survive default cutoff | default change |
| anchor bank edits | baseline quality; e.g. more texture-neutral anchors | data edit |
| LLM reranker (parked by user) | per-track precision on top ~100 | LLM call per query |
| seed-track mode for texture queries | TFS-like "give me more like this" already works via `--seed-track` | free |

## Suggested next step

~~Fix the `--llm` pass-through prompt~~ — superseded by §6: resolved by
convention (don't pass `--llm` for canonical attribute lists). Remaining:
revisit §3 cutoff semantics before touching the margin formula again —
ranking order is now stable and sane; the cutoff decides what a playlist
*is*.

## §6 Read-time correction experiments (2026 follow-up, measured)

Implemented as opt-in flags (defaults unchanged) + `scripts/eval_rank.py`
harness; exemplar ranks on the live index (6152 tracks):

| variant | TFS (`dreamy, melancholic`) | Terminal Show | Celeste (`calm piano`) |
|---|---|---|---|
| baseline margin | **800** (+0.120) | 1 (+0.311) | **1** (+0.219) |
| `--norm cohort` (AS-Norm z-score) | 945 (+0.622) | 3 | 11 |
| `--debias` (density-direction projection) | 1912 (−0.011) | 1 | 113 |
| cohort + debias | 1810 | 2 | 72 |
| rank-norm (percentile vs anchor bank, scratch) | 1342 | 4 | 10 |

Findings:

- **None beats the additive margin.** Dividing by per-window anchor spread
  does not isolate texture scale — TFS's anchor-affinity spread is not
  small relative to its margin, so z-scoring just reshuffles noise; the
  z-scale also pollutes `calm piano` (metal remixes surface).
- The 6-pair soft-vs-dense projection direction is *not* the real bias axis:
  projecting it out removes query-relevant variance (Celeste 1 → 113) and
  the `calm piano` top-k fills with dense covers. Consistent with COMET
  (arXiv 2605.29628): the gap is not a low-dimensional mean/axis artifact.
- Conclusion: §1's absolute-scale gap is a whole-space property of
  MuQ-MuLan, not a correctable projection. All experiment code was
  reverted after measurement (ranker, CLI flags, debias module); the eval
  harness (`scripts/eval_rank.py`) stays. §2 was resolved by convention
  (below), no code change shipped.

§2 resolved by convention, not code: the pass-through hack (comma-list
detection in `adapt_mood`) was reverted — flag semantics stay honest, `--llm`
adapts whatever it gets, and the user simply doesn't pass it for canonical
attribute lists (measured harm: TFS margin 0.120 → 0.040, rank 799 → 2012).
