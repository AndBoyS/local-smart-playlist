# Current difficulties

Working notes on ranking quality. Updated 2026-09-18 after switching `sp play`
text queries to the sustained-mood contrast ranker (`query/contrast.py`) and
removing the caption-document pipeline (`--ranker docs`, `sp describe`
kept as readout-only diagnostic). Numbers from the live index
(`.sp-index/index.db`, ~6150 windowed tracks), scripts were scratch.

## Current pipeline

Text query → unit query vector (embedded exactly as written — the old
`{q} mood.` variant was removed, it moved comma queries off soft textures) →
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

### 2. LLM query adaptation actively harms (fixed)

`--llm` used to rewrite already-good comma lists: `'dreamy, melancholic' ->
'dreamy mood, melancholic mood.'`. Measured: TFS margin 0.120 → 0.040, rank
799 → 2012. The added `mood` padding + rewording moves the vector off soft
textures (difficulties §4 style constraint still applies). Raw comma lists
beat every adapted form.

Fixed in `query/prompts.py` (`_ADAPT_PROMPT`) and
`query/contrast.py` (`query_vector_contrast`): the prompt passes
comma-attribute inputs through unchanged (typos/translation only, never
append "mood"/genre words) and only rewrites free text into attributes;
the static `"{q} mood."` embedding variant was removed — queries embed
exactly as written.

Padding re-check under the vocab-cal ranker (2026): appending `mood`
(`'dreamy mood, melancholic mood.'`) is no longer uniformly worse — TFS
rank 33 → 164 but Terminal Show 61 → 4, Celeste 53 → 30 — i.e. the padding
penalty was a margin-ranker artifact; under vocab-cal it is just noise,
not a consistent win. Attribute *invention* still strongly harms:
`'melancholic mood, slow tempo, sparse piano.'` is a different query
(cos 0.34), TFS 33 → 950, score 0.698 → dropped below the 0.9 cutoff;
the prompt's no-invention rule is the load-bearing part.

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
| ~~LLM pass-through prompt (§2)~~ | done — prompt passes comma lists through | one prompt edit |
| min-score default rethink (§3) | soft textures survive default cutoff | default change |
| anchor bank edits | baseline quality; e.g. more texture-neutral anchors | data edit |
| LLM reranker (parked by user) | per-track precision on top ~100 | LLM call per query |
| seed-track mode for texture queries | TFS-like "give me more like this" already works via `--seed-track` | free |

## Suggested next step

~~Fix the `--llm` pass-through prompt~~ — done: prompt passes comma lists
through unchanged and the static `{q} mood.` variant was removed. Remaining:
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
  harness (`scripts/eval_rank.py`) stays (it now runs the production
  vocab-cal ranker, so §6's margin-rank table is historical). §2 is now
  fixed in code (below), not just by convention.

§2 later fixed in code (`query/prompts.py` `_ADAPT_PROMPT`): the prompt
passes comma-attribute lists through unchanged (typos/translation only)
and the static `"{q} mood."` embedding variant was removed — `--llm` for
comma lists now equals the raw-phrase playlist. (Older finding, for the
record: the pass-through hack in `adapt_mood` was first reverted in favor
of convention — flag semantics were kept honest and users just didn't
pass `--llm` for canonical attribute lists; measured harm: TFS margin
0.120 → 0.040, rank 799 → 2012.)

## §7 Caption-vocab calibration ranker (2026 follow-up, shipped)

Production ranker: `sp play` scores tracks with caption-vocab calibration
(lever #5, CAF-Score style; `query/vocab_cal.py`), not additive margin. The
legacy margin ranker remains in `query/contrast.py` for historical tests; it
is not part of production text-query scoring.

Mechanism: per window, score = percentile of cos(w, query) among that
window's cos(w, caption) values over the 215-caption `sp describe` vocab
(self-referential bar — the window's own caption profile, not 20
hand-picked anchors). Track score = mean window percentile; ties by
coverage (share of windows > 0.5). Every track uses this same percentile
calculation, including low-caption-affinity tracks; there is no margin
fallback. The earlier <0.30 fallback was removed after checking low-affinity
playlist membership: for `ethereal, melancholic`, forced percentile scoring
added one track at the 0.90 cutoff; for `energetic`, membership was unchanged.

Cutoff semantics changed with the scale: scores are absolute calibrated
probabilities, so `--min-score` is now an absolute threshold (default
0.9), floored at 0.5 (neutral). The old `min_score × best` shape is gone
— it made no sense on a calibrated scale and re-introduced §3's
texture bias via `best`.

Live-index measurements (`scripts/eval_vocab_cal.py`, 6152 tracks):

| probe | exemplar | contrast rank | vocab-cal rank |
|---|---|---|---|
| dreamy, melancholic | TFS | 800 (median-margin era) / 1957 (re-measure) | **38** (0.986) |
| dreamy, melancholic | Terminal Show | 1 | 112 (0.975) |
| calm piano | Celeste | 1 | 6–8 (0.961–0.969) |

Pass rates at the 0.9 cutoff: `dreamy, melancholic` 18.0%, `calm piano`
0.9%. Top of `dreamy, melancholic` is now soft game-OST/ambient
(SIGNALIS, TUNIC, Silent Hill 2 piano pieces, Binding of Isaac calm
tracks, Portal 2) — the dense-texture bias at the top is visibly broken.
Within-cohort ordering above TFS is percentile-consistent, i.e. the
§1 absolute-scale gap is *bypassed at decision level*, not proven fixed:
TFS still sits below ~0.6% of the library, but that residue is no longer
texture-correlated by construction of the bar.

Known trade-offs: vocab re-embedded per run (215 captions, seconds);
low-affinity caption profiles can still produce extreme percentiles;
`dark, aggressive` probe unmeasured (no exemplars).
