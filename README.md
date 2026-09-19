# local-smart-playlist

Mood-based smart playlists for a local music library. MuQ-MuLan audio
embeddings → rank tracks by text queries like `melancholic`,
`moody late night`. Describe what you want in plain words — a single word,
a comma list of attributes, or a free-text vibe — and `sp play` turns it
into a playlist; with `--llm`, any OpenAI-compatible endpoint aligns your
phrasing with the embedding model's caption style before ranking.

```console
uv sync
uv run sp index "/Users/user/Library/CloudStorage/Dropbox/Music/Selected"
uv run sp play "melancholic" -n 30
uv run sp status
```

- CLI binary: `sp`
- `sp index <root> [--rescan] [--quiet] [--db PATH]`
- `sp play "<query>" [-n 30] [-o out.m3u8|-] [--llm] [--seed-track PATH] [--dry] [--min-score 0.75]`
- `--min-score` is relative: keeps tracks scoring ≥ max(0.02, fraction × top score), default 0.6; margins are query-scale dependent, so no fixed absolute threshold works
- `uv run python scripts/eval_rank.py` — offline eval over the live index: ranks of known soft/dense exemplar tracks per probe query, no playlist written (read-time correction experiments and their measured outcomes are recorded in difficulties.md §6)
- `sp status [--db PATH]`
- `sp describe PATH [-n 10] [--db PATH]` — top caption-vocab neighbors for an indexed track (offline; no LLM; diagnostics only — ranking never uses captions)

Playlists are written to `<library root>/[Playlists]/<query slug>.m3u8` with
paths relative to the playlist file, so they stay portable across synced
copies of the library (e.g. Dropbox → Android via FolderSync, consumed by
AIMP). Use `-o` for a custom destination, `-o -` to print the m3u8 to stdout.

Text queries rank tracks by **caption-vocab calibration** (default): each
stored 10 s window is scored by the percentile of `cos(window, query)` among
that window's own similarities to the 215 readout captions (`sp describe`
vocabulary) — a self-referential bar that cancels most of the
timbre-dependent audio→text similarity scale (soft ambient and dense rock
calibrated against their own texture, not an absolute scale). The track
score is the **mean per-window percentile**, so most of the track must fit;
ties break by the share of windows above the 0.5 neutral point. Tracks whose
best caption affinity is far below the library norm (uninformative profiles)
fall back to a sigmoid of the broad-mood margin against ~20 wide-coverage
anchors. Full-library scan over stored windows; tracks without stored
windows must be re-indexed. Scores are absolute calibrated probabilities:
0.5 = the query fits as well as a typical caption, 0.9+ = clearly on-mood;
`--min-score` is that absolute cutoff (default 0.9), floored at 0.5.

`--seed-track` keeps the two-stage audio ranking:
track-mean KNN prefilter, then exact peak-window rescore. Seed score =
alpha * max cos(query, window) + (1-alpha) * cos(query, track mean);
`--alpha` tunes peak-mood vs whole-track mood (default 0.7).

Index DB defaults to `.sp-index/index.db` relative to the current working
directory; override with `--db`.

LLM query alignment (`--llm`) uses any OpenAI-compatible endpoint. It adapts
your phrase for embedding only: free text becomes a comma-attribute caption;
an already-comma-separated attribute list passes through unchanged (typo and
translation fixes only). It never adds genres or moods not implied by the
input — appended padding like `mood` measurably worsens retrieval. A failed
LLM call falls back to the raw phrase:

```
SP_LLM_BASE_URL  # default https://opencode.ai/zen/go/v1 (OpenCode Go)
SP_LLM_API_KEY   # default: $OPENCODE_API_KEY
SP_LLM_MODEL     # default deepseek-v4-flash
```

Model test suite (MuQ-MuLan download) is marked `model` and skipped by default:
`uv run pytest -m model` to run locally.
