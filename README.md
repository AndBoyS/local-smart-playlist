# local-smart-playlist

Mood-based smart playlists for a local music library. MuQ-MuLan audio embeddings →
rank tracks by text queries like `melancholic`, `moody late night`.

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

Text queries rank tracks by **sustained-mood contrast** (default):
each stored 10 s window is scored as a margin — `cos(window, query) −
cos(window, broad-mood baseline)` — where the baseline is the mean of ~20
wide-coverage mood anchors. The margin cancels the timbre-dependent audio→text
similarity scale (quiet ambient vs dense rock compete on one scale), and the
track score is the **median** margin over its windows, so most of the track
must fit; ties break by the share of positive-margin windows. Full-library
scan over stored windows; tracks without stored windows must be re-indexed.

`--seed-track` keeps the two-stage audio ranking:
track-mean KNN prefilter, then exact peak-window rescore. Seed score =
alpha * max cos(query, window) + (1-alpha) * cos(query, track mean);
`--alpha` tunes peak-mood vs whole-track mood (default 0.7).

Index DB defaults to `.sp-index/index.db` relative to the current working
directory; override with `--db`.

LLM query correction (`--llm`) uses any OpenAI-compatible endpoint. The LLM
only adapts the query phrase for embedding (translation, typo/phrase
normalization) — it never adds genres or moods not implied by the input, and
the query stays one line. A failed LLM call falls back to the raw phrase:

```
SP_LLM_BASE_URL  # default https://opencode.ai/zen/go/v1 (OpenCode Go)
SP_LLM_API_KEY   # default: $OPENCODE_API_KEY
SP_LLM_MODEL     # default deepseek-v4-flash
```

Model test suite (MuQ-MuLan download) is marked `model` and skipped by default:
`uv run pytest -m model` to run locally.
