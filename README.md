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
- `sp index <root> [--rescan] [--progress] [--db PATH]`
- `sp play "<query>" [-n 30] [-o out.m3u8|-] [--llm] [--seed-track PATH] [--dry]`
- `sp status [--db PATH]`
- `sp describe PATH [-n 10] [--db PATH]` — top caption-vocab neighbors for an indexed track (offline; no LLM)

Playlists are written to `<library root>/[Playlists]/<query slug>.m3u8` with
paths relative to the playlist file, so they stay portable across synced
copies of the library (e.g. Dropbox → Android via FolderSync, consumed by
AIMP). Use `-o` for a custom destination, `-o -` to print the m3u8 to stdout.

Search is two-stage: track-mean KNN prefilter, then exact peak-window rescore.
Score = alpha * max cos(query, window) + (1-alpha) * cos(query, track mean);
`--alpha` tunes peak-mood vs whole-track mood (default 0.7).

Index DB defaults to `.sp-index/index.db` relative to the current working
directory; override with `--db`.

LLM caption expansion (`--llm`) uses any OpenAI-compatible endpoint, by
default OpenCode Go; the prompt is grounded in the readout vocabulary
(captions in the embedding model's text-caption style, `local_smart_playlist/data/caption_vocab.txt`):

```
SP_LLM_BASE_URL  # default https://opencode.ai/zen/go/v1 (OpenCode Go)
SP_LLM_API_KEY   # default: $OPENCODE_API_KEY
SP_LLM_MODEL     # default deepseek-v4-flash
```

Model test suite (MuQ-MuLan download) is marked `model` and skipped by default:
`uv run pytest -m model` to run locally.
