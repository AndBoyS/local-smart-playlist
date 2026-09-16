# local-smart-playlist

Mood-based smart playlists for a local music library. CLAP audio embeddings →
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

Playlists are written to `<library root>/[Playlists]/<query slug>.m3u8` with
paths relative to the playlist file, so they stay portable across synced
copies of the library (e.g. Dropbox → Android via FolderSync, consumed by
AIMP). Use `-o` for a custom destination, `-o -` to print the m3u8 to stdout.

Index DB defaults to `$XDG_DATA_HOME/sp/index.db` (macOS:
`~/Library/Application Support/sp/index.db`); override with `--db`.

LLM caption expansion (`--llm`) uses any OpenAI-compatible endpoint, by
default a local ollama:

```
SP_LLM_BASE_URL  # default http://127.0.0.1:11434/v1
SP_LLM_API_KEY   # optional
SP_LLM_MODEL     # default llama3.2
```

Model test suite (CLAP download) is marked `model` and skipped by default:
`uv run pytest -m model` to run locally.
