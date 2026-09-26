# local-smart-playlist

Mood-based smart playlists for a local music library. MuQ-MuLan audio embeddings +
LLM query expansion → rank tracks by text queries like "melancholic", "moody
late night".

## Stack

- Python 3.13, managed with **uv** (`uv_build` backend)
- **MuQ-MuLan** (`OpenMuQ/MuQ-MuLan-large`, 512-dim joint audio/text space,
  24 kHz input, CC-BY-NC weights) via the `muq` pip package + torch
- **LSP index**: SQLite via SQLAlchemy ORM, NumPy full-scan vector ranking
- Optional LLM caption expansion (OpenAI-compatible endpoint / ollama)
- CLI: **typed-argparse**
- Lint: **ruff**, Types: **pyrefly** (strict), Tests: **pytest**, CI: GitHub Actions

## Pipeline (track → vector)

```
file → decode (mono, float32) → resample 48 kHz
     → RMS-trim silence → disjoint 10 s windows
     → resample 48→24 kHz → batch → MuQ-MuLan
     → per-window 512-dim vecs → store
     → track vec = mean / p90 (both kept)
```

Vector store resumable: tracks already in DB skipped on re-scan.

## Query pipeline

1. Mood word → LLM expands to 5-10 caption-style prompts ("a slow melancholic
   song with soft vocals, sparse piano, minor key...") → average embed
2. Score track = max over windows cos(query, window_vec) (peak-mood matching;
   track-mean as secondary signal)
3. Rank → wrap to `.m3u8`

Optional index-time enrichment: LLM caption from metadata/lyrics blended
α≈0.8 audio / 0.2 text.

## Structure

```
local-smart-playlist/
├── pyproject.toml
├── .python-version
├── .gitignore               # .venv, .lsp-index/, dist, .pytest_cache, .ruff_cache
├── README.md
├── AGENTS.md
├── plan.md
├── local_smart_playlist/
│   ├── __init__.py
│   ├── main.py                      # typed-argparse root program
│   ├── config.py                    # paths: db, model cache
│   ├── audio/
│   │   ├── decode.py                # decode → mono float32, soxr resample
│   │   ├── windowing.py             # RMS-trim, 10 s windows
│   │   └── features.py              # mel-spec prep, batch assembly
│   ├── embed/
│   │   ├── model.py                 # lazy MuQ-MuLan load, batched window embed
│   │   └── aggregate.py             # mean/p90 track vectors
│   ├── index/
│   │   ├── library.py               # file discovery, stable track ids
│   │   └── store.py                 # ORM schema, vector storage, resumable scan
│   ├── query/
│   │   ├── prompts.py               # mood → LLM caption expansion
│   │   ├── search.py                # text embed, max-window scoring, rank
│   │   └── playlist.py              # results → m3u8 writer
│   └── commands/
│       ├── index_cmd.py             # sp index ~/Music [--rescan]
│       ├── playlist_cmd.py          # sp play "melancholic" -n 30 -o x.m3u8
│       └── status_cmd.py            # coverage stats
├── tests/
│   ├── test_windowing.py
│   ├── test_search.py
│   └── fixtures/                    # small generated wavs
└── .github/workflows/ci.yml         # ruff + pyrefly + pytest
```

## CLI (typed-argparse)

Subcommands via typed-argparse program:

```console
sp index ~/Music [--rescan] [--quiet]
sp play "melancholic" [-n 30] [-o out.m3u8] [--seed-track path] [--dry]
sp status
```

## Dependencies

```
transformers, torch, soundfile, numpy, sqlalchemy, typed-argparse, rich
dev: pyrefly, ruff, pytest, types-requests
optional: httpx (LLM expansion client)
```

## Build order

1. Scaffold: uv init, deps, ruff/pyrefly config (copied from bandcamp-dl), git
2. `audio/` + unit tests (pure functions, no torch)
3. `embed/` + MuQ sanity test on fixture wav
4. `index/store.py` schema + resumable scan
5. `index_cmd.py` end-to-end indexing
6. `query/` direct-text search (skip LLM) + `play_cmd`
7. LLM prompt expansion behind flag
8. CI
