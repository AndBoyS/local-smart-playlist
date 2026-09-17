# AGENTS.md

Guidelines for coding agents working in this repo.

## Environment

- Python 3.13, managed with uv. Run everything through `uv run ...`, never
  bare `python`/`pip`.
- Lint: `uv run ruff check .`, config in `pyproject.toml`.
- Types: `uv run pyrefly check` — strict preset, do not weaken it; annotate
  everything.
- Tests: `uv run pytest` (model tests deselected by default; `uv run pytest
-m model` runs the CLAP sanity suite locally, requires network).

## Conventions

- Paths in and out of the CLI: `pathlib.Path` only.
- Audio: mono float32, 48 kHz everywhere after `decode.decode_mono`.
- Track identity: POSIX relative path from library root (stable across
  re-indexes). Never store absolute paths in the DB.
- Playlist files: UTF-8 m3u8, `#EXTM3U` + `#EXTINF` headers, paths relative
  to the playlist file location.
- DB access only through `index/store.py`; schema lives there.
- Embedding math: vectors are L2-normalized before storage; aggregation
  helpers in `embed/aggregate.py` re-normalize after mean/p90.
- Keep torch/transformers imports inside `embed/` and `audio/features.py`;
  everything else stays pure numpy/sqlite so tests run without the model.
