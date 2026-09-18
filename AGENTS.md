# AGENTS.md

Guidelines for coding agents working in this repo.

## Environment

- Python 3.13, managed with uv. Run everything through `uv run ...`, never
  bare `python`/`pip`.
- Lint: `uv run ruff check .`, config in `pyproject.toml`.
- Types: `uv run pyrefly check` — strict preset, do not weaken it; annotate
  everything.
- Tests: `uv run pytest` (model tests deselected by default; `uv run pytest
-m model` runs the MuQ-MuLan sanity suite locally, requires network).

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
- NumPy shape annotations: numeric modules (`embed/aggregate.py`,
  `audio/features.py`, `audio/windowing.py`, `audio/decode.py`) use pyrefly shape
  types (`np.ndarray[[N, D]]` + `shape_extensions.IntVar`, one `IntVar` per line —
  tuple-assignment breaks the binding). `embed/model.py` torch boundary stays
  unannotated/`Any`. Every pyrefly suppression must name its error code —
  `# pyrefly: ignore[<code>]` — bare `# pyrefly: ignore` is not allowed (same
  rule as ruff's PGH). Known stub gaps, with the code to use at each site:
  - `# pyrefly: ignore[unsupported-operation]`: `__setitem__` on shape-typed
    arrays (facebook/pyrefly#4901), ndarray comparison dunders, reshape without
    reshape/astype.
  - `# pyrefly: ignore[unknown-argument-type]`: `np.where` fed `np.linalg.norm`
    output, `np.asarray`/`np.log`/`np.exp`/`sf.write` over unknown-typed args.
  Prefer the tracked method forms (`x.mean(axis=)`, `x.argmax()`) instead of a
  new ignore. If a suppression stops being needed, delete it — verify by
  removing it and re-running `uv run pyrefly check` (`unused-type-ignore = true`
  catches stale ones).
- Keep torch/transformers imports inside `embed/` and `audio/features.py`;
  everything else stays pure numpy/sqlite so tests run without the model.
