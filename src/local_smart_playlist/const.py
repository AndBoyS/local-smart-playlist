from pathlib import Path

REPO_DIR = Path(__file__).parents[2]
DATA_DIR = REPO_DIR / "src" / "local_smart_playlist" / "data"
if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"data directory not found: {DATA_DIR}")

CAPTION_VOCAB_PATH = DATA_DIR / "caption_vocab.txt"
if not CAPTION_VOCAB_PATH.is_file():
    raise FileNotFoundError(f"caption vocabulary file not found: {CAPTION_VOCAB_PATH}")
