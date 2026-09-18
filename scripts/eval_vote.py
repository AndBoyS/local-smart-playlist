"""Scratch eval: anchor-competition voting variants vs production margin (difficulties.md §6 follow-up).

Read-only over the live index. Computes, per probe query, exemplar ranks and
top-5 previews for:

- V0  production median margin (reference)
- V1  hard argmax vote share, candidates = {query} + 20 anchors
- V1b same, twin-pruned (drop the single anchor most similar to the query)
- V2  softmax vote mass, p = exp(q/T) / sum exp(c/T), T in {0.02, 0.05, 0.1}
- V3  relevance probability p = sigmoid((cos_q - bar) / T), bar from the
      window's anchor-affinity distribution (mean/median/p25/p75/logsumexp),
      T = 0.03

Also prints query<->anchor affinity and per-window diagnostics for TFS.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np

from local_smart_playlist.embed.model import MODEL_ID, cap_torch_threads, embed_texts
from local_smart_playlist.index.store import Store, TrackRow
from local_smart_playlist.query.contrast import MOOD_ANCHORS, query_vector_contrast

PROBES: dict[str, list[str]] = {
    "dreamy, melancholic": ["To Far Shores", "The Terminal Show"],
    "calm piano": ["Celeste"],
}
TFS_NEEDLE = "To Far Shores"
PREVIEW_COUNT = 5
SOFTMAX_TAUS = (0.02, 0.05, 0.1)
SIGMOID_TAU = 0.03
LOGSUMEXP_TAU = 0.05
PERCENTILE_SCALE = 100.0


def _unit(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)  # pyrefly: ignore[unknown-argument-type]
    return np.asarray(vecs / safe, dtype=np.float32)


# Variant fn args: (per-window query cos (W,), per-window anchor cos (W, A)) -> track score.
Variant = tuple[str, Callable[[np.ndarray, np.ndarray], float]]


def argmax_share(q: np.ndarray, a: np.ndarray) -> float:
    """Share of windows whose argmax candidate is the query (last column)."""
    cand = np.concatenate([a, q[:, None]], axis=1)
    wins = np.asarray(cand.argmax(axis=1) == cand.shape[1] - 1)  # pyrefly: ignore[unknown-argument-type]
    return float(wins.mean())


def pruned_argmax_share(q: np.ndarray, a: np.ndarray, *, twin_idx: int) -> float:
    """Argmax share after dropping the twin anchor column."""
    kept = np.delete(a, twin_idx, axis=1)
    return argmax_share(q, kept)


def sigmoid_prob(q: np.ndarray, a: np.ndarray, *, bar: Callable[[np.ndarray], np.ndarray], tau: float) -> float:
    """Mean over windows of sigmoid((cos_q - bar(anchor affinities)) / tau)."""
    p = 1.0 / (1.0 + np.exp(-(q - bar(a)) / tau))
    return float(np.asarray(p).mean())  # pyrefly: ignore[unknown-argument-type]


def softmax_mass_fixed(q: np.ndarray, a: np.ndarray, *, tau: float) -> float:
    """Stable softmax mass for the query column."""
    cand = np.concatenate([a, q[:, None]], axis=1) / tau
    cand = cand - cand.max(axis=1, keepdims=True)
    cand = np.exp(cand)
    p = cand[:, -1] / cand.sum(axis=1)
    return float(np.asarray(p).mean())  # pyrefly: ignore[unknown-argument-type]


def make_variants(twin_idx: int) -> list[Variant]:
    bars: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "mean": lambda a: a.mean(axis=1),
        "median": lambda a: np.median(a, axis=1),
        "p25": lambda a: np.percentile(a, 25, axis=1),
        "p75": lambda a: np.percentile(a, 75, axis=1),
        "logsumexp": lambda a: np.asarray(LOGSUMEXP_TAU * np.log(np.exp(a / LOGSUMEXP_TAU).sum(axis=1))),  # pyrefly: ignore[unknown-argument-type]
    }
    variants: list[Variant] = [
        ("V0 median margin", lambda q, a: float(np.median(q - a.mean(axis=1)))),
        ("V1 argmax share", argmax_share),
        (
            f"V1b argmax share (drop twin #{twin_idx})",
            lambda q, a: pruned_argmax_share(q, a, twin_idx=twin_idx),
        ),
    ]
    variants.extend(
        (f"V2 softmax T={tau}", lambda q, a, tau=tau: softmax_mass_fixed(q, a, tau=tau)) for tau in SOFTMAX_TAUS
    )
    variants.extend(
        (f"V3 sig {name} T={SIGMOID_TAU}", lambda q, a, bar=bar: sigmoid_prob(q, a, bar=bar, tau=SIGMOID_TAU))
        for name, bar in bars.items()
    )
    variants.extend(
        (f"V3 sig mean T={tau}", lambda q, a, tau=tau: sigmoid_prob(q, a, bar=bars["mean"], tau=tau))
        for tau in (0.05, 0.1)
    )
    return variants


def _matches(track: TrackRow, needle: str) -> bool:
    needle_l = needle.lower()
    return needle_l in track.rel_path.lower() or needle_l in track.title.lower()


def tfs_diagnostics(*, query_vec: np.ndarray, anchor_vecs: np.ndarray, windows: np.ndarray) -> None:
    """Per-window picture for TFS: where the query sits inside the anchor distribution."""
    q = np.asarray(windows @ query_vec, dtype=np.float64).ravel()
    a = np.asarray(windows @ anchor_vecs.T, dtype=np.float64)
    am = a.mean(axis=1)
    print("  TFS per-window diagnostics:")
    print(f"    cos(q)   mean {q.mean():.3f}  min {q.min():.3f}  max {q.max():.3f}")
    print(f"    anchor aff  mean {am.mean():.3f}  spread +-2s [{am.mean() - 2 * am.std():.3f}, {am.max():.3f}]")
    print(f"    per-window anchors beaten by q: {np.asarray(q[:, None] > a).sum(axis=1).mean():.1f}/20")  # pyrefly: ignore[unsupported-operation]
    bars = (
        ("mean", am),
        ("p25", np.asarray(np.percentile(a, 25, axis=1))),
        ("p75", np.asarray(np.percentile(a, 75, axis=1))),
        ("max", a.max(axis=1)),
    )
    for name, bar in bars:
        wins = float(np.asarray(q > bar).mean())  # pyrefly: ignore[unsupported-operation]
        print(f"    windows q > anchor {name}: {wins * PERCENTILE_SCALE:.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--db", type=Path, default=Path(".sp-index/index.db"))
    args = parser.parse_args()

    cap_torch_threads()
    if not args.db.is_file():
        raise SystemExit(f"no index at {args.db}; run `sp index` first")

    with Store(args.db) as store:
        store.require_model(MODEL_ID)
        anchor_vecs = _unit(np.asarray(embed_texts(list(MOOD_ANCHORS)), dtype=np.float32))
        windows_all = store.load_windows(store.track_rel_paths())
        tracks: list[tuple[TrackRow, np.ndarray]] = []
        for rel_path in store.track_rel_paths():
            track = store.get_track(rel_path)
            window_vecs = windows_all.get(rel_path)
            if track is None or window_vecs is None or window_vecs.shape[0] == 0:
                continue
            tracks.append((track, window_vecs))
        print(f"# db={args.db}  tracks={len(tracks)}  anchors={len(MOOD_ANCHORS)}")

        for query, exemplars in PROBES.items():
            qvec = query_vector_contrast(query, embed_texts)
            anchor_sims = np.asarray(anchor_vecs @ qvec, dtype=np.float64).ravel()
            twin_idx = int(anchor_sims.argmax())
            twins = ", ".join(
                f"{MOOD_ANCHORS[i][:34]!r}={anchor_sims[i]:.3f}" for i in np.argsort(-anchor_sims)[:3].tolist()
            )
            print(f"\n== {query!r}")
            print(f"  closest anchors to query: {twins}")
            variants = make_variants(twin_idx)
            scores: dict[str, list[float]] = {name: [] for name, _ in variants}
            for _track, window_vecs in tracks:
                q = np.asarray(window_vecs @ qvec, dtype=np.float64).ravel()
                a = np.asarray(window_vecs @ anchor_vecs.T, dtype=np.float64)
                for name, fn in variants:
                    scores[name].append(fn(q, a))
            total = len(tracks)
            print(f"  {total} ranked tracks")
            # Decision-mode readout: share of library passing p >= threshold.
            for name, _ in variants:
                if not name.startswith("V3 sig mean"):
                    continue
                arr = np.asarray(scores[name], dtype=np.float64)
                rates = [
                    float(np.asarray(arr >= t).mean()) * PERCENTILE_SCALE  # pyrefly: ignore[unsupported-operation]
                    for t in (0.5, 0.9, 0.98)
                ]
                print(f"  [{name}] pass-rate p>=0.5/0.9/0.98: {rates[0]:.1f}% / {rates[1]:.1f}% / {rates[2]:.1f}%")
                for needle in exemplars:
                    hits = [scores[name][i] for i, (t, _w) in enumerate(tracks) if _matches(t, needle)]
                    if len(hits) > 0:
                        print(f"      {needle!r} p: " + ", ".join(f"{h:.3f}" for h in sorted(hits, reverse=True)[:3]))
                qs = np.percentile(arr, (50, 75, 90, 95)).tolist()
                print("      library p quantiles 50/75/90/95: " + " / ".join(f"{v:.3f}" for v in qs))
            for name, _ in variants:
                order: list[int] = [int(i) for i in np.argsort(-np.asarray(scores[name], dtype=np.float64)).tolist()]
                parts: list[str] = []
                for needle in exemplars:
                    hit = next((i for i in order if _matches(tracks[i][0], needle)), None)
                    if hit is None:
                        parts.append(f"{needle!r}: NOT FOUND")
                    else:
                        parts.append(f"{needle!r}: rank {hit + 1}/{total} ({scores[name][hit]:+.4f})")
                print(f"  [{name}]  " + " | ".join(parts))
                for i in order[:PREVIEW_COUNT]:
                    print(f"      {scores[name][i]:+.4f}  {tracks[i][0].title[:34]}")
            if TFS_NEEDLE in exemplars:
                tfs = next(w for t, w in tracks if _matches(t, TFS_NEEDLE))
                tfs_diagnostics(query_vec=qvec, anchor_vecs=anchor_vecs, windows=tfs)


if __name__ == "__main__":
    main()
