"""`sp` CLI entry point."""

from __future__ import annotations

from pathlib import Path

import typed_argparse as tap

from local_smart_playlist.commands import index_cmd, playlist_cmd, status_cmd


class IndexArgs(tap.TypedArgs):
    root: Path = tap.arg(positional=True, metavar="ROOT", help="Library root directory")
    rescan: bool = tap.arg(help="Re-index tracks already present in the DB")
    progress: bool = tap.arg(help="Show a progress bar")
    db: Path | None = tap.arg(type=Path, help="Index DB path (default: $XDG_DATA_HOME/sp/index.db)")

    def run(self) -> None:
        index_cmd.IndexArgs.run(root=Path(self.root).expanduser(), db=self.db, rescan=self.rescan, progress=self.progress)


class PlayArgs(tap.TypedArgs):
    query: str = tap.arg(positional=True, metavar="QUERY", help="Mood word or phrase")
    n: int = tap.arg("-n", default=30, help="Number of tracks")
    out: str | None = tap.arg("-o", default=None, help="Output m3u8 path, or '-' for stdout")
    llm: bool = tap.arg(help="Expand the query with an LLM (OpenAI-compatible endpoint)")
    seed_track: str | None = tap.arg(default=None, metavar="PATH", help="Use a track's vector as the query instead of text")
    alpha: float = tap.arg(default=0.7, help="Peak-window weight in scoring (0 = mean only, 1 = peak only)")
    dry: bool = tap.arg(help="Print the result without writing a playlist")
    db: Path | None = tap.arg(type=Path, help="Index DB path (default: $XDG_DATA_HOME/sp/index.db)")

    def run(self) -> None:
        playlist_cmd.PlayArgs.run(
            query=self.query,
            db=self.db,
            n=self.n,
            out=self.out,
            llm=self.llm,
            seed_track=self.seed_track,
            alpha=self.alpha,
            dry=self.dry,
        )


class StatusArgs(tap.TypedArgs):
    db: Path | None = tap.arg(type=Path, help="Index DB path (default: $XDG_DATA_HOME/sp/index.db)")

    def run(self) -> None:
        status_cmd.StatusArgs.run(db=self.db)


def main() -> None:
    parser = tap.Parser(
        tap.SubParserGroup(
            tap.SubParser("index", IndexArgs, help="Index a music library"),
            tap.SubParser("play", PlayArgs, help="Build a mood playlist"),
            tap.SubParser("status", StatusArgs, help="Show index coverage"),
            description="Mood-based smart playlists over a local music library.",
        ),
        prog="sp",
    )
    args = parser.parse_args()
    if isinstance(args, IndexArgs | PlayArgs | StatusArgs):
        args.run()


if __name__ == "__main__":
    main()
