"""`sp` CLI entry point."""


from pathlib import Path

import typed_argparse as tap

from local_smart_playlist.commands import describe_cmd, index_cmd, playlist_cmd, status_cmd


class IndexArgs(tap.TypedArgs):
    root: Path = tap.arg(positional=True, metavar="ROOT", help="Library root directory")
    rescan: bool = tap.arg(help="Re-index tracks already present in the DB")
    progress: bool = tap.arg(help="Show a progress bar")
    db: Path | None = tap.arg(type=Path, help="Index DB path (default: .sp-index/index.db, relative to the working directory)")

    def run(self) -> None:
        index_cmd.IndexArgs.run(root=Path(self.root).expanduser(), db=self.db, rescan=self.rescan, progress=self.progress)


class PlayArgs(tap.TypedArgs):
    query: str = tap.arg(positional=True, metavar="QUERY", help="Mood word or phrase")
    n: int | None = tap.arg("-n", default=None, help="Number of tracks (default: all indexed)")
    out: str | None = tap.arg("-o", default=None, help="Output m3u8 path, or '-' for stdout")
    llm: bool = tap.arg(help="Expand the query with an LLM (OpenAI-compatible endpoint)")
    seed_track: str | None = tap.arg(default=None, metavar="PATH", help="Use a track's vector as the query instead of text")
    alpha: float = tap.arg(default=0.7, help="Peak-window weight in scoring (0 = mean only, 1 = peak only)")
    min_score: float = tap.arg(default=0.45, help="Keep only tracks scoring at least this (default: all are ranked, kept ones filtered)")
    dry: bool = tap.arg(help="Print the result without writing a playlist")
    db: Path | None = tap.arg(type=Path, help="Index DB path (default: .sp-index/index.db, relative to the working directory)")

    def run(self) -> None:
        playlist_cmd.PlayArgs.run(
            query=self.query,
            db=self.db,
            n=self.n,
            out=self.out,
            llm=self.llm,
            seed_track=self.seed_track,
            alpha=self.alpha,
            min_score=self.min_score,
            dry=self.dry,
        )


class DescribeArgs(tap.TypedArgs):
    path: str = tap.arg(positional=True, metavar="PATH", help="Track path (absolute, cwd-relative, or indexed rel path)")
    n: int = tap.arg("-n", default=describe_cmd.DEFAULT_TOP_N, help="Number of captions to show (default: 10)")
    db: Path | None = tap.arg(type=Path, help="Index DB path (default: .sp-index/index.db, relative to the working directory)")

    def run(self) -> None:
        describe_cmd.DescribeArgs.run(path=self.path, db=self.db, n=self.n)


class StatusArgs(tap.TypedArgs):
    db: Path | None = tap.arg(type=Path, help="Index DB path (default: .sp-index/index.db, relative to the working directory)")

    def run(self) -> None:
        status_cmd.StatusArgs.run(db=self.db)


def main() -> None:
    parser = tap.Parser(
        tap.SubParserGroup(
            tap.SubParser("index", IndexArgs, help="Index a music library"),
            tap.SubParser("play", PlayArgs, help="Build a mood playlist"),
            tap.SubParser("describe", DescribeArgs, help="Describe an indexed track with caption-vocab neighbors"),
            tap.SubParser("status", StatusArgs, help="Show index coverage"),
            description="Mood-based smart playlists over a local music library.",
        ),
        prog="sp",
    )
    args = parser.parse_args()
    if isinstance(args, IndexArgs | PlayArgs | DescribeArgs | StatusArgs):
        args.run()


if __name__ == "__main__":
    main()
