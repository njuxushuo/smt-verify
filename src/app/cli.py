import argparse
from pathlib import Path

from ..verification.verifier import verify_file
from .bootstrap import default_registry
from .reporting import demo_path, write_report


def positive(value):
    try:
        number = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("expected positive integer") from e
    if number < 1:
        raise argparse.ArgumentTypeError("expected positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify original-shape index/value stage relations using Z3."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("verify")
    single.add_argument("input", type=Path)
    single.add_argument("--report", type=Path)
    batch = commands.add_parser("batch")
    batch.add_argument("input_root", type=Path, nargs="?", default=Path("input"))
    batch.add_argument("--write-reports", action="store_true")
    for command in (single, batch):
        command.add_argument("--timeout-ms", type=positive, default=30000)
    args = parser.parse_args(argv)
    if args.command == "verify":
        paths = (args.input,)
    else:
        if not args.input_root.is_dir():
            parser.error("input_root must be an existing directory")
        paths = tuple(sorted(args.input_root.rglob("*.json")))
        if not paths:
            parser.error("no input JSON files found")
        if args.write_reports:
            destinations = [demo_path(p) for p in paths]
            if len(set(destinations)) != len(destinations):
                parser.error(
                    "multiple JSON files share a demo.txt destination; use verify --report for each"
                )
    registry = default_registry()
    failed = False
    for path in paths:
        result = verify_file(path, registry, timeout_ms=args.timeout_ms)
        print(f"{path}: {result.status.value} [{result.scope}]", flush=True)
        if result.reason:
            print("  " + result.reason, flush=True)
        report = (
            args.report
            if args.command == "verify"
            else demo_path(path)
            if args.write_reports
            else None
        )
        if report is not None:
            try:
                if any(
                    report.resolve() == source.resolve()
                    or (report.exists() and source.exists() and report.samefile(source))
                    for source in paths
                ):
                    parser.error(
                        "report must not overwrite input JSON (including hard-link aliases)"
                    )
                write_report(result, report)
                print(f"  Report: {report}", flush=True)
            except OSError as e:
                parser.error(str(e))
        failed |= result.status.value != "PROVED"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
