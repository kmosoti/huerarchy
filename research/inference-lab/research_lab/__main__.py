import argparse
import json
import os
import subprocess
import sys

# Set before importing NumPy/BLAS, including subprocesses. Total runnable CPU work is bounded.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1" if "active" in sys.argv else "4"

from .common import DATA, RESULTS, ROOT, profile_environment, write_json


def main():
    parser = argparse.ArgumentParser(description="Local research pilot: exact facts, closure, retrieval, and context deltas")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("profile")
    generate = sub.add_parser("generate")
    generate.add_argument("--events", type=int, default=100_000)
    generate.add_argument("--nodes", type=int, default=5_000)
    generate.add_argument("--queries", type=int, default=10_000)
    generate.add_argument("--seed", type=int, default=42)
    bench = sub.add_parser("bench")
    bench.add_argument("--variant", choices=("all", "exact", "index-build", "closure", "communication", "retrieval", "instruction-set", "vector-kernels"), default="all")
    bench.add_argument("--repeats", type=int, default=5)
    sub.add_parser("audit")
    sub.add_parser("report")
    from .learning.cli import add_parser as add_learning_parser
    add_learning_parser(sub)
    from .discovery.cli import add_parser as add_discovery_parser
    add_discovery_parser(sub)
    from .active.cli import add_parser as add_active_parser
    add_active_parser(sub)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.command == "active":
        from .active.cli import main as active_main
        active_main(args)
    elif args.command == "discovery":
        from .discovery.cli import main as discovery_main
        discovery_main(args)
    elif args.command == "learning":
        from .learning.cli import main as learning_main
        learning_main(args)
    elif args.command == "profile":
        profile = profile_environment()
        write_json(RESULTS / "environment.json", profile)
        print(json.dumps({k: v for k, v in profile.items() if k not in ("cpu_flags",)}, indent=2))
    elif args.command == "generate":
        from .data import generate as run_generate
        print(json.dumps(run_generate(events=args.events, nodes=args.nodes, queries=args.queries, seed=args.seed), indent=2))
    elif args.command == "audit":
        from .audit import write_audit
        write_audit()
    elif args.command == "report":
        from .report import render
        print(render())
    elif args.command == "bench":
        from .runner import exact_benchmark, snapshot_closure, communication_benchmark, retrieval_benchmark, compare_builds
        from .artifacts import record_suites
        from .indexing import index_build_benchmark
        if args.repeats < 1:
            parser.error("repeats must be positive")
        if args.variant in ("all", "exact"):
            exact_benchmark(args.repeats)
        if args.variant in ("all", "index-build"):
            index_build_benchmark(args.repeats)
        if args.variant in ("all", "closure"):
            with (RESULTS / "closure.jsonl").open("w") as stream:
                subprocess.run([str(ROOT / "target/release/bench"), "bench", str(args.repeats)], check=True, stdout=stream)
            snapshot_closure(args.repeats)
        if args.variant in ("all", "communication"):
            communication_benchmark(args.repeats)
        if args.variant in ("all", "retrieval"):
            retrieval_benchmark(args.repeats)
        if args.variant in ("all", "instruction-set"):
            compare_builds(args.repeats)
        if args.variant in ("all", "vector-kernels"):
            from .vector_kernels import benchmark as vector_benchmark
            vector_benchmark(args.repeats)
        record_suites(("exact", "index-build", "closure", "communication", "retrieval", "instruction-set", "vector-kernels") if args.variant == "all" else (args.variant,))


if __name__ == "__main__":
    main()
