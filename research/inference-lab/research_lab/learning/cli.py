import json
from pathlib import Path

from . import components, evaluation, worlds
from .types import catalog


def add_parser(subparsers):
    parser = subparsers.add_parser("learning", help="Pluggable online learning experiment")
    sub = parser.add_subparsers(dest="learning_command", required=True)
    sub.add_parser("components", help="List registered components")
    sub.add_parser("config", help="Print the complete default configuration")
    run = sub.add_parser("run", help="Run a controlled teaching experiment")
    run.add_argument("--config", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--smoke", action="store_true", help="Small end-to-end verification fixture")
    report = sub.add_parser("report", help="Verify and render a completed experiment")
    report.add_argument("--run", type=Path, required=True)


def main(args):
    if args.learning_command == "components":
        print(json.dumps(catalog(), indent=2))
    elif args.learning_command == "config":
        print(json.dumps(worlds.DEFAULT_CONFIG, indent=2))
    elif args.learning_command == "report":
        from .report import render
        print(render(args.run))
    else:
        from .experiment import run
        config = json.loads(args.config.read_text()) if args.config else {}
        if args.smoke:
            config = {**config, "stable_worlds": 1, "changing_worlds": 1,
                      "steps": 48, "changes": [16, 32], "noise_rates": [0, 0.2], "seeds": [0]}
        print(run(config, args.output))
