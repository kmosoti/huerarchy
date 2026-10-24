import json
from pathlib import Path

from ..learning.types import catalog
from . import expansion, learner
from .worlds import DEFAULT


def add_parser(subparsers):
    parser = subparsers.add_parser("discovery", help="Discovery worlds and evidence-tested program expansion")
    sub = parser.add_subparsers(dest="discovery_command", required=True)
    sub.add_parser("components")
    sub.add_parser("config")
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--smoke", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("--run", type=Path, required=True)


def main(args):
    if args.discovery_command == "components":
        print(json.dumps({k: v for k, v in catalog().items() if k.startswith("discovery-")}, indent=2))
    elif args.discovery_command == "config":
        print(json.dumps(DEFAULT, indent=2))
    elif args.discovery_command == "report":
        from .report import render
        print(render(args.run))
    else:
        from .experiment import run
        config = json.loads(args.config.read_text()) if args.config else {}
        if args.smoke:
            config = {**config, "workers": 1, "curricula": 2, "steps": 72, "changes": [24, 48], "noise_rates": [0.1],
                      "seeds": [0], "scene_count": 12}
        print(run(config, args.output))
