import gzip
import json
from pathlib import Path
import sqlite3
import zlib

from ..learning.types import catalog
from . import beliefs, concepts, learner, policies, worlds
from .types import DEFAULT


def add_parser(sub):
    parser = sub.add_parser("active", help="Active probabilistic learning and concept transfer")
    commands = parser.add_subparsers(dest="active_command", required=True)
    commands.add_parser("config")
    commands.add_parser("components")
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--smoke", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--run", type=Path, required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--run", type=Path, required=True)
    replay.add_argument("--id", required=True)


def replay(root, run_id):
    from .study import verify
    from .experiment import learner_for, next_query
    from ..discovery.programs import Program
    manifest = verify(root)
    for block in manifest["blocks"]:
        directory = Path(root) / block["path"]
        connection = sqlite3.connect(f"file:{directory / 'memory.sqlite'}?mode=ro", uri=True)
        try:
            row = connection.execute("SELECT state FROM checkpoints WHERE run_id=?", (run_id,)).fetchone()
        finally:
            connection.close()
        if row is None:
            continue
        state = json.loads(zlib.decompress(row[0]))
        with gzip.open(directory / "worlds.json.gz", "rt") as stream:
            task = worlds.OracleTask.from_dict(json.load(stream)[state["task"]])
        definitions = {k: Program.from_dict(v).clauses[0] for k, v in state["library"].items()}
        obj, policy = learner_for(task, state["config"], state["variant"], 0, definitions, state["transfer"], False, state["learner"])
        obj.restore(state["learner"])
        if policy:
            policy.restore(state["policy"])
        index, reason = next_query(obj, policy)
        program, mass = obj.best()
        return {"run_id": run_id, "next_question": index, "selection": reason, "program": program.text(),
                "program_mass": mass, "teacher_answer_acquired": False, "library_instances": len(definitions)}
    raise ValueError("unknown run id")


def main(args):
    if args.active_command == "config":
        print(json.dumps(DEFAULT, indent=2))
    elif args.active_command == "components":
        print(json.dumps({k: v for k, v in catalog().items() if k.startswith("active-")}, indent=2))
    elif args.active_command == "report":
        from .report import render
        print(render(args.run))
    elif args.active_command == "replay":
        print(json.dumps(replay(args.run, args.id), indent=2))
    else:
        from .study import run
        print(run(json.loads(args.config.read_text()) if args.config else None, args.output, args.smoke))
