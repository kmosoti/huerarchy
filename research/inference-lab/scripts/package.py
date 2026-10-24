"""Package source, evidence and measured results; omit rebuildable large binaries."""
import sys
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_lab.artifacts import verify_suites
from research_lab.common import file_hash
from research_lab.learning.experiment import verify_manifest as verify_learning_manifest
from research_lab.discovery.experiment import verify as verify_discovery_manifest
from research_lab.active.study import verify as verify_active_manifest

verify_suites()
learning_results = ROOT / "results/learning"
if learning_results.exists():
    for directory in sorted(learning_results.iterdir()):
        if directory.is_dir():
            verify_learning_manifest(directory)
discovery_results = ROOT / "results/discovery"
if discovery_results.exists():
    for directory in sorted(discovery_results.iterdir()):
        if directory.is_dir():
            verify_discovery_manifest(directory)
active_results = ROOT / "results/active"
if active_results.exists():
    for directory in sorted(active_results.iterdir()):
        if directory.is_dir():
            verify_active_manifest(directory)
manifest = json.loads((ROOT / "results/manifest.json").read_text())
for category in ("files", "source_files", "configuration_files"):
    for name, metadata in manifest[category].items():
        expected = metadata["sha256"] if isinstance(metadata, dict) else metadata
        if file_hash(ROOT / name) != expected:
            raise RuntimeError(f"artifact/source changed since the report was rendered: {name}")
archive = ROOT.parent / "inference-lab-pilot.tar.gz"
temporary = archive.with_suffix(archive.suffix + ".tmp")
with tarfile.open(temporary, "w:gz", compresslevel=6) as stream:
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in {".venv", "target", "models", "__pycache__"} for part in relative.parts):
            continue
        if path.suffix in (".pyc", ".tmp"):
            continue
        if relative.parts[0] == "data" and (path.suffix == ".vec" or path.name.startswith(("embeddings-", "vectors-", "vector-scales"))):
            continue
        stream.add(path, arcname=str(Path(ROOT.name) / relative), recursive=False)
with tarfile.open(temporary) as stream:
    names = set(stream.getnames())
    for required in ("README.md", "REPORT.html", "results/summary.json", "src/vectors.rs", "results/vector-kernel-validation.json"):
        if f"{ROOT.name}/{required}" not in names:
            raise RuntimeError(f"incomplete package: {required}")
    if (learning_results / "default").exists():
        for required in ("research_lab/learning/types.py", "configs/learning/default.json", "results/learning/default/REPORT.html", "results/learning/default/memory.sqlite"):
            if f"{ROOT.name}/{required}" not in names:
                raise RuntimeError(f"incomplete learning package: {required}")
    if (discovery_results / "default").exists():
        for required in ("research_lab/discovery/programs.py", "configs/discovery/default.json", "results/discovery/default/REPORT.html", "results/discovery/default/memory.sqlite", "evidence/discovery-research.md"):
            if f"{ROOT.name}/{required}" not in names:
                raise RuntimeError(f"incomplete discovery package: {required}")
    if (active_results / "default").exists():
        for required in ("research_lab/active/learner.py", "configs/active/default.json", "results/active/default/REPORT.html",
                         "results/active/default/selection.json", "evidence/active-research.md"):
            if f"{ROOT.name}/{required}" not in names:
                raise RuntimeError(f"incomplete active package: {required}")
temporary.replace(archive)
archive.with_suffix(archive.suffix + ".sha256").write_text(f"{file_hash(archive)}  {archive.name}\n")
print(f"{archive} ({archive.stat().st_size:,} bytes)")
