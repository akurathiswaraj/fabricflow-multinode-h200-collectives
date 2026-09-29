"""Build an H200-only public release from a reviewed allowlist of files."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import zipfile

from fabricflow.h200_v07_evidence import audit_public_directory


DOCS = (
    "architecture.md", "benchmark_methodology.md", "h200_v07_validation_plan.md",
    "operations_runbook.md", "portfolio_story.md", "results_h200_v07.md",
)


def build(project_root: Path, stage: Path, archive_path: Path) -> tuple[int, int]:
    project_root = project_root.resolve()
    stage = stage.resolve()
    archive_path = archive_path.resolve()
    outputs = (project_root / "outputs").resolve()
    if not stage.is_relative_to(outputs) or not archive_path.is_relative_to(outputs):
        raise ValueError("release targets must be inside project outputs")
    if stage.exists() or archive_path.exists():
        raise FileExistsError("release stage or archive already exists; do not overwrite")
    audit_public_directory(project_root / "outputs/h200_v07_public_evidence", project_root)

    paths = [
        project_root / name for name in
        (".gitignore", "LICENSE", "README.md", "pyproject.toml", "modal_fabricflow.py")
    ]
    paths += [project_root / ".github/workflows/ci.yml"]
    paths += [project_root / "docs" / name for name in DOCS]
    paths += sorted((project_root / "configs").glob("*.json"))
    paths += [project_root / "configs/README.md"]
    paths += sorted((project_root / "fabricflow").glob("*.py"))
    paths += sorted((project_root / "tests").glob("*.py"))
    paths += [
        project_root / "scripts/prepare_h200_v07_public.py",
        project_root / "scripts/build_h200_v07_release.py",
    ]
    paths += sorted((project_root / "outputs/h200_v07_public_evidence").iterdir())
    for source in paths:
        if not source.is_file():
            raise FileNotFoundError(source)
        relative = source.relative_to(project_root)
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    release_files = sorted(path for path in stage.rglob("*") if path.is_file())
    checksums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(stage).as_posix()}"
        for path in release_files
    ]
    (stage / "RELEASE_SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    release_files = sorted(path for path in stage.rglob("*") if path.is_file())

    for path in release_files:
        relative = path.relative_to(stage).as_posix().lower()
        if relative.endswith(".log") or "hardware_evidence/" in relative or "a10" in relative:
            raise ValueError(f"private or out-of-scope file in release: {relative}")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in release_files:
            archive.write(path, arcname=path.relative_to(stage).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("release ZIP CRC failure")
    return len(release_files), archive_path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    count, size = build(args.project_root, args.stage, args.archive)
    print(f"Release files: {count}")
    print(f"ZIP bytes: {size}")
    print(f"Release archive: {args.archive.resolve()}")


if __name__ == "__main__":
    main()
