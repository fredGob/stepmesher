"""Assemble stepmesher Abaqus parts into one model using *INCLUDE."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def _safe_name(value: str, used: set[str], prefix: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", value).strip("_") or prefix
    name = name[:80]
    candidate = name
    index = 2
    while candidate.upper() in used:
        suffix = f"_{index}"
        candidate = f"{name[:80 - len(suffix)]}{suffix}"
        index += 1
    used.add(candidate.upper())
    return candidate


def _part_name(path: Path) -> str:
    """Read the existing PART name; fall back to the input filename."""
    pattern = re.compile(r"^\*PART\s*,.*?\bNAME\s*=\s*([^,\s]+)", re.IGNORECASE)
    with path.open(encoding="ascii", errors="replace") as stream:
        for line in stream:
            match = pattern.match(line.strip())
            if match:
                return match.group(1)
    raise ValueError(f"aucun *PART trouve dans {path.name}")


def _copy_with_part_name(source: Path, destination: Path, part_name: str) -> None:
    rewritten = _part_text(source, part_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rewritten, encoding="ascii")


def _part_text(source: Path, part_name: str) -> str:
    """Return one PART block, optionally replacing its Abaqus part name."""
    text = source.read_text(encoding="ascii", errors="replace")
    start = re.search(r"^\*PART\s*,", text, re.IGNORECASE | re.MULTILINE)
    if start is None:
        raise ValueError(f"aucun *PART trouve dans {source.name}")
    text = text[start.start():]
    pattern = re.compile(r"^(\*PART\s*,.*?\bNAME\s*=\s*)[^,\s]+", re.IGNORECASE | re.MULTILINE)
    rewritten, count = pattern.subn(rf"\g<1>{part_name}", text, count=1)
    if count != 1:
        raise ValueError(f"impossible de renommer *PART dans {source.name}")
    return rewritten


def collect_inputs(folder: Path, recursive: bool, include_failed: bool, output: Path) -> list[Path]:
    pattern = "**/*.inp" if recursive else "*.inp"
    paths = sorted(folder.glob(pattern), key=lambda path: str(path).lower())
    return [
        path for path in paths
        if path.resolve() != output.resolve()
        and not path.name.lower().startswith("master")
        and "_includes" not in path.parts
        and (include_failed or ".failed." not in path.name.lower())
    ]


def write_master(output: Path, inputs: list[Path], flatten: bool) -> None:
    used_parts: set[str] = set()
    used_instances: set[str] = set()
    lines = [
        "*HEADING",
        "stepmesher assembly generated from included part input files",
        "** Each source file contributes one *PART definition.",
    ]
    instances: list[tuple[str, str]] = []
    for path in inputs:
        source_part = _part_name(path)
        part = source_part
        include_path = path
        if source_part.upper() != re.sub(r"[^A-Za-z0-9_]", "_", path.stem).upper() \
                or source_part.upper() in used_parts:
            part = _safe_name(path.stem, used_parts, "PART")
            if not flatten:
                include_path = output.parent / f"{output.stem}_includes" / path.name
                _copy_with_part_name(path, include_path, part)
        else:
            used_parts.add(part.upper())
        instance = _safe_name(f"{part}_INSTANCE", used_instances, "INSTANCE")
        if flatten:
            lines.extend([_part_text(path, part), ""])
        else:
            relative = include_path.relative_to(output.parent).as_posix()
            lines.extend([f"*INCLUDE, INPUT={relative}", ""])
        instances.append((instance, part))

    lines.extend(["*ASSEMBLY, NAME=ASSEMBLY"])
    for instance, part in instances:
        lines.extend([f"*INSTANCE, NAME={instance}, PART={part}", "*END INSTANCE"])
    lines.extend(["*END ASSEMBLY", ""])
    output.write_text("\n".join(lines), encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble des .inp Abaqus stepmesher avec *INCLUDE et *ASSEMBLY."
    )
    parser.add_argument("folder", type=Path, help="dossier contenant les .inp")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="fichier maitre (defaut: <dossier>/master.inp)")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="chercher aussi dans les sous-dossiers")
    parser.add_argument("--include-failed", action="store_true",
                        help="inclure les fichiers dont le nom contient .FAILED.")
    parser.add_argument("--flatten", action="store_true",
                        help="ecrire les blocs *PART dans le master, sans *INCLUDE")
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        parser.error(f"dossier introuvable : {folder}")
    output = (args.output or folder / "master.inp").resolve()
    if output.parent != folder and not args.recursive:
        parser.error("le fichier maitre doit etre dans le dossier source sans --recursive")
    inputs = collect_inputs(folder, args.recursive, args.include_failed, output)
    if not inputs:
        parser.error("aucun fichier .inp trouve")
    write_master(output, inputs, args.flatten)
    print(f"{len(inputs)} pieces incluses dans {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
