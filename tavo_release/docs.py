from __future__ import annotations

import shlex
from pathlib import Path

from .matrix import DATASET_METHODS


KNOWN_COMMANDS = {
    "check",
    "collect",
    "command",
    "da-command",
    "da-config",
    "docs-audit",
    "download",
    "matrix",
    "officehome-config",
    "pathway-audit",
    "plan",
    "plan-audit",
    "route-audit",
    "route-inventory",
    "repro-smoke",
    "search",
    "select",
    "selection-route",
    "smoke",
    "split",
    "tavo-command",
}


def bash_blocks(path: str | Path) -> list[str]:
    lines = []
    in_block = False
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if stripped in {"```bash", "```sh"}:
            in_block = True
            continue
        if stripped == "```":
            in_block = False
            continue
        if in_block and stripped:
            lines.append(stripped)
    return lines


def normalized_command(line: str) -> list[str]:
    parts = shlex.split(line)
    while parts and "=" in parts[0] and not parts[0].startswith("--"):
        parts = parts[1:]
    return parts


def flag_value(parts: list[str], flag: str) -> str | None:
    if flag not in parts:
        return None
    index = parts.index(flag) + 1
    if index >= len(parts):
        return None
    return parts[index]


def readme_audit(path: str | Path = "README.md") -> dict:
    errors = []
    commands = [normalized_command(line) for line in bash_blocks(path)]
    cli_commands = [cmd for cmd in commands if len(cmd) >= 4 and cmd[:3] == ["python", "-m", "tavo_release.cli"]]
    if not cli_commands:
        errors.append({"error": "no_cli_commands"})
    for cmd in cli_commands:
        name = cmd[3]
        if name not in KNOWN_COMMANDS:
            errors.append({"command": " ".join(cmd), "error": "unknown_cli_command"})
        if name == "search" and cmd.count("--score") != 8:
            errors.append({"command": " ".join(cmd), "error": "search_not_8d"})
        if name == "select":
            if cmd.count("--score") != 8:
                errors.append({"command": " ".join(cmd), "error": "select_not_8d"})
            if "--weight" not in cmd:
                errors.append({"command": " ".join(cmd), "error": "select_missing_weight"})
            else:
                index = cmd.index("--weight") + 1
                weights = []
                while index < len(cmd) and not cmd[index].startswith("--"):
                    weights.append(cmd[index])
                    index += 1
                if len(weights) != 8:
                    errors.append({"command": " ".join(cmd), "error": "select_weight_not_8d"})
        if name == "da-config":
            dataset = flag_value(cmd, "--dataset")
            method = flag_value(cmd, "--method")
            if dataset in DATASET_METHODS and method not in DATASET_METHODS[dataset]["domain_adaptation"]:
                errors.append({"command": " ".join(cmd), "error": "da_unknown_method"})
            target = {"mamamia": "NACT", "brats": "C5", "officehome": "Art"}.get(dataset or "")
            if target and flag_value(cmd, "--target") != target:
                errors.append({"command": " ".join(cmd), "error": "da_missing_target"})
            if dataset == "mamamia" and flag_value(cmd, "--nnunet-dataset-id") is None:
                errors.append({"command": " ".join(cmd), "error": "mamamia_da_missing_dataset_id"})
    return {"ok": not errors, "commands": len(commands), "cli_commands": len(cli_commands), "errors": errors}
