"""Read-only planning and upstream-preservation checks for this adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

GIT_TIMEOUT_SECONDS = 30


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git(repo: Path, *args: str) -> bytes:
    env = os.environ.copy()
    env.update(
        GIT_OPTIONAL_LOCKS="0",
        GIT_NO_LAZY_FETCH="1",
        GIT_TERMINAL_PROMPT="0",
    )
    result = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repo), *args],
        capture_output=True,
        timeout=GIT_TIMEOUT_SECONDS,
        env=env,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} in {repo}: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    return result.stdout


def guard(root: Path) -> dict:
    session = read_json(root / ".agents/dsv41/session.json")
    baseline_path = (
        root / session.get("readonly_baseline", ".agents/dsv41/readonly-baseline.json")
    ).resolve()
    if not baseline_path.is_relative_to(root):
        raise ValueError("Read-only baseline must remain within this workspace")
    baseline = read_json(baseline_path)
    lock = read_json(root / ".agents/dsv41/sources.lock.json")
    for name, saved in baseline["repositories"].items():
        repo = Path(saved["path"])
        actual = {
            "path": saved["path"],
            "head": git(repo, "rev-parse", "HEAD").decode().strip(),
            "status": git(
                repo, "status", "--porcelain=v1", "--untracked-files=all"
            ).decode("utf-8"),
            "tracked_diff_sha256": hashlib.sha256(
                git(repo, "diff", "--no-ext-diff", "--binary", "HEAD")
            ).hexdigest(),
        }
        if actual != saved:
            raise ValueError(f"Upstream workspace changed since snapshot: {name}")
        git(repo, "cat-file", "-e", lock["repositories"][name]["commit"] + "^{commit}")
    return {"upstreams": list(baseline["repositories"]), "unchanged": True}


def markdown_links(root: Path, path: Path, content: str) -> None:
    if content.count("~~~") % 2 or content.count(chr(96) * 3) % 2:
        raise ValueError(f"Unclosed Markdown fence: {path.relative_to(root)}")
    for target in re.findall(r"(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)", content):
        target = target.strip().strip("<>")
        if urlsplit(target).scheme or target.startswith("#"):
            continue
        local = unquote(target.split("#", 1)[0])
        if not local:
            continue
        resolved = (path.parent / local).resolve()
        if not resolved.is_relative_to(root) or not resolved.exists():
            raise ValueError(
                f"Invalid local link in {path.relative_to(root)}: {target}"
            )


def check_plan(root: Path) -> dict:
    manifest = read_json(root / ".agents/dsv41/plan-manifest.json")
    session = read_json(root / ".agents/dsv41/session.json")
    lock = read_json(root / ".agents/dsv41/sources.lock.json")
    declared = set(manifest["files"])
    for name in declared:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or escaping manifest path: {name}")
        content = path.read_text(encoding="utf-8")
        if any(line != line.rstrip() for line in content.splitlines()):
            raise ValueError(f"Trailing whitespace: {name}")
        if path.suffix == ".md":
            markdown_links(root, path, content)
        if path.suffix == ".json":
            read_json(path)

    base = lock["repositories"]["afd_plugin"]["commit"]
    git(root, "merge-base", "--is-ancestor", base, "HEAD")
    for source in lock["sources"]:
        repo = Path(lock["repositories"][source["repo"]]["path"])
        git(repo, "cat-file", "-e", source["commit"] + ":" + source["path"])

    task_count = 0
    for change in manifest["changes"]:
        folder = root / "openspec/changes" / change["id"]
        for required in ("proposal.md", "design.md", "tasks.md", ".openspec.yaml"):
            if not (folder / required).is_file():
                raise ValueError(f"Missing change artifact: {change['id']}/{required}")
        spec = (folder / "specs" / change["capability"] / "spec.md").read_text(
            encoding="utf-8"
        )
        if not spec.startswith("## ADDED Requirements"):
            raise ValueError(f"Missing delta heading: {change['id']}")
        requirements = spec.split("### Requirement: ")[1:]
        if not requirements:
            raise ValueError(f"No requirements: {change['id']}")
        for requirement in requirements:
            if not re.search(r"\b(SHALL|MUST)\b", requirement):
                raise ValueError(f"No normative requirement: {change['id']}")
            scenarios = requirement.split("#### Scenario: ")[1:]
            if not scenarios or any(
                "**WHEN**" not in scenario or "**THEN**" not in scenario
                for scenario in scenarios
            ):
                raise ValueError(f"Missing WHEN/THEN scenario: {change['id']}")
        tasks = (folder / "tasks.md").read_text(encoding="utf-8")
        checkboxes = re.findall(r"^- \[([ xX])\] \d+\.\d+ ", tasks, re.MULTILINE)
        if not checkboxes:
            raise ValueError(f"No actionable tasks: {change['id']}")
        if session["phase"] == "planning" and any(c != " " for c in checkboxes):
            raise ValueError(
                f"Implementation task marked complete in planning: {change['id']}"
            )
        task_count += len(checkboxes)

    for skill in manifest["skills"]:
        entry = root / ".agents/skills" / skill / "SKILL.md"
        content = entry.read_text(encoding="utf-8")
        if not content.startswith("---\n") or "\n---\n" not in content[4:]:
            raise ValueError(f"Invalid skill frontmatter: {skill}")
        if f"name: {skill}\n" not in content or "\ndescription: " not in content:
            raise ValueError(f"Missing skill metadata: {skill}")

    expected_tasks = f"openspec/changes/{session['active_change']}/tasks.md"
    if session["tasks"] != expected_tasks or expected_tasks not in declared:
        raise ValueError("Session task pointer does not match the active change")

    if session["phase"] == "planning":
        tracked = git(root, "diff", "--name-only", base).decode("utf-8").splitlines()
        untracked = (
            git(root, "ls-files", "--others", "--exclude-standard")
            .decode("utf-8")
            .splitlines()
        )
        unexpected = (set(tracked) | set(untracked)) - declared
        if unexpected:
            raise ValueError("Out-of-plan changes: " + ", ".join(sorted(unexpected)))
    preserved = guard(root)
    return {
        "files": len(declared),
        "changes": len(manifest["changes"]),
        "tasks": task_count,
        "source_objects": len(lock["sources"]),
        "skills": len(manifest["skills"]),
        "phase": session["phase"],
        "upstreams_unchanged": preserved["unchanged"],
        "validation_scope": "local plan checks; not official CLI or NPU qualification",
    }


def status(root: Path) -> dict:
    session = read_json(root / ".agents/dsv41/session.json")
    tasks = (root / session["tasks"]).read_text(encoding="utf-8")
    next_task = next(
        (line[6:] for line in tasks.splitlines() if line.startswith("- [ ] ")),
        "No unchecked task in active change",
    )
    return {
        "phase": session["phase"],
        "active_change": session["active_change"],
        "tasks": session["tasks"],
        "next_task": next_task,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "guard", "status"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        if args.command == "check":
            result = check_plan(root)
        elif args.command == "guard":
            result = guard(root)
        else:
            result = status(root)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True))
        return 1
    print(json.dumps({"ok": True, **result}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
