#!/usr/bin/env python3
"""Explicitly regenerate research snapshots from the locked upstream commit."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote, unquote

from preflight import ROOT, git, sha256

NAMES = [
    "VERPO到Agent_VERPO_完整推导.md",
    "Agent_VERPO_讨论总结与Observation回放方案.md",
    "Agent_VERPO_联合控制研究与实现报告.md",
]


def main() -> None:
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    dep = (ROOT / lock["path"]).resolve()
    if git(dep, "rev-parse", "HEAD") != lock["commit"]:
        raise SystemExit("Dependency HEAD does not match upstream.lock.json")
    if git(dep, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Refusing to export a modified dependency")
    source_dir = dep / "agent opsd/agent_verpo_report"
    out = ROOT / "docs/research"
    out.mkdir(parents=True, exist_ok=True)
    mapped = {(source_dir / name).resolve(): name for name in NAMES}
    entries = []
    for name in NAMES:
        source = source_dir / name

        def rewrite(match):
            label, target = match.groups()
            if target.startswith(("https://", "http://", "#", "mailto:")):
                return match.group(0)
            link, mark, fragment = target.partition("#")
            resolved = (source_dir / unquote(link)).resolve()
            if not resolved.is_relative_to(dep) or not resolved.exists():
                raise ValueError("Unresolved source link: " + target)
            if resolved in mapped:
                new = quote(mapped[resolved])
            else:
                new = "https://github.com/hamsterjiang23/PGR-Probe/blob/" + lock["commit"] + "/" + quote(resolved.relative_to(dep).as_posix())
            return f"{label}({new}{mark}{fragment})"

        # Formula expressions such as [F(p)r](v) are not Markdown links.
        # Keep math and code regions byte-for-byte intact during link rewriting.
        chunks = re.split(r"(```[\s\S]*?```|\$\$[\s\S]*?\$\$|\$(?:\\.|[^$\n])*\$|`[^`\n]*`)", source.read_text())
        text = "".join(
            chunk if index % 2 else re.sub(r"(\[[^\]\n]+\])\(([^)\n]+)\)", rewrite, chunk)
            for index, chunk in enumerate(chunks)
        )
        title, rest = text.split("\n", 1)
        banner = (
            "\n> 上游研究快照，来源提交 `" + lock["commit"] + "`。"
            "历史候选中的 FEC、step gate 或收益预测器不覆盖[当前设计](../current_design.md)。"
            "原文与导出文件指纹见 [provenance.json](provenance.json)。\n"
        )
        destination = out / name
        exported = title + "\n" + banner + rest
        exported = "\n".join(
            line.rstrip() + "\\" if line.endswith("  ") else line.rstrip()
            for line in exported.splitlines()
        ).rstrip() + "\n"
        destination.write_text(exported)
        entries.append({
            "source": source.relative_to(dep).as_posix(),
            "source_sha256": sha256(source),
            "destination": destination.relative_to(ROOT).as_posix(),
            "export_sha256": sha256(destination),
        })
    provenance = {
        "upstream_commit": lock["commit"],
        "transformation": "rewrite relative links, prepend current-design notice, normalize Markdown hard breaks",
        "files": entries,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"exported_documents": len(entries), "upstream_commit": lock["commit"]}))


if __name__ == "__main__":
    main()
