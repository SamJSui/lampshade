#!/usr/bin/env python3
"""Compile a standalone consumer against Cargo's extracted package artifact."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    args = parser.parse_args()

    package_dir = args.package.resolve()
    manifest = package_dir / "Cargo.toml"
    if not manifest.is_file() or not (package_dir / ".cargo_vcs_info.json").is_file():
        raise SystemExit(f"{package_dir} is not an extracted cargo package")

    manifest_text = manifest.read_text(encoding="utf-8")
    package_section = re.search(r"(?ms)^\[package\]\s+(.*?)(?=^\[|\Z)", manifest_text)
    name = (
        re.search(r'^name\s*=\s*"([^"]+)"', package_section.group(1), re.MULTILINE)
        if package_section
        else None
    )
    if name is None or name.group(1) != "lampshade":
        raise SystemExit("package artifact does not contain lampshade")

    fixture = Path(__file__).with_name("src") / "main.rs"
    with tempfile.TemporaryDirectory(prefix="lampshade-package-consumer-") as temporary:
        consumer = Path(temporary)
        (consumer / "src").mkdir()
        shutil.copy2(fixture, consumer / "src" / "main.rs")
        dependency_path = json.dumps(str(package_dir))
        (consumer / "Cargo.toml").write_text(
            "\n".join(
                (
                    "[package]",
                    'name = "lampshade-package-consumer"',
                    'version = "0.0.0"',
                    'edition = "2024"',
                    "publish = false",
                    "",
                    "[dependencies]",
                    f"lampshade = {{ path = {dependency_path} }}",
                    "",
                )
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                os.environ.get("CARGO", "cargo"),
                "check",
                "--manifest-path",
                str(consumer / "Cargo.toml"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
