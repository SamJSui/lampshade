#!/usr/bin/env python3
"""Select the semver baseline for Lampshade's pre-1.0 stabilization line."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def package_version(manifest: Path) -> str:
    text = manifest.read_text(encoding="utf-8")
    package = re.search(r"(?ms)^\[package\]\s+(.*?)(?=^\[|\Z)", text)
    version = re.search(r'^version\s*=\s*"([^"]+)"', package.group(1), re.MULTILINE) if package else None
    if version is None:
        raise ValueError(f"missing package version in {manifest}")
    return version.group(1)


def compatibility_policy(candidate: str) -> tuple[str, str]:
    matched = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", candidate)
    if matched is None:
        raise ValueError(f"unsupported package version: {candidate}")
    major, minor, patch = (int(part) for part in matched.groups())

    if major == 0 and minor == 12 and patch == 0:
        # cargo-semver-checks classifies module removal and adding
        # #[non_exhaustive] as a major change, even for a 0.x minor bump.
        return "0.11.0", "major"
    if major == 0 and minor == 12:
        return "0.12.0", "patch"
    if major == 0 and minor == 13 and patch == 0:
        # wgpu types cross Lampshade's public API, so moving from wgpu 29 to
        # wgpu 30 is an intentional pre-1.0 source-compatibility break.
        return "0.12.1", "major"
    if major == 0 and minor == 13:
        return "0.13.0", "patch"
    raise ValueError(
        f"version {candidate} requires an explicit semver baseline policy update"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("Cargo.toml"))
    args = parser.parse_args()
    candidate = package_version(args.manifest)
    baseline, release_type = compatibility_policy(candidate)
    print(f"candidate={candidate}")
    print(f"baseline={baseline}")
    print(f"release_type={release_type}")


if __name__ == "__main__":
    main()
