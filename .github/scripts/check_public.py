"""Conservative public-content scan of source, history and distributions; no secret output."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path

# Exact synthetic negative-test sentinels; never exempt an entire test directory.
SYNTHETIC = (
    b"/Users/" + b"synthetic-private/run",
    b"https://private.example.invalid/opaque-project",
    b"https://private.example.invalid",
)
PATTERNS = {
    "personal-home": rb"/(?:Users|home)/[A-Za-z0-9_.-]+/",
    "private-operations": (
        rb"(?:slack\.com/archives|linear\.app/|"
        rb"shapleyai/"
        rb"code(?:/|\b))"
    ),
    "private-key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "credential": (
        rb"(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}|"
        rb"gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})"
    ),
}


def inspect(name: str, content: bytes) -> list[dict[str, str]]:
    if name == "tests/test_reporting.py":
        # These exact values are assertions/inputs proving export sanitization.
        for sentinel in SYNTHETIC:
            content = content.replace(sentinel, b"SYNTHETIC_REDACTION_FIXTURE")
    if name == "tests/test_dashboard_models.py":
        # Historical negative-test input, replaced with a portable label in 7df2976.
        sentinel = re.escape(b"/Users/" + b"private/PRIVATE_LABEL")
        content = re.sub(
            rb"(?<![\w./-])" + sentinel + rb"(?![\w./-])",
            b"SYNTHETIC_REDACTION_FIXTURE",
            content,
        )
    return [{"file": name, "rule": rule} for rule, pattern in PATTERNS.items() if re.search(pattern, content)]


def tracked_content(path: Path) -> bytes:
    """Inspect Git symlink text without following targets outside the checkout."""
    return os.fsencode(os.readlink(path)) if path.is_symlink() else path.read_bytes()


def worktree_files():
    """Include new files and omit deleted paths without following symlinks."""
    names = (
        subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"])
        .decode()
        .split("\0")
    )
    return sorted({name for name in names if name and os.path.lexists(name)})


def publication_findings() -> list[dict[str, str]]:
    """Validate the initial public export, never the private development repository."""
    findings = []
    if subprocess.check_output(["git", "rev-list", "--count", "HEAD"]).strip() != b"1":
        findings.append({"file": "git", "rule": "publication-history"})
    refs = subprocess.check_output(["git", "for-each-ref", "--format=%(refname)"]).splitlines()
    if len(refs) != 1 or not refs[0].startswith(b"refs/heads/"):
        findings.append({"file": "git", "rule": "publication-refs"})
    if subprocess.check_output(["git", "remote"]).strip():
        findings.append({"file": "git", "rule": "publication-remotes"})
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"]).strip():
        findings.append({"file": "git", "rule": "publication-dirty"})
    metadata = subprocess.check_output(["git", "show", "-s", "--format=fuller", "HEAD"])
    findings.extend(inspect("initial-commit-metadata", metadata))
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--history", action="store_true", help="also scan private Git blob history")
    scope.add_argument(
        "--publication",
        action="store_true",
        help="verify a clean one-commit export without remotes, plus wheel and sdist",
    )
    args = parser.parse_args()
    files = worktree_files()
    findings = publication_findings() if args.publication else []
    if args.publication and (
        len(list(Path("dist").glob("*.whl"))) != 1 or len(list(Path("dist").glob("*.tar.gz"))) != 1
    ):
        findings.append({"file": "dist", "rule": "publication-packages"})
    for name in filter(None, files):
        findings.extend(inspect(name, tracked_content(Path(name))))
    if args.history:
        # Walk unique blobs, not commit messages or repeated snapshots.
        objects = subprocess.check_output(["git", "rev-list", "--objects", "--all"]).decode()
        for line in objects.splitlines():
            oid, _, name = line.partition(" ")
            if not name:
                continue
            kind = subprocess.check_output(["git", "cat-file", "-t", oid]).strip()
            if kind == b"blob":
                findings.extend(inspect(name, subprocess.check_output(["git", "cat-file", "blob", oid])))
    archives = []
    for archive in sorted(Path("dist").glob("*")):
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as handle:
                archives.extend((n, handle.read(n)) for n in handle.namelist() if not n.endswith("/"))
        elif archive.name.endswith(".tar.gz"):
            with tarfile.open(archive) as handle:
                for member in handle.getmembers():
                    if member.issym() or member.islnk():
                        archives.append((member.name, member.linkname.encode()))
                    elif member.isfile():
                        stream = handle.extractfile(member)
                        assert stream is not None
                        archives.append((member.name.split("/", 1)[-1], stream.read()))
    for name, content in archives:
        findings.extend(inspect(name, content))
    print(
        json.dumps(
            {
                "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
                "tracked_files": len(list(filter(None, files))),
                "archive_files": len(archives),
                "findings": findings,
            }
        )
    )
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
