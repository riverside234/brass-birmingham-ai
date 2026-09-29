"""Verify pinned reference sources and the selected local rules binary pair."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re
import urllib.request

from .contracts import ContractError, ROOT, read_json

LOCK_PATH = ROOT / "configs/references/upstream.lock.json"
CACHE = ROOT / "artifacts/r0-references"


def _need(condition, message):
    if not condition:
        raise ContractError(message)


def safe_relative(value):
    _need(isinstance(value, str) and bool(value) and "\\" not in value and ":" not in value,
          "invalid relative path")
    path = PurePosixPath(value)
    _need(not path.is_absolute() and all(part not in ("", ".", "..") for part in value.split("/")),
          "unsafe relative path")
    _need(bool(re.fullmatch(r"[A-Za-z0-9_./-]+", value)), "unsupported path characters")
    return path


def confined(root, relative):
    root = Path(root).resolve()
    path = (root / safe_relative(relative)).resolve()
    _need(path.is_relative_to(root), "path escapes its declared root")
    return path


def validate_lock(lock):
    _need(lock["format"] == "brass-ai-reference-lock" and lock["format_version"] == 1,
          "unsupported reference lock")
    _need(lock["reuse_status"] == "reference_only_no_upstream_source_vendored",
          "unexpected source reuse status")
    expected = {"daiyu": "Yuiffy/brass-birmingham-daiyu-ai",
                "open-spiel": "google-deepmind/open_spiel"}
    _need(len(lock["references"]) == 2, "expected both selected references")
    _need({r["id"] for r in lock["references"]} == set(expected), "wrong reference IDs")
    for reference in lock["references"]:
        _need(reference["repository"] == expected[reference["id"]], "unexpected repository")
        for field in ("commit", "git_tree"):
            _need(bool(re.fullmatch(r"[a-f0-9]{40}", reference[field])), "expected immutable Git object ID")
        files = reference["files"]
        paths = [item["path"] for item in files]
        _need(files and len(paths) == len(set(paths)), "missing or duplicate source paths")
        for item in files:
            safe_relative(item["path"])
            _need(type(item["bytes"]) is int and 0 < item["bytes"] <= 5_000_000, "invalid source length")
            _need(bool(re.fullmatch(r"[a-f0-9]{64}", item["sha256"])), "invalid source SHA-256")
            _need(bool(re.fullmatch(r"[a-f0-9]{40}", item["git_blob_sha1"])), "invalid Git blob ID")
        _need(reference["license_path"] in paths, "license file is not pinned")
        _need(all(path in paths for path in reference["notice_paths"]), "notice is not pinned")
    return lock


def verify_blob(data, entry):
    _need(len(data) == entry["bytes"], f"source length mismatch: {entry['path']}")
    _need(hashlib.sha256(data).hexdigest() == entry["sha256"], f"source SHA-256 mismatch: {entry['path']}")
    git_blob = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
    _need(git_blob == entry["git_blob_sha1"], f"Git blob mismatch: {entry['path']}")


def verify_sources(lock, cache=CACHE, fetch=False):
    """Download only missing files. A modified cache fails rather than being replaced."""
    validate_lock(lock)
    checked = 0
    for reference in lock["references"]:
        for entry in reference["files"]:
            relative = f"{reference['id']}/{reference['commit']}/files/{entry['path']}"
            destination = confined(cache, relative)
            if destination.exists():
                data = destination.read_bytes()
            elif fetch:
                url = (f"https://raw.githubusercontent.com/{reference['repository']}/"
                       f"{reference['commit']}/{entry['path']}")
                request = urllib.request.Request(url, headers={"User-Agent": "brass-ai-r0-verifier"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = response.read(entry["bytes"] + 1)
                verify_blob(data, entry)
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Never replace an existing cache file, even after a concurrent fetch.
                try:
                    with destination.open("xb") as output:
                        output.write(data)
                except FileExistsError:
                    data = destination.read_bytes()
            else:
                raise ContractError(f"missing cached reference: {relative}; use --fetch-references")
            verify_blob(data, entry)
            checked += 1
    return checked


def verify_rules_files(profile, game_root):
    """Read bytes only. Never load an assembly, reinstall a patch or access saves."""
    checked = 0
    for entry in profile["files"]:
        path = confined(game_root, entry["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _need(digest == entry["sha256"], f"rules fingerprint mismatch: {entry['path']}")
        checked += 1
    return checked
