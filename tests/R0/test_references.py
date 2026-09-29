from copy import deepcopy
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brass_ai.contracts import ContractError, ROOT, read_json
from brass_ai.references import LOCK_PATH, safe_relative, validate_lock, verify_blob, verify_rules_files, verify_sources


class ReferenceTests(unittest.TestCase):
    def tiny_lock(self):
        data = b"fixture license\n"
        lock = deepcopy(read_json(LOCK_PATH))
        entry = {"path": "LICENSE", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                 "git_blob_sha1": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()}
        for reference in lock["references"]:
            reference["files"] = [deepcopy(entry)]
            reference["notice_paths"] = []
        return lock, data

    def test_project_manifest_map_and_all_examples(self):
        spec = importlib.util.spec_from_file_location("verify_r0", ROOT / "scripts/verify-r0.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _, _, components, examples = module.verify_project()
        self.assertEqual((components, examples), (10, 12))

    def test_mutable_ref_duplicate_paths_and_missing_license_rejected(self):
        for mutation in ("branch", "duplicate", "license"):
            lock = deepcopy(read_json(LOCK_PATH))
            reference = lock["references"][0]
            if mutation == "branch":
                reference["commit"] = "main"
            elif mutation == "duplicate":
                reference["files"].append(deepcopy(reference["files"][0]))
            else:
                reference["license_path"] = "missing-license"
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                validate_lock(lock)

    def test_traversal_and_absolute_paths_rejected(self):
        for path in ("../outside", "a/../../outside", "/tmp/a", "C:/Windows/a", "a\\b", "a//b", "a/./b"):
            with self.subTest(path=path), self.assertRaises(ContractError):
                safe_relative(path)

    def test_exact_bytes_and_git_blob_identity(self):
        data = b"source\n"
        entry = {"path": "example.py", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                 "git_blob_sha1": hashlib.sha1(b"blob 7\0" + data).hexdigest()}
        verify_blob(data, entry)
        for changed in (b"SOURCE\n", b"source\r\n"):
            with self.subTest(changed=changed), self.assertRaises(ContractError):
                verify_blob(changed, entry)

    def test_missing_cache_does_not_fetch_implicitly(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ContractError):
            verify_sources(read_json(LOCK_PATH), cache=directory)

    def test_fetch_accepts_exact_bytes_and_then_works_offline(self):
        lock, data = self.tiny_lock()
        with tempfile.TemporaryDirectory() as directory:
            with patch("brass_ai.references.urllib.request.urlopen", side_effect=lambda *a, **k: io.BytesIO(data)) as fetch:
                self.assertEqual(verify_sources(lock, cache=directory, fetch=True), 2)
                self.assertEqual(fetch.call_count, 2)
            with patch("brass_ai.references.urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
                self.assertEqual(verify_sources(lock, cache=directory), 2)
            reference = lock["references"][0]
            path = Path(directory) / reference["id"] / reference["commit"] / "files/LICENSE"
            path.write_bytes(b"tampered")
            with patch("brass_ai.references.urllib.request.urlopen", side_effect=AssertionError("must not replace")):
                with self.assertRaises(ContractError):
                    verify_sources(lock, cache=directory, fetch=True)
            self.assertEqual(path.read_bytes(), b"tampered")

    def test_failed_download_is_not_published_to_cache(self):
        lock, _ = self.tiny_lock()
        with tempfile.TemporaryDirectory() as directory:
            with patch("brass_ai.references.urllib.request.urlopen", side_effect=lambda *a, **k: io.BytesIO(b"truncated")):
                with self.assertRaises(ContractError):
                    verify_sources(lock, cache=directory, fetch=True)
            self.assertEqual(list(Path(directory).rglob("LICENSE")), [])

    def test_rules_mismatch_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.dll"
            path.write_bytes(b"unchanged fixture")
            profile = {"files": [{"path": "fixture.dll", "sha256": "0" * 64}]}
            with self.assertRaises(ContractError):
                verify_rules_files(profile, directory)
            self.assertEqual(path.read_bytes(), b"unchanged fixture")
            profile["files"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(verify_rules_files(profile, directory), 1)


if __name__ == "__main__":
    unittest.main()
