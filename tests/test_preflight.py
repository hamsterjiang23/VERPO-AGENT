"""Ensure version drift and contaminated dependencies fail before execution."""

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/preflight.py"
spec = importlib.util.spec_from_file_location("preflight", SCRIPT)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def run(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dep = self.root / "third_party/PGR-Probe"
        self.dep.mkdir(parents=True)
        for path in (self.root, self.dep):
            run(path, "init", "-b", "main")
            run(path, "config", "user.name", "Test")
            run(path, "config", "user.email", "test@example.invalid")
        (self.dep / "kernel.py").write_text("# fixture\n")
        run(self.dep, "add", "kernel.py")
        run(self.dep, "commit", "-m", "fixture")
        commit = run(self.dep, "rev-parse", "HEAD")
        self.lock = {
            "path": "third_party/PGR-Probe", "url": "https://example.invalid/dependency.git",
            "commit": commit, "source_sha256": {"kernel.py": preflight.sha256(self.dep / "kernel.py")},
        }
        self.save_lock()
        run(self.root, "config", "-f", ".gitmodules", "submodule.third_party/PGR-Probe.url", self.lock["url"])
        run(self.root, "update-index", "--add", "--cacheinfo", f"160000,{commit},third_party/PGR-Probe")
        (self.root / "configs").mkdir()
        design = {
            "status": "design_only_not_runnable", "training_enabled": False,
            "step_gate_enabled": False, "benefit_predictor_enabled": False,
            "fec_projection_enabled": False, "replay": {"unit": "full_trajectory"},
        }
        self.design_path = self.root / "configs/observation_replay_design.json"
        self.design_path.write_text(json.dumps(design))
        research = self.root / "docs/research"
        research.mkdir(parents=True)
        (research / "provenance.json").write_text(json.dumps({"upstream_commit": commit, "files": []}))

    def save_lock(self):
        (self.root / "upstream.lock.json").write_text(json.dumps(self.lock))

    def test_clean_fixture_passes_without_claiming_training_ready(self):
        result = preflight.inspect_workspace(self.root)
        self.assertEqual(result["status"], "passed", result)
        self.assertFalse(result["training_ready"])

    def test_wrong_pin_fails(self):
        self.lock["commit"] = "0" * 40
        self.save_lock()
        self.assertEqual(preflight.inspect_workspace(self.root)["status"], "failed")

    def test_dirty_dependency_fails(self):
        (self.dep / "kernel.py").write_text("# changed\n")
        self.assertEqual(preflight.inspect_workspace(self.root)["status"], "failed")

    def test_staged_gitlink_drift_fails(self):
        run(self.dep, "commit", "--allow-empty", "-m", "next")
        commit = run(self.dep, "rev-parse", "HEAD")
        run(self.root, "update-index", "--cacheinfo", f"160000,{commit},third_party/PGR-Probe")
        run(self.dep, "checkout", self.lock["commit"])
        self.assertEqual(preflight.inspect_workspace(self.root)["status"], "failed")

    def test_forbidden_gate_fails(self):
        design = json.loads(self.design_path.read_text())
        design["step_gate_enabled"] = True
        self.design_path.write_text(json.dumps(design))
        self.assertEqual(preflight.inspect_workspace(self.root)["status"], "failed")

    def test_manifest_path_escape_fails(self):
        self.lock["source_sha256"] = {"../../outside": hashlib.sha256(b"").hexdigest()}
        self.save_lock()
        self.assertEqual(preflight.inspect_workspace(self.root)["status"], "failed")

    def test_missing_dependency_fails(self):
        self.lock["path"] = "missing"
        self.save_lock()
        self.assertEqual(preflight.inspect_workspace(self.root)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
