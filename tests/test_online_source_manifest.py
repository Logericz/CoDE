"""CPU-only deployment identity checks; these do not establish GPU parity."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from online_source_manifest import METHOD_IDENTITY_FILES, METHOD_SOURCE_FILES
import analyze_dense_collection as analysis
import diagnose_probe_parity as parity
import run_online_dense_collection as collection
import run_online_development as development
import run_online_diagnostic as diagnostic
import run_online_long_context as long_context
import validate_online_boundaries as boundaries
import validate_online_capacity as capacity
import validate_online_continuation as continuation
import validate_online_gpu as gpu


class OnlineSourceManifestTests(unittest.TestCase):
    def test_explicit_nine_module_inventory_covers_the_deployed_package(self):
        names = ("__init__", "common", "vanilla", "dense", "fixed", "adaptive",
                 "logarithmic", "random_schedule", "backoff")
        self.assertEqual(METHOD_SOURCE_FILES, tuple(f"src/online_methods/{name}.py" for name in names))
        self.assertEqual(METHOD_IDENTITY_FILES, ("scripts/online_source_manifest.py", *METHOD_SOURCE_FILES))
        actual = {str(path.relative_to(ROOT)) for path in (ROOT / "src/online_methods").glob("*.py")}
        self.assertEqual(actual, set(METHOD_SOURCE_FILES))

    def test_all_direct_and_inherited_evidence_manifests_include_method_sources_once(self):
        inventories = {
            "single question": diagnostic.CODE_FILES,
            "initial GPU validation": gpu.SOURCE_FILES,
            "same KV parity critical": parity.CRITICAL_SOURCE_FILES,
            "continuation": continuation.SOURCE_FILES,
            "capacity critical": capacity.CRITICAL_SOURCES,
            "capacity": capacity.SOURCE_FILES,
            "natural boundaries": boundaries.SOURCE_FILES,
            "long context": long_context.SOURCE_FILES,
            "long context critical": long_context.CRITICAL,
            "development": development.SOURCE_FILES,
            "dense collection": collection.SOURCE_FILES,
            "dense analysis shared implementation": analysis.CORE_SOURCE_FILES,
            "dense analysis local replay": analysis.REPLAY_SOURCE_FILES,
        }
        for name, values in inventories.items():
            with self.subTest(inventory=name):
                self.assertEqual(len(values), len(set(values)))
                self.assertTrue(set(METHOD_IDENTITY_FILES) <= set(values))
                self.assertTrue(all((ROOT / filename).is_file() for filename in values))

    def test_each_missing_or_changed_method_identity_fails_the_prerequisite_gate(self):
        hashes = {name: capacity.sha256(ROOT / name) for name in capacity.CRITICAL_SOURCES}
        capacity.check_source_hashes(hashes)
        for filename in METHOD_IDENTITY_FILES:
            with self.subTest(filename=filename, mutation="missing"):
                missing = {name: value for name, value in hashes.items() if name != filename}
                with self.assertRaisesRegex(ValueError, "Prerequisite source differs"):
                    capacity.check_source_hashes(missing)
            with self.subTest(filename=filename, mutation="changed"):
                with self.assertRaisesRegex(ValueError, "Prerequisite source differs"):
                    capacity.check_source_hashes({**hashes, filename: "0" * 64})

    def test_legacy_core_only_gate_cannot_certify_the_refactored_sources(self):
        legacy = {name: capacity.sha256(ROOT / name) for name in
                  ("src/online_contract.py", "src/online_protocol.py", "src/torch_online_backend.py")}
        with self.assertRaisesRegex(ValueError, "online_source_manifest"):
            capacity.check_source_hashes(legacy)

    def test_change_to_current_method_source_is_detected_without_touching_real_files(self):
        hashes = {name: capacity.sha256(ROOT / name) for name in capacity.CRITICAL_SOURCES}
        original = capacity.sha256
        target = ROOT / "src/online_methods/adaptive.py"
        with patch.object(capacity, "sha256", side_effect=lambda path: "f" * 64 if Path(path) == target else original(path)):
            with self.assertRaisesRegex(ValueError, "online_methods/adaptive"):
                capacity.check_source_hashes(hashes)


if __name__ == "__main__":
    unittest.main()
