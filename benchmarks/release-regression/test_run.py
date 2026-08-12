import importlib.util
import tomllib
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).with_name("run.py")
    spec = importlib.util.spec_from_file_location("release_regression", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


release_regression = load_module()


class ReleaseRegressionTests(unittest.TestCase):
    def test_published_runner_targets_previous_lampshade_release(self):
        manifest = Path(__file__).with_name("published-runner") / "Cargo.toml"
        contents = manifest.read_text(encoding="utf-8")

        self.assertEqual(release_regression.BASELINE_VERSION, "0.10.1")
        self.assertIn('name = "lampshade-release-baseline-runner"', contents)
        self.assertIn('lampshade = "=0.10.1"', contents)
        self.assertNotIn('package = "wgpu-primitives"', contents)

    def test_runtime_stack_comes_from_the_locked_dependency_graph(self):
        candidate_manifest = (
            Path(__file__).parents[1]
            / "massively-comparison"
            / "lampshade-runner"
            / "Cargo.toml"
        )
        baseline_manifest = Path(__file__).with_name("published-runner") / "Cargo.toml"

        self.assertEqual(
            release_regression.resolved_wgpu_stack(candidate_manifest),
            "wgpu 29.0.4; wgpu-core 29.0.4; wgpu-hal 29.0.4; wgpu-types 29.0.4",
        )
        self.assertEqual(
            release_regression.resolved_wgpu_stack(baseline_manifest),
            "wgpu 29.0.4; wgpu-core 29.0.4; wgpu-hal 29.0.4; wgpu-types 29.0.4",
        )

    def test_candidate_version_does_not_require_a_root_lockfile(self):
        manifest = Path(__file__).parents[2] / "Cargo.toml"

        self.assertEqual(release_regression.package_version(manifest), "0.11.0")

    def test_path_consumers_lock_the_current_checkout_version(self):
        root = Path(__file__).parents[2]
        locks = [
            root / "validation" / "particle-app" / "Cargo.lock",
            root / "benchmarks" / "massively-comparison" / "lampshade-runner" / "Cargo.lock",
            root / "benchmarks" / "wgpu-sort-comparison" / "lampshade-runner" / "Cargo.lock",
        ]

        for lock in locks:
            packages = tomllib.loads(lock.read_text(encoding="utf-8"))["package"]
            versions = [package["version"] for package in packages if package["name"] == "lampshade"]
            self.assertEqual(versions, ["0.11.0"], lock)

    def test_aggregates_process_medians_and_applies_threshold(self):
        adapter = {"name": "GPU", "vendor": 1, "device": 2, "device_type": "discrete_gpu", "backend": "vulkan"}
        runs = [
            {"source": "published", "result": {"adapter": adapter, "config": {"workload": "exclusive_scan", "items": 10}, "median_ms": 10.0}},
            {"source": "published", "result": {"adapter": adapter, "config": {"workload": "exclusive_scan", "items": 10}, "median_ms": 12.0}},
            {"source": "checkout", "result": {"adapter": adapter, "config": {"workload": "exclusive_scan", "items": 10}, "median_ms": 11.0}},
            {"source": "checkout", "result": {"adapter": adapter, "config": {"workload": "exclusive_scan", "items": 10}, "median_ms": 11.2}},
        ]
        aggregates = release_regression.aggregate_runs(runs)
        comparisons = release_regression.compare_aggregates(aggregates, 2.0)

        self.assertEqual(len(comparisons), 1)
        self.assertAlmostEqual(comparisons[0]["published_ms"], 11.0)
        self.assertAlmostEqual(comparisons[0]["checkout_ms"], 11.1)
        self.assertTrue(comparisons[0]["passed"])

    def test_marks_regressions_above_the_budget(self):
        adapter = {"name": "GPU", "vendor": 1, "device": 2, "device_type": "discrete_gpu", "backend": "vulkan"}
        aggregates = [
            {"source": "published", "workload": "reduce_sum", "items": 10, "median_of_process_medians_ms": 1.0, "adapter": adapter, "adapter_consistent": True},
            {"source": "checkout", "workload": "reduce_sum", "items": 10, "median_of_process_medians_ms": 1.03, "adapter": adapter, "adapter_consistent": True},
        ]

        comparison = release_regression.compare_aggregates(aggregates, 2.0)[0]

        self.assertAlmostEqual(comparison["change_percent"], 3.0)
        self.assertFalse(comparison["passed"])

    def test_accepts_the_exact_regression_boundary(self):
        adapter = {"name": "GPU", "vendor": 1, "device": 2, "device_type": "discrete_gpu", "backend": "vulkan"}
        aggregates = [
            {"source": "published", "workload": "reduce_sum", "items": 10, "median_of_process_medians_ms": 1.0, "adapter": adapter, "adapter_consistent": True},
            {"source": "checkout", "workload": "reduce_sum", "items": 10, "median_of_process_medians_ms": 1.02, "adapter": adapter, "adapter_consistent": True},
        ]

        comparison = release_regression.compare_aggregates(aggregates, 2.0)[0]

        self.assertAlmostEqual(comparison["change_percent"], 2.0)
        self.assertTrue(comparison["passed"])

    def test_rejects_different_adapters(self):
        published = {"name": "GPU A", "vendor": 1, "device": 2, "device_type": "discrete_gpu", "backend": "vulkan"}
        checkout = {**published, "name": "GPU B"}
        aggregates = [
            {"source": "published", "workload": "reduce_sum", "items": 10, "median_of_process_medians_ms": 1.0, "adapter": published, "adapter_consistent": True},
            {"source": "checkout", "workload": "reduce_sum", "items": 10, "median_of_process_medians_ms": 0.5, "adapter": checkout, "adapter_consistent": True},
        ]

        comparison = release_regression.compare_aggregates(aggregates, 2.0)[0]

        self.assertFalse(comparison["adapter_match"])
        self.assertFalse(comparison["passed"])

    def test_quick_gate_ignores_timing_but_requires_the_same_adapter(self):
        slow_same_adapter = {"adapter_match": True, "passed": False}
        different_adapter = {"adapter_match": False, "passed": False}

        self.assertTrue(release_regression.passes_gate([], [slow_same_adapter], 1, True))
        self.assertFalse(release_regression.passes_gate([], [different_adapter], 1, True))

    def test_characterization_gate_ignores_timing_but_requires_complete_runs(self):
        slow_same_adapter = {"adapter_match": True, "passed": False}

        self.assertTrue(release_regression.passes_gate([], [slow_same_adapter], 1, True))
        self.assertFalse(
            release_regression.passes_gate(
                [{"error": "runner failed"}], [slow_same_adapter], 1, True
            )
        )


if __name__ == "__main__":
    unittest.main()
