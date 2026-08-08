import unittest

from aggregate import aggregate_runs, comparisons, median


def run(implementation: str, process: int, duration: float) -> dict:
    return {
        "implementation": implementation,
        "config": {
            "workload": "exclusive_scan",
            "items": 1_000_000,
            "process_index": process,
        },
        "median_ms": duration,
        "memory": {"model": "test"},
    }


class AggregateTests(unittest.TestCase):
    def test_median_handles_odd_and_even_values(self) -> None:
        self.assertEqual(median([3.0, 1.0, 2.0]), 2.0)
        self.assertEqual(median([4.0, 1.0, 3.0, 2.0]), 2.5)

    def test_aggregate_and_speedup_use_process_medians(self) -> None:
        aggregates = aggregate_runs(
            [
                run("wgpu-primitives", 1, 2.0),
                run("massively", 1, 6.0),
                run("wgpu-primitives", 2, 1.0),
                run("massively", 2, 4.0),
                run("wgpu-primitives", 3, 3.0),
                run("massively", 3, 5.0),
            ]
        )
        comparison = comparisons(aggregates)[0]
        self.assertEqual(comparison["wgpu_primitives_ms"], 2.0)
        self.assertEqual(comparison["massively_ms"], 5.0)
        self.assertEqual(comparison["wgpu_primitives_speedup"], 2.5)


if __name__ == "__main__":
    unittest.main()
