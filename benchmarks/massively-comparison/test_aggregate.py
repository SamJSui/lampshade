import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aggregate import aggregate_runs, comparisons, load_failures, median


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

    def test_load_failures_preserves_case_and_error_output(self) -> None:
        with TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            error_path = root / "failure.log"
            error_path.write_text("pipeline creation failed\nlimit: 29\n", encoding="utf-8")
            index_path = root / "failures.tsv"
            index_path.write_text(
                f"massively\texclusive_scan\t1000000\t2\t{error_path}\n",
                encoding="utf-8",
            )

            self.assertEqual(
                load_failures(index_path),
                [
                    {
                        "implementation": "massively",
                        "workload": "exclusive_scan",
                        "items": 1_000_000,
                        "process_index": 2,
                        "error": "pipeline creation failed\nlimit: 29",
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
