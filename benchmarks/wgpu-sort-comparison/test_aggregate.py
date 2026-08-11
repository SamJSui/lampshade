import unittest

from aggregate import aggregate_runs, median


def run(implementation: str, process: int, duration: float) -> dict:
    return {
        "implementation": implementation,
        "config": {
            "mode": "resident",
            "workload": "bounded16",
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

    def test_aggregates_process_medians_by_implementation(self) -> None:
        aggregates = aggregate_runs(
            [
                run("lampshade", 1, 3.0),
                run("wgpu_sort", 1, 5.0),
                run("lampshade", 2, 1.0),
                run("wgpu_sort", 2, 7.0),
                run("lampshade", 3, 2.0),
                run("wgpu_sort", 3, 6.0),
            ]
        )
        by_name = {row["implementation"]: row for row in aggregates}
        self.assertEqual(by_name["lampshade"]["median_of_process_medians_ms"], 2.0)
        self.assertEqual(by_name["wgpu_sort"]["median_of_process_medians_ms"], 6.0)
        self.assertEqual(
            by_name["lampshade"]["throughput_pairs_per_second"],
            500_000_000.0,
        )


if __name__ == "__main__":
    unittest.main()
