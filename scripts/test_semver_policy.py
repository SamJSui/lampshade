import unittest

from semver_policy import compatibility_policy


class SemverPolicyTests(unittest.TestCase):
    def test_012_initial_release_allows_the_documented_breaking_cleanup(self):
        self.assertEqual(compatibility_policy("0.12.0"), ("0.11.0", "major"))
        self.assertEqual(compatibility_policy("0.12.0-rc.1"), ("0.11.0", "major"))

    def test_012_patch_releases_use_the_strict_012_baseline(self):
        self.assertEqual(compatibility_policy("0.12.1"), ("0.12.0", "patch"))

    def test_013_initial_release_allows_the_wgpu_30_break(self):
        self.assertEqual(compatibility_policy("0.13.0"), ("0.12.1", "major"))
        self.assertEqual(compatibility_policy("0.13.0-rc.1"), ("0.12.1", "major"))

    def test_013_patch_releases_use_the_strict_013_baseline(self):
        self.assertEqual(compatibility_policy("0.13.1"), ("0.13.0", "patch"))

    def test_future_minor_and_major_releases_require_an_explicit_policy_update(self):
        for version in ("0.14.0", "0.99.0-rc.1", "1.0.0"):
            with self.subTest(version=version):
                with self.assertRaisesRegex(
                    ValueError, "explicit semver baseline policy update"
                ):
                    compatibility_policy(version)


if __name__ == "__main__":
    unittest.main()
