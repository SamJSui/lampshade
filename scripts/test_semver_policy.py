import unittest

from semver_policy import compatibility_policy


class SemverPolicyTests(unittest.TestCase):
    def test_012_initial_release_allows_the_documented_breaking_cleanup(self):
        self.assertEqual(compatibility_policy("0.12.0"), ("0.11.0", "major"))
        self.assertEqual(compatibility_policy("0.12.0-rc.1"), ("0.11.0", "major"))

    def test_every_later_pre_1x_release_uses_the_strict_012_baseline(self):
        self.assertEqual(compatibility_policy("0.12.1"), ("0.12.0", "patch"))
        self.assertEqual(compatibility_policy("0.13.0"), ("0.12.0", "patch"))
        self.assertEqual(compatibility_policy("0.99.0-rc.1"), ("0.12.0", "patch"))

    def test_1x_requires_an_explicit_policy_update(self):
        with self.assertRaisesRegex(ValueError, "explicit semver baseline policy update"):
            compatibility_policy("1.0.0")


if __name__ == "__main__":
    unittest.main()
