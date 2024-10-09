import unittest

from utils import is_version_compatible


class TestVersionCompatibility(unittest.TestCase):

    def test_compatible_versions(self):
        compatible_versions = ["1.2.3", "1.3.0", "2.0.0"]

        # Test cases to assert compatibility
        self.assertTrue(is_version_compatible("1.2", compatible_versions))
        self.assertTrue(is_version_compatible("1.3", compatible_versions))

        # Test cases to assert incompatibility
        self.assertFalse(is_version_compatible("2.1", compatible_versions))

        # Matching exact version
        self.assertTrue(is_version_compatible("2.1", ["2.1"]))

    def test_pre_release_versions(self):
        self.assertTrue(is_version_compatible("0.10", ["0.10", "0.10pre", "0.10.1"]))

        # Compatible pre-release version
        self.assertTrue(is_version_compatible("0.10", ["0.10pre"]))
        self.assertTrue(is_version_compatible("0.10pre", ["0.10"]))


if __name__ == '__main__':
    unittest.main()
