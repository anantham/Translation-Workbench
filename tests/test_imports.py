import unittest
import utils

class TestUtilsImports(unittest.TestCase):

    def test_all_imports_work(self):
        """Check that all functions in utils.__all__ can be imported."""
        for func_name in utils.__all__:
            with self.subTest(function=func_name):
                try:
                    # Attempt to get the attribute from the package
                    getattr(utils, func_name)
                except AttributeError:
                    self.fail(f"Failed to import '{func_name}' from utils package. Check __init__.py for a missing import or alias.")

if __name__ == '__main__':
    unittest.main()
