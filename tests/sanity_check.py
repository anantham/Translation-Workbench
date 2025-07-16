import sys
import os
import importlib
import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils

print("utils module path:", utils.__file__)
pprint.pprint({name: getattr(utils, name) for name in utils.__all__})