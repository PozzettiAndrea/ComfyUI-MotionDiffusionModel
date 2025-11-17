# MDM package - add self to path for internal absolute imports
import os
import sys

_mdm_path = os.path.dirname(os.path.abspath(__file__))
if _mdm_path not in sys.path:
    sys.path.insert(0, _mdm_path)
