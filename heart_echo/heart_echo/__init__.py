import os
import sys
sys.path.append(os.path.abspath("/cluster/home/elucas/thesis/"))
from config import regensburg_wrapper_path
sys.path.append(os.path.abspath(regensburg_wrapper_path))
from heart_echo import CLI
from heart_echo import Processing
from heart_echo import Helpers
from heart_echo import pytorch
from heart_echo import numpy