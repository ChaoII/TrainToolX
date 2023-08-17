__version__ = '2.1.0'

import os.path

from traintoolx.utils.env import get_environ_info, init_parallel_env

init_parallel_env()

from . import cv
from . import det
from . import tools

env_info = get_environ_info()
datasets = cv.datasets
transforms = cv.transforms
log_level = 2
load_model = cv.models.load_model


