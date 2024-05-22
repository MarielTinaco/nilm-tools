import os
from pathlib import Path

dirname = os.path.dirname(__file__)

UKDALE_H5_PATH = os.path.join(dirname, '../../data/UKDALE/UKDALE.h5')
SRC_DIR = Path(os.path.join(dirname, "../../src"))
PROFILES_DIR = Path(os.path.join(dirname, "../../profiles"))