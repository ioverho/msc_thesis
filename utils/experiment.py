import os
import re
import random
from datetime import datetime

import torch
import numpy as np

def find_version(experiment_version: str, checkpoint_dir: str, debug: bool = False):
    """[summary]

    Args:
        experiment_version (str): [description]
        checkpoint_dir (str): [description]
        debug (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """

    # Default version number
    version = 0

    if debug:
        version = 'debug'
    else:
        for subdir, dirs, files in os.walk(f"{checkpoint_dir}/{experiment_version}"):
            match = re.search(r".*version_([0-9]+)$", subdir)
            if match:
                match_version = int(match.group(1))
                if match_version > version:
                    version = match_version

        version = str(version + 1)

    full_version = experiment_version + "/version_" + str(version)

    return full_version, experiment_version, str(version)


def set_seed(seed):
    """[summary]

    Args:
        seed ([type]): [description]
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def set_deterministic():
    """[summary]
    """
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

class Timer:
    """[summary]
    """
    def __init__(self, silent = False):
        self.start = datetime.now()
        self.silent = silent
        if not self.silent:
            print(f"Started at {self.start}")

    def time(self):
        end = datetime.now()

        return end - self.start

    def end(self):
        end = datetime.now()

        if not self.silent:
            print(f"Ended at at {end}")
