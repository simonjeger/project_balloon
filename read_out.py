from analysis import plot_reward, plot_path, write_overview, disp_overview, clear

import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import shutil

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

# analyse
#plot_reward()
#plot_path()
disp_overview()
