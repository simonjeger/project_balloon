from analysis import plot_reward, plot_path, plot_2d_path, plot_3d_path, tuning, dist_hist, write_overview, disp_overview, clear

import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import shutil

# analyse
#tuning('logger_test_wind_vicon2')
#tuning()
#plot_2d_path()
#dist_hist()
dist_hist(['process09462/logger_test/', 'process09450/logger_test/', 'process99991/logger_test/'])
#plot_3d_path()
#write_overview()
#disp_overview()
