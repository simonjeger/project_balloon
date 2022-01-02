from analysis import plot_reward, plot_path, plot_2d_path, plot_3d_path, make_2d_gif, tuning, dist_hist, wind_est, write_overview, disp_overview, clear

import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import shutil

# analyse
tuning()
#tuning('logger_test_tuning_vicon')
#plot_2d_path()
#plot_path()
#dist_hist()
#make_2d_gif()
#dist_hist(['process09462/logger_test/', 'process09450/logger_test/', 'process99991/logger_test/'])
#plot_3d_path()
#write_overview()
#disp_overview()
#wind_est()
