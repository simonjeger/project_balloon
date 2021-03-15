import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import shutil
import imageio
import os

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

def plot_reward(history):
    # plot reward
    episode_reward = history.history['episode_reward']
    N = int(len(episode_reward)/10)
    cumsum = np.cumsum(np.insert(episode_reward, 0, 0))
    mean_reward = (cumsum[N:] - cumsum[:-N]) / float(N)

    episode_steps = history.history['nb_episode_steps']
    cumsum = np.cumsum(np.insert(episode_steps, 0, 0))
    mean_steps = (cumsum[N:] - cumsum[:-N]) / float(N)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('reward')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('episode steps')  # we already handled the x-label with ax1
    ax2.tick_params(axis='y')

    ax2.plot(episode_steps, color='lightblue', alpha=0.1)
    ax1.plot(episode_reward, color='orange', alpha=0.1)
    ax2.plot(mean_steps, color='dodgerblue')
    ax1.plot(mean_reward, color='firebrick')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax1.legend(['reward per episode', 'mean reward'], loc='upper left')
    ax2.legend(['steps per episode', 'mean steps'], loc='upper right')
    plt.savefig('process' + str(yaml_p['process_nr']).zfill(5) + '/learning_curve.pdf')


def plot_path():
    # sort by episode
    df = pd.read_csv('process' + str(yaml_p['process_nr']).zfill(5) + '/path.csv', header=None)

    epi = np.array(df.iloc[:,0])
    size_x = np.array(df.iloc[:,1])
    size_z = np.array(df.iloc[:,2])
    pos_x = np.array(df.iloc[:,3])
    pos_z = np.array(df.iloc[:,4])
    tar_x = np.array(df.iloc[:,5])
    tar_z = np.array(df.iloc[:,6])

    n = epi[-1]
    list_size_x = [[] for i in range(n+1)]
    list_size_z = [[] for i in range(n+1)]
    list_pos_x = [[] for i in range(n+1)]
    list_pos_z = [[] for i in range(n+1)]
    list_tar_x = [[] for i in range(n+1)]
    list_tar_z = [[] for i in range(n+1)]
    for i in range(len(epi)):
        list_size_x[epi[i]].append(size_x[i])
        list_size_z[epi[i]].append(size_z[i])
        list_pos_x[epi[i]].append(pos_x[i])
        list_pos_z[epi[i]].append(pos_z[i])
        list_tar_x[epi[i]].append(tar_x[i])
        list_tar_z[epi[i]].append(tar_z[i])

    # save pictre
    for n in range(len(list_pos_x)):

        fig, ax = plt.subplots()
        ax.plot(list_pos_x[n], list_pos_z[n])
        ax.scatter(list_tar_x[n], list_tar_z[n])
        ax.set_xlim(0,list_size_x[n][0])
        ax.set_ylim(0,list_size_z[n][0])
        ax.set_aspect('equal')
        ax.set_title(str(int(n/len(list_pos_x)*100)) + ' %')

        # Build folder structure if it doesn't exist yet
        path = 'process' + str(yaml_p['process_nr']).zfill(5) + '/temp'
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + '/gif_' + str(n).zfill(5) + '.png', dpi=50)
        plt.close()
        print('saving frames: ' + str(int(n/len(list_pos_x)*100)) + ' %')

    # Build GIF
    with imageio.get_writer('process' + str(yaml_p['process_nr']).zfill(5) + '/path.gif', mode='I') as writer:
        name_list = os.listdir(path)
        name_list.sort()
        n = 0
        for name in name_list:
            image = imageio.imread(path + '/' + name)
            writer.append_data(image)
            print('generating gif: ' + str(int(n/len(list_pos_x)*100)) + ' %')
            n += 1

    # Delete temp folder
    shutil.rmtree(path)
