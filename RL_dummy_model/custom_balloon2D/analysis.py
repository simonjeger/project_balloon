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

def plot_reward():
    # extract data
    df = pd.read_csv('process' + str(yaml_p['process_nr']).zfill(5) + '/log_environment.csv', header=None)
    epi = np.array(df.iloc[:,0])
    rew_step = np.array(df.iloc[:,7])

    # plot reward
    N = int(len(rew_step)/10)
    cumsum = np.cumsum(np.insert(rew_step, 0, 0))
    mean_reward = (cumsum[N:] - cumsum[:-N]) / float(N)

    fig, ax1 = plt.subplots()
    ax1.set_title('max mean: ' + str(np.round(max(mean_reward),2)) + '   last mean: ' + str(np.round(mean_reward[-1],2)))
    ax1.set_xlabel('step')
    ax1.set_ylabel('reward')
    ax1.tick_params(axis='y')

    ax1.plot(mean_reward, color='firebrick')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax1.legend(['mean reward'], loc='upper left')
    plt.savefig('process' + str(yaml_p['process_nr']).zfill(5) + '/learning_curve.pdf')


def plot_path():
    # sort by episode
    df_env = pd.read_csv('process' + str(yaml_p['process_nr']).zfill(5) + '/log_environment.csv', header=None)
    df_ag = pd.read_csv('process' + str(yaml_p['process_nr']).zfill(5) + '/log_agent.csv', header=None)

    epi = np.array(df_env.iloc[:,0])
    size_x = np.array(df_env.iloc[:,1])
    size_z = np.array(df_env.iloc[:,2])
    pos_x = np.array(df_env.iloc[:,3])
    pos_z = np.array(df_env.iloc[:,4])
    tar_x = np.array(df_env.iloc[:,5])
    tar_z = np.array(df_env.iloc[:,6])
    eps = np.array(df_ag.iloc[:,0])

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

    step = 0
    # save pictre
    for n in range(len(list_pos_x)):
        step += len(list_pos_x[n])

        fig, axs = plt.subplots(2,1)

        # plot path
        axs[0].plot(list_pos_x[n], list_pos_z[n])
        axs[0].scatter(list_tar_x[n], list_tar_z[n])
        axs[0].set_xlim(0,list_size_x[n][0])
        axs[0].set_ylim(0,list_size_z[n][0])
        axs[0].set_aspect('equal')
        axs[0].set_title(str(int(n/len(list_pos_x)*100)) + ' %')

        # plot epsilon
        axs[1].plot(np.arange(0,step,1), eps[0:step])
        axs[1].set_ylim(0,step)
        axs[1].set_ylim(0,1)

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
