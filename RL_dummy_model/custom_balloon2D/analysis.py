import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import shutil
import imageio
import os
import torch

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
    # import from log files
    df_env = pd.read_csv('process' + str(yaml_p['process_nr']).zfill(5) + '/log_environment.csv', names=['epi', 'size_x', 'size_z', 'pos_x', 'pos_z', 'tar_x', 'tar_z', 'rew_step', 'rew_epi'])
    df_ag = pd.read_csv('process' + str(yaml_p['process_nr']).zfill(5) + '/log_agent.csv', names=['eps'])

    # set up parameters to generate gif
    duration = yaml_p['duration']
    N = df_env.iloc[-1,0]+1
    fps = min(int(N/duration),20)

    n_f = duration*fps
    idx = np.linspace(0,N-N/n_f,n_f)
    idx = [int(i) for i in idx]

    step = 0
    for i in range(len(idx)-1):
        fig, axs = plt.subplots(2,1)

        idx_fra = np.arange(idx[i], idx[i+1],1)
        df_env_fra = df_env[df_env['epi'].isin(idx_fra)]
        df_ag_fra = df_ag[df_env['epi'].isin(idx_fra)]

        for j in idx_fra:
            df_env_loc = df_env_fra[df_env_fra['epi'].isin([j])]

            # plot path
            axs[0].plot(df_env_loc['pos_x'], df_env_loc['pos_z'], color='grey')
            axs[0].scatter(df_env_loc['tar_x'], df_env_loc['tar_z'], color='grey')
            axs[0].set_xlim(0,df_env_loc['size_x'].iloc[0])
            axs[0].set_ylim(0,df_env_loc['size_z'].iloc[0])

            step += len(df_env_loc['pos_x'])

        axs[0].set_aspect('equal')
        axs[0].set_title(str(int(i/n_f*100)) + ' %')

        # plot epsilon
        axs[1].plot(df_ag.iloc[0:step], color='grey')
        axs[1].set_xlabel('steps')
        axs[1].set_ylabel('epsilon')
        axs[1].set_xlim(0,step)
        axs[1].set_ylim(0,1)

        # Build folder structure if it doesn't exist yet
        path = 'process' + str(yaml_p['process_nr']).zfill(5) + '/temp'
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + '/gif_' + str(i).zfill(5) + '.png', dpi=50)
        plt.close()
        print('saving frames: ' + str(int(i/n_f*100)) + ' %')

    # Build GIF
    with imageio.get_writer('process' + str(yaml_p['process_nr']).zfill(5) + '/path.gif', mode='I', fps=fps) as writer:
        name_list = os.listdir(path)
        name_list.sort()
        n = 0
        for name in name_list:
            image = imageio.imread(path + '/' + name)
            writer.append_data(image)
            print('generating gif: ' + str(int(n/n_f*100)) + ' %')
            n += 1

    # Delete temp folder
    shutil.rmtree(path)

def plot_qmap():
    # import from log files
    df_env = pd.read_csv('process' + str(yaml_p['process_nr']).zfill(5) + '/log_environment.csv', names=['epi', 'size_x', 'size_z', 'pos_x', 'pos_z', 'tar_x', 'tar_z', 'rew_step', 'rew_epi'])

    name_list = os.listdir('process' + str(yaml_p['process_nr']).zfill(5) + '/log_qmap')
    name_list.sort()
    tensor_list = []
    for name in name_list:
        tensor_list.append(torch.load('process' + str(yaml_p['process_nr']).zfill(5) + '/log_qmap/' + name))

    # set up parameters to generate gif
    duration = yaml_p['duration']
    N = df_env.iloc[-1,0]+1
    fps = min(int(N/duration),20)

    n_f = duration*fps
    idx = np.linspace(0,N-N/n_f,n_f)
    idx = [int(i) for i in idx]

    step = 0
    for i in range(len(idx)-1):
        fig, axs = plt.subplots(4,1)

        idx_fra = np.arange(idx[i], idx[i+1],1)
        df_env_fra = df_env[df_env['epi'].isin(idx_fra)]

        for j in idx_fra:
            df_env_loc = df_env_fra[df_env_fra['epi'].isin([j])]
            step += len(df_env_loc['pos_x'])

        # plot qmap
        a0 = np.transpose(tensor_list[i][:,:,0])
        a1 = np.transpose(tensor_list[i][:,:,1])
        a2 = np.transpose(tensor_list[i][:,:,2])
        a3 = np.transpose(tensor_list[i][:,:,3])

        vmin = np.min(tensor_list[i])
        vmax = np.max(tensor_list[i])

        q = axs[0].imshow(a0, vmin=vmin, vmax=vmax)
        axs[1].imshow(a1, vmin=vmin, vmax=vmax)
        axs[2].imshow(a2, vmin=vmin, vmax=vmax)
        a = axs[3].imshow(a3, vmin=0, vmax=2)

        axs[0].set_title(str(int(i/n_f*100)) + ' %')
        fig.colorbar(q, ax=axs[0:3], orientation="vertical")
        fig.colorbar(a, ax=axs[3], orientation="vertical")

        # Build folder structure if it doesn't exist yet
        path = 'process' + str(yaml_p['process_nr']).zfill(5) + '/temp'
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + '/gif_' + str(i).zfill(5) + '.png', dpi=50)
        plt.close()
        print('saving frames: ' + str(int(i/n_f*100)) + ' %')

    # Build GIF
    with imageio.get_writer('process' + str(yaml_p['process_nr']).zfill(5) + '/qmap.gif', mode='I', fps=fps) as writer:
        name_list = os.listdir(path)
        name_list.sort()
        n = 0
        for name in name_list:
            image = imageio.imread(path + '/' + name)
            writer.append_data(image)
            print('generating gif: ' + str(int(n/n_f*100)) + ' %')
            n += 1

    # Delete temp folder
    shutil.rmtree(path)
