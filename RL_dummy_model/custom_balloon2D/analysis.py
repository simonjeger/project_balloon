import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
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
    rew_epi = np.array(df.iloc[:,8].dropna())

    # plot reward
    N_epi = yaml_p['phase']
    cumsum_epi = np.cumsum(np.insert(rew_epi, 0, 0))
    mean_reward_epi = (cumsum_epi[N_epi:] - cumsum_epi[:-N_epi]) / float(N_epi)

    N_step = min(int(len(rew_step)/10),10000)
    cumsum_step = np.cumsum(np.insert(rew_step, 0, 0))
    mean_reward_step = (cumsum_step[N_step:] - cumsum_step[:-N_step]) / float(N_step)

    fig, axs = plt.subplots(2,1)
    axs[0].plot(rew_epi, alpha=0.1)
    axs[0].plot(mean_reward_epi)
    axs[1].plot(rew_step, alpha=0.1)
    axs[1].plot(mean_reward_step)

    axs[0].set_title('max mean: ' + str(np.round(max(mean_reward_epi),5)))
    axs[0].set_xlabel('episode')
    axs[0].set_ylabel('reward')
    axs[0].tick_params(axis='y')

    axs[1].set_title('max mean: ' + str(np.round(max(mean_reward_step),5)))
    axs[1].set_xlabel('step')
    axs[1].set_ylabel('reward')
    axs[1].tick_params(axis='y')

    axs[0].legend(['reward', 'running mean over ' + str(N_epi) + ' episodes'], loc='upper left')
    axs[1].legend(['reward', 'running mean over ' + str(N_step) + ' steps'], loc='upper left')

    #fig.suptitle('learning rate over ' + str(len(rew_epi)) + ' episodes')
    fig.tight_layout()
    plt.savefig('process' + str(yaml_p['process_nr']).zfill(5) + '/learning_curve.pdf')


def plot_path():
    # import from log files
    df_env = pd.read_csv('process' + str(yaml_p['process_nr']).zfill(5) + '/log_environment.csv', names=['epi', 'size_x', 'size_z', 'pos_x', 'pos_z', 'tar_x', 'tar_z', 'rew_step', 'rew_epi'])
    df_ag = pd.read_csv('process' + str(yaml_p['process_nr']).zfill(5) + '/log_agent.csv', names=['eps'])

    # set up parameters to generate gif
    duration = yaml_p['duration']
    N = df_env.iloc[-1,0]+1
    fps = min(int(N/duration),yaml_p['fps'])

    n_f = duration*fps
    idx = np.linspace(0,N-N/n_f,n_f)
    idx = [int(i) for i in idx]

    vmin = min(df_env['rew_epi'])
    vmax = max(df_env['rew_epi'])
    vn = 100
    spectrum = np.linspace(vmin, vmax, vn)
    colors = pl.cm.jet(np.linspace(0,1,vn))

    step = 0
    for i in range(len(idx)-1):
        fig, axs = plt.subplots(2,1)

        idx_fra = np.arange(idx[i], idx[i+1],1)
        df_env_fra = df_env[df_env['epi'].isin(idx_fra)]
        df_ag_fra = df_ag[df_env['epi'].isin(idx_fra)]

        for j in idx_fra:
            df_env_loc = df_env_fra[df_env_fra['epi'].isin([j])]

            c = np.argmin(np.abs(spectrum - df_env_loc['rew_epi'].iloc[-1]))

            # plot path
            axs[0].plot(df_env_loc['pos_x'], df_env_loc['pos_z'], color=colors[c])
            axs[0].scatter(df_env_loc['tar_x'], df_env_loc['tar_z'], s=20, facecolors='none', edgecolors='grey')
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

    for i in range(len(name_list)):
        fig, axs = plt.subplots(4,2)

        # plot qmap
        q0 = np.flip(np.transpose(tensor_list[i][:,:,0]), axis=0)
        q1 = np.flip(np.transpose(tensor_list[i][:,:,1]), axis=0)
        q2 = np.flip(np.transpose(tensor_list[i][:,:,2]), axis=0)
        q3 = np.flip(np.transpose(tensor_list[i][:,:,3]), axis=0)

        a0 = np.flip(np.transpose(tensor_list[i][:,:,4]), axis=0)
        a1 = np.flip(np.transpose(tensor_list[i][:,:,5]), axis=0)
        a2 = np.flip(np.transpose(tensor_list[i][:,:,6]), axis=0)
        a3 = np.flip(np.transpose(tensor_list[i][:,:,7]), axis=0)

        vmin_q = np.min(tensor_list[i][:,:,0:3])
        vmax_q = np.max(tensor_list[i][:,:,0:3])

        vmin_a = np.min(tensor_list[i][:,:,4:7])
        vmax_a = np.max(tensor_list[i][:,:,4:7])

        img_02 = axs[0,0].imshow(q0, vmin=vmin_q, vmax=vmax_q)
        axs[1,0].imshow(q1, vmin=vmin_q, vmax=vmax_q)
        axs[2,0].imshow(q2, vmin=vmin_q, vmax=vmax_q)
        img_3 = axs[3,0].imshow(q3, vmin=0, vmax=2)

        img_46 = axs[0,1].imshow(a0, vmin=vmin_a, vmax=vmax_a)
        axs[1,1].imshow(a1, vmin=vmin_a, vmax=vmax_a)
        axs[2,1].imshow(a2, vmin=vmin_a, vmax=vmax_a)
        img_7 = axs[3,1].imshow(a3, vmin=0, vmax=2)

        axs[0,0].set_title('model')
        axs[0,1].set_title('target model')
        fig.colorbar(img_02, ax=axs[0:3,0], orientation="vertical")
        fig.colorbar(img_3, ax=axs[3,0], orientation="vertical")
        fig.colorbar(img_46, ax=axs[0:3,1], orientation="vertical")
        fig.colorbar(img_7, ax=axs[3,1], orientation="vertical")

        fig.suptitle(str(int(i/len(tensor_list)*100)) + ' %')

        # Build folder structure if it doesn't exist yet
        path = 'process' + str(yaml_p['process_nr']).zfill(5) + '/temp'
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + '/gif_' + str(i).zfill(5) + '.png', dpi=50, bbox_inches='tight')
        plt.close()
        print('saving frames: ' + str(int(i/len(name_list)*100)) + ' %')

    # set up parameters to generate gif
    duration = yaml_p['duration']
    fps = min(int(len(name_list)/duration),yaml_p['fps'])

    # Build GIF
    with imageio.get_writer('process' + str(yaml_p['process_nr']).zfill(5) + '/qmap.gif', mode='I', fps=fps) as writer:
        name_list = os.listdir(path)
        name_list.sort()
        n = 0
        for name in name_list:
            image = imageio.imread(path + '/' + name)
            writer.append_data(image)
            print('generating gif: ' + str(int(n/len(name_list)*100)) + ' %')
            n += 1

    # Delete temp folder
    shutil.rmtree(path)

def write_overview():
    df_env = pd.read_csv('process' + str(yaml_p['process_nr']).zfill(5) + '/log_environment.csv', names=['epi', 'size_x', 'size_z', 'pos_x', 'pos_z', 'tar_x', 'tar_z', 'rew_step', 'rew_epi'])
    rew_epi = np.array(df_env.iloc[:,8].dropna())

    N_epi = yaml_p['phase']
    cumsum_epi = np.cumsum(np.insert(rew_epi, 0, 0))
    mean_reward_epi = (cumsum_epi[N_epi:] - cumsum_epi[:-N_epi]) / float(N_epi)

    maximum = max(mean_reward_epi)

    df = pd.DataFrame.from_dict(yaml_p)
    df = df.drop([0]) # for some reason it imports the yaml_p file twice
    df.insert(len(df.columns),'rew_epi', maximum, True)
    dirpath = Path('overview.csv')
    if dirpath.exists() and dirpath.is_file():
        df.to_csv(dirpath, mode='a', header=False, index=False)
    else:
        df.to_csv(dirpath, mode='a', header=True, index=False)

def disp_overview():
    df = pd.read_csv('overview.csv')
    n = len(df.columns)-1
    m = int(np.ceil(np.sqrt(n)))
    n = int(np.floor(n/m))

    fig, axs = plt.subplots(n,m)
    x = 0
    for i in range(n):
        for j in range(m):
            if isinstance(df.iloc[0,x], str):
                check = all(elem == df.iloc[0,x] for elem in df.iloc[:,x])
                if check:
                    color='grey'
                else:
                    color='blue'
            else:
                if np.std(df.iloc[:,x])<1e-10:
                    color='grey'
                else:
                    color='blue'

            axs[i,j].scatter(df.iloc[:,x],df['rew_epi'], color=color)
            axs[i,j].set_title(df.columns[x])
            x += 1

    #fig.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=1)
    plt.show()
    plt.close()

def clear():
    dirpath = Path('process' + str(yaml_p['process_nr']).zfill(5) + '/temp')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    dirpath = Path('process' + str(yaml_p['process_nr']).zfill(5) + '/log_qmap')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    dirpath = Path('process' + str(yaml_p['process_nr']).zfill(5) + '/log_environment.csv')
    if dirpath.exists() and dirpath.is_file():
        os.remove(dirpath)

    dirpath = Path('process' + str(yaml_p['process_nr']).zfill(5) + '/log_agent.csv')
    if dirpath.exists() and dirpath.is_file():
        os.remove(dirpath)
