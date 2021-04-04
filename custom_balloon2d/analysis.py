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


def plot_path():
    # read in logger file as pandas
    from load_tf import tflog2pandas, many_logs2pandas
    path_logger = yaml_p['path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger/'
    name_list = os.listdir(path_logger)
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list)

    # set up parameters to generate gif
    duration = yaml_p['duration']
    N = df['episode'].iloc[-1]+1
    fps = min(int(N/duration),yaml_p['fps'])

    n_f = duration*fps
    idx = np.linspace(0,N-N/n_f,n_f)
    idx = [int(i) for i in idx]

    vmin = min(df['reward_epi'])
    vmax = max(df['reward_epi'])
    vn = 100
    spectrum = np.linspace(vmin, vmax, vn)
    colors = pl.cm.jet(np.linspace(0,1,vn))

    step = 0
    for i in range(len(idx)-1):
        fig, axs = plt.subplots(2,1)

        idx_fra = np.arange(idx[i], idx[i+1],1)
        df_fra = df[df['episode'].isin(idx_fra)]

        for j in idx_fra:
            df_loc = df_fra[df_fra['episode'].isin([j])]

            c = np.argmin(np.abs(spectrum - df_loc['reward_epi'].iloc[-1]))

            # plot path
            axs[0].plot(df_loc['position_x'], df_loc['position_z'], color=colors[c])
            axs[0].scatter(df_loc['target_x'], df_loc['target_z'], s=20, facecolors='none', edgecolors='grey')
            axs[0].set_xlim(0,df_loc['size_x'].iloc[-1])
            axs[0].set_ylim(0,df_loc['size_z'].iloc[-1])

            step += len(df_loc['position_x'])

        axs[0].set_aspect('equal')
        axs[0].set_title(str(int(i/n_f*100)) + ' %')

        # plot epsilon
        df = df.interpolate() #to fill the NaN values
        axs[1].plot(df['epsilon'].iloc[0:step], color='grey')
        axs[1].set_xlabel('steps')
        axs[1].set_ylabel('epsilon')
        axs[1].set_xlim(0,step)
        axs[1].set_ylim(0,1)

        # Build folder structure if it doesn't exist yet
        path = yaml_p['path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/temp'
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + '/gif_' + str(i).zfill(5) + '.png', dpi=50)
        plt.close()
        print('saving frames: ' + str(int(i/n_f*100)) + ' %')

    # Build GIF
    with imageio.get_writer(yaml_p['path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/path.gif', mode='I', fps=fps) as writer:
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
    name_list = os.listdir(yaml_p['path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/log_qmap')
    name_list.sort()
    tensor_list = []
    for name in name_list:
        tensor_list.append(torch.load(yaml_p['path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/log_qmap/' + name))

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
        path = yaml_p['path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/temp'
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + '/gif_' + str(i).zfill(5) + '.png', dpi=50, bbox_inches='tight')
        plt.close()
        print('saving frames: ' + str(int(i/len(name_list)*100)) + ' %')

    # set up parameters to generate gif
    duration = yaml_p['duration']
    fps = min(int(len(name_list)/duration),yaml_p['fps'])

    # Build GIF
    with imageio.get_writer(yaml_p['path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/qmap.gif', mode='I', fps=fps) as writer:
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
    # read in logger file as pandas
    from load_tf import tflog2pandas, many_logs2pandas
    path_logger = yaml_p['path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger/'
    name_list = os.listdir(path_logger)
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list)

    rew_epi = np.array(df['reward_epi'].dropna())

    N_epi = yaml_p['phase']
    cumsum_epi = np.cumsum(np.insert(rew_epi, 0, 0))
    mean_reward_epi = (cumsum_epi[N_epi:] - cumsum_epi[:-N_epi]) / float(N_epi)

    maximum = max(mean_reward_epi)
    mean = np.mean(mean_reward_epi)

    df_reward = pd.DataFrame.from_dict(yaml_p)
    df_reward = df_reward.drop([0]) # for some reason it imports the yaml_p file twice
    df_reward.insert(len(df_reward.columns),'rew_epi_max', maximum, True)
    df_reward.insert(len(df_reward.columns),'rew_epi_mean', mean, True)
    dirpath = Path('overview.csv')
    if dirpath.exists() and dirpath.is_file():
        df_reward.to_csv(dirpath, mode='a', header=False, index=False)
    else:
        df_reward.to_csv(dirpath, mode='a', header=True, index=False)

def disp_overview():
    df = pd.read_csv('overview.csv')
    n = len(df.columns)-1
    m = int(np.floor(np.sqrt(n)))
    n = int(np.floor(n/m))

    fig, axs = plt.subplots(n,m)
    x = 0
    for i in range(n):
        for j in range(m):
            if isinstance(df.iloc[0,x], str):
                check = all(elem == df.iloc[0,x] for elem in df.iloc[:,x])
                if check:
                    color_max='grey'
                    color_mean='grey'
                else:
                    color_max='red'
                    color_mean='blue'
            else:
                if np.std(df.iloc[:,x])<1e-10:
                    color_max='grey'
                    color_mean='grey'
                else:
                    color_max='red'
                    color_mean='blue'

            axs[i,j].scatter(df.iloc[:,x],df['rew_epi_max'], s=0.1, facecolors='none', edgecolors=color_max)
            axs[i,j].scatter(df.iloc[:,x],df['rew_epi_mean'], s=0.1, facecolors='none', edgecolors=color_mean)
            axs[i,j].set_title(df.columns[x])
            x += 1

    #fig.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=1)
    plt.show()
    plt.close()

def clear():
    dirpath = Path(yaml_p['path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    dirpath = Path(yaml_p['path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/temp')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    dirpath = Path(yaml_p['path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/log_qmap')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
