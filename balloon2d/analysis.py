import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import pandas as pd
from pathlib import Path
import shutil
import imageio
import os
import torch
from sklearn.linear_model import LinearRegression

from utils.load_tf import tflog2pandas, many_logs2pandas

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

def plot_reward():
    # read in logger file as pandas
    path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_train/'
    name_list = os.listdir(path_logger)
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list)

    rew_epi = np.array(df['reward_epi'].dropna())
    qloss = np.array(df['loss_qfunction'].dropna())
    #rew_step = np.array(df['reward_step'])

    # plot mean reward
    N_epi = yaml_p['phase']
    cumsum_rew = np.cumsum(np.insert(rew_epi, 0, 0))
    mean_reward_epi = (cumsum_rew[N_epi:] - cumsum_rew[:-N_epi]) / float(N_epi)

    # plot big mean
    N_epi_big = int(len(rew_epi)/10)
    cumsum_rew_big = np.cumsum(np.insert(rew_epi, 0, 0))
    mean_reward_epi_big = (cumsum_rew_big[N_epi_big:] - cumsum_rew_big[:-N_epi_big]) / float(N_epi_big)

    # linear regression
    Y = rew_epi
    X = np.linspace(0,len(rew_epi)-1,len(rew_epi)).reshape((-1, 1))

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions

    slope = linear_regressor.coef_[0]
    score = linear_regressor.score(X,Y)

    # plot
    fig, ax = plt.subplots(1,1)
    ax.plot(rew_epi, alpha=0.1)
    ax.plot(mean_reward_epi)
    ax.plot(Y_pred)
    ax.plot(mean_reward_epi_big)

    if yaml_p['cherry_pick'] > 0:
        # validation
        rew_epi_val = np.array(df['reward_epi_val'].dropna())
        N_epi_val = int(len(rew_epi)/10)
        cumsum_rew_val = np.cumsum(np.insert(rew_epi_val, 0, 0))
        mean_reward_epi_val = (cumsum_rew_val[N_epi_val:] - cumsum_rew_val[:-N_epi_val]) / float(N_epi_val)
        ax.plot(mean_reward_epi_val)

        # save_weights
        weights_saved = np.array(df['weights_saved'].dropna())
        for w in weights_saved:
            ax.axvline(w, color='black', linewidth=0.5)

    #ax.set_title('max. mean (' + str(N_epi) + '): ' + str(np.round(max(mean_reward_epi),5)) + '   avg. reward (' + str(N_epi) + '): ' + str(np.round(np.mean(rew_epi),5)))
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    ax.tick_params(axis='y')

    ax.legend(
        ['reward',
        'running mean over ' + str(N_epi) + ' episodes, max: ' + str(np.round(max(mean_reward_epi),5)) + ', avg: ' + str(np.round(np.mean(rew_epi),5)),
        r'linear regression, slope $\times$ N_epi: ' + str(np.round(slope*N_epi,5)) + ', score: ' + str(np.round(score,5)),
        'running mean over ' + str(N_epi_big) + ' episodes', 'reward_val']
        )

    fig.tight_layout()
    plt.savefig(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/learning_curve.pdf')
    plt.close()

def plot_path():
    # read in logger file as pandas
    path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_train/'
    name_list = os.listdir(path_logger)
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list)

    # set up parameters to generate gif
    duration = yaml_p['duration']
    N = df['epi_n'].dropna().iloc[-1]+1
    fps = min(int(N/duration),yaml_p['fps'])

    n_f = duration*fps
    idx = np.linspace(0,N-N/n_f,n_f)
    idx = [int(i) for i in idx]

    #vmin = np.min(df['reward_epi'])
    #vmax = np.max(df['reward_epi'])
    vmin = -2
    vmax = 1


    vn = 100
    spectrum = np.linspace(vmin, vmax, vn)
    colors = pl.cm.jet(np.linspace(0,1,vn))

    step = 0
    for i in range(len(idx)-1):
        fig, axs = plt.subplots(2,1)

        idx_fra = np.arange(idx[i], idx[i+1],1)
        df_fra = df[df['epi_n'].isin(idx_fra)]

        for j in idx_fra:
            df_loc = df_fra[df_fra['epi_n'].isin([j])]

            # add legend
            legend = [-2, -1.5, -1, -0.5, 0, 0.5, 1]
            for l in range(len(legend)):
                c = np.argmin(np.abs(spectrum - legend[l]))
                axs[0].text(df_fra['size_x'].iloc[-1],df_fra['size_z'].iloc[-1]/len(legend)*(l), str(legend[l]), verticalalignment='bottom', horizontalalignment='left', color=colors[c])

            # add path
            c = np.argmin(np.abs(spectrum - df_loc['reward_epi'].iloc[-1]))

            # plot path
            axs[0].plot(df_loc['position_x'], df_loc['position_z'], color=colors[c])
            axs[0].scatter(df_loc['target_x'].dropna().iloc[-1], df_loc['target_z'].dropna().iloc[-1], s=20, facecolors='none', edgecolors='grey')
            axs[0].set_xlim(0,df_loc['size_x'].dropna().iloc[-1])
            axs[0].set_ylim(0,df_loc['size_z'].dropna().iloc[-1])

            step = df_loc['position_x'].index[0] + 1

        axs[0].set_title(str(int(i/n_f*100)) + ' %')

        # plot epsilon
        df = df.interpolate() #to fill the NaN values
        axs[1].plot(df['epsilon'].iloc[0:step], color='grey')
        axs[1].set_xlabel('steps')
        axs[1].set_ylabel('epsilon')
        axs[1].set_xlim(0,step)
        axs[1].set_ylim(0,1)

        # Build folder structure if it doesn't exist yet
        path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/temp'
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + '/gif_' + str(i).zfill(5) + '.png', dpi=150)
        plt.close()
        print('saving frames: ' + str(int(i/n_f*100)) + ' %')

    # Build GIF
    with imageio.get_writer(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/path.gif', mode='I', fps=fps) as writer:
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


def write_overview():
    # read in logger file as pandas
    path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/'
    name_list = os.listdir(path_logger)
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list)

    rew_epi = np.array(df['reward_epi'].dropna())

    # maximum and mean
    N_epi = yaml_p['phase']
    cumsum_rew = np.cumsum(np.insert(rew_epi, 0, 0))
    mean_reward_epi = (cumsum_rew[N_epi:] - cumsum_rew[:-N_epi]) / float(N_epi)

    maximum = max(mean_reward_epi)
    mean = np.mean(mean_reward_epi)

    # mean of normalized reward
    rew_epi_norm = np.array(df['reward_epi_norm'].dropna())
    mean_norm = np.mean(rew_epi_norm)

    # linear regression
    Y = rew_epi
    X = np.linspace(0,len(rew_epi)-1,len(rew_epi)).reshape((-1, 1))

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions

    slope = linear_regressor.coef_[0]
    intercept = linear_regressor.intercept_
    score = linear_regressor.score(X,Y)

    # to pass in pandas	df
    dic_copy = yaml_p.copy()
    for i in dic_copy:
        dic_copy[i] = [dic_copy[i]]

    # success_n
    success_n = np.array(df['success_n'].dropna())
    success_rate = success_n[-1]/yaml_p['num_epochs_test']

    # write down
    df_reward = pd.DataFrame(dic_copy)
    df_reward.insert(len(df_reward.columns),'rew_epi_max', maximum, True)
    df_reward.insert(len(df_reward.columns),'rew_epi_mean', mean, True)
    df_reward.insert(len(df_reward.columns),'rew_epi_norm_mean', mean_norm, True)
    df_reward.insert(len(df_reward.columns),'linreg_slope', slope, True)
    df_reward.insert(len(df_reward.columns),'linreg_intercept', slope, True)
    df_reward.insert(len(df_reward.columns),'linreg_score', score, True)
    df_reward.insert(len(df_reward.columns),'success_rate', success_rate, True)
    dirpath = Path('overview.csv')
    if dirpath.exists() and dirpath.is_file():
        df_reward.to_csv(dirpath, mode='a', header=False, index=False)
    else:
        df_reward.to_csv(dirpath, mode='a', header=True, index=False)

def disp_overview():
    df = pd.read_csv('overview.csv')

    # determening the number of different subplots needed
    to_delete = []
    for i in range(len(df.columns)):
        if isinstance(df.iloc[0,i], str):
            check = all(elem == df.iloc[0,i] for elem in df.iloc[:,i])
            if check:
                to_delete.append(i)
        else:
            if np.std(df.iloc[:,i])<1e-10:
                to_delete.append(i)

    df.drop(df.columns[to_delete],axis=1,inplace=True)

    N = len(df.columns)
    n = int(np.ceil(np.sqrt(N)))
    m = int(np.ceil(N/n))

    fig, axs = plt.subplots(n,m)

    x = 0
    for i in range(n):
        for j in range(m):
            if x < len(df.columns):
                axs[i,j].grid(linewidth=0.1)
                # colors
                color_max='red'
                color_mean='Indigo'
                color_norm='MediumSlateBlue'
                color_success='violet'
                color_slope='green'
                color_intercept='orange'
                color_score='pink'

                # scatter
                """
                axs[i,j].scatter(df.iloc[:,x],df['rew_epi_max'], s=0.1, facecolors='none', edgecolors=color_max, alpha=0.2)
                axs[i,j].scatter(df.iloc[:,x],df['rew_epi_mean'], s=0.1, facecolors='none', edgecolors=color_mean, alpha=0.2)
                axs[i,j].scatter(df.iloc[:,x],df['linreg_intercept'], s=0.1, facecolors='none', edgecolors=color_intercept, alpha=0.2)
                axs[i,j].scatter(df.iloc[:,x],df['linreg_score'], s=0.1, facecolors='none', edgecolors=color_score, alpha=0.2)
                """

                #ax2 = axs[i,j].twinx()
                #ax2.tick_params(axis='y', colors='green')

                """
                ax2.scatter(df.iloc[:,x],df['linreg_slope'], s=0.1, facecolors='none', edgecolors=color_slope, alpha=0.2)
                """

                """
                # max
                df_mean_max = pd.concat([df.iloc[:,x], df['rew_epi_max']], axis=1)

                if df_mean_max.columns[0] != df_mean_max.columns[1]:
                    mean_rew_max = df_mean_max.groupby(df_mean_max.columns[0]).mean().reset_index()
                    axs[i,j].scatter(mean_rew_max.iloc[:,0], mean_rew_max.iloc[:,1], s=0.1, color=color_max)
                """

                # mean
                df_mean_mean = pd.concat([df.iloc[:,x], df['rew_epi_mean']], axis=1)
                if df_mean_mean.columns[0] != df_mean_mean.columns[1]:
                    mean_rew_mean = df_mean_mean.groupby(df_mean_mean.columns[0]).mean().reset_index()
                    axs[i,j].scatter(mean_rew_mean.iloc[:,0], mean_rew_mean.iloc[:,1], s=0.1, color=color_mean)

                # mean_norm
                df_mean_norm = pd.concat([df.iloc[:,x], df['rew_epi_norm_mean']], axis=1)
                if df_mean_norm.columns[0] != df_mean_norm.columns[1]:
                    mean_rew_norm = df_mean_norm.groupby(df_mean_norm.columns[0]).mean().reset_index()
                    axs[i,j].scatter(mean_rew_norm.iloc[:,0], mean_rew_norm.iloc[:,1], s=0.1, color=color_norm)

                # success
                df_mean_success = pd.concat([df.iloc[:,x], df['success_rate']], axis=1)

                if df_mean_success.columns[0] != df_mean_success.columns[1]:
                    mean_success = df_mean_success.groupby(df_mean_success.columns[0]).mean().reset_index()
                    axs[i,j].scatter(mean_success.iloc[:,0], mean_success.iloc[:,1], s=0.1, color=color_success)

                """
                df_mean_slope = pd.concat([df.iloc[:,x], df['linreg_slope']], axis=1)
                if df_mean_slope.columns[0] != df_mean_slope.columns[1]:
                    mean_linreg_slope = df_mean_slope.groupby(df_mean_slope.columns[0]).mean().reset_index()
                    ax2.scatter(mean_linreg_slope.iloc[:,0], mean_linreg_slope.iloc[:,1], s=0.1, color=color_slope)
                df_mean_intercept= pd.concat([df.iloc[:,x], df['linreg_intercept']], axis=1)
                if df_mean_intercept.columns[0] != df_mean_intercept.columns[1]:
                    mean_linreg_intercept = df_mean_intercept.groupby(df_mean_intercept.columns[0]).mean().reset_index()
                    axs[i,j].scatter(mean_linreg_intercept.iloc[:,0], mean_linreg_intercept.iloc[:,1], s=0.1, color=color_intercept)
                df_mean_score = pd.concat([df.iloc[:,x], df['linreg_score']], axis=1)
                if df_mean_score.columns[0] != df_mean_score.columns[1]:
                    mean_linreg_score = df_mean_score.groupby(df_mean_score.columns[0]).mean().reset_index()
                    axs[i,j].scatter(mean_linreg_score.iloc[:,0], mean_linreg_score.iloc[:,1], s=0.1, color=color_score)
                """

                axs[i,j].set_title(df.columns[x])
                x += 1

    #fig.tight_layout()
    #fig.suptitle('max reward: red     mean reward: blue     success rate: violet     linreg slope: green     linreg intercept: orange     linreg scoret: pink')
    fig.suptitle('mean reward: blue     success rate: violet')
    plt.subplots_adjust(wspace=0.5, hspace=1)
    plt.show()
    plt.close()

def clear(train_or_test):
    dirpath = Path(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_' + train_or_test)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath, ignore_errors=True)

    dirpath = Path(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/temp')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath, ignore_errors=True)

    dirpath = Path(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/log_qmap')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath, ignore_errors=True)

    dirpath = Path(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/temp_w')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath, ignore_errors=True)
