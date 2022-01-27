import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits import mplot3d
import pandas as pd
from pathlib import Path
import shutil
import imageio
import os
import torch
from sklearn.linear_model import LinearRegression
from scipy.stats import beta
from scipy.interpolate import interpn
import json

from utils.load_tf import tflog2pandas, many_logs2pandas

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

render_ratio = yaml_p['unit_xy']/yaml_p['unit_z']

def plot_reward():
    # read in logger file as pandas
    path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_train/'
    name_list = os.listdir(path_logger)
    name_list.sort()
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list)

    rew_epi = np.array(df['reward_epi'].dropna())
    #rew_step = np.array(df['reward_step'])

    # plot mean reward
    N_epi = 1
    cumsum_rew = np.cumsum(np.insert(rew_epi, 0, 0))
    mean_reward_epi = (cumsum_rew[N_epi:] - cumsum_rew[:-N_epi]) / float(N_epi)

    # plot big mean
    N_epi_big = int(len(rew_epi)/10)
    cumsum_rew_big = np.cumsum(np.insert(rew_epi, 0, 0))
    mean_reward_epi_big = (cumsum_rew_big[N_epi_big:] - cumsum_rew_big[:-N_epi_big]) / float(N_epi_big)

    # plot
    fig, ax = plt.subplots(1,1)
    ax.plot(rew_epi, alpha=0.3)
    ax.plot(mean_reward_epi)
    ax.plot(mean_reward_epi_big)

    #ax.set_title('max. mean (' + str(N_epi) + '): ' + str(np.round(max(mean_reward_epi),5)) + '   avg. reward (' + str(N_epi) + '): ' + str(np.round(np.mean(rew_epi),5)))
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    ax.tick_params(axis='y')

    ax.legend(
        ['reward',
        'running mean over ' + str(N_epi) + ' episodes, max: ' + str(np.round(max(mean_reward_epi),5)) + ', avg: ' + str(np.round(np.mean(rew_epi),5)),
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
        fig, axs = plt.subplots(4,1)

        idx_fra = np.arange(idx[i], idx[i+1],1)
        df_fra = df[df['epi_n'].isin(idx_fra)]

        for j in idx_fra:
            df_loc = df_fra[df_fra['epi_n'].isin([j])]

            # add legend
            legend = [-2, -1.5, -1, -0.5, 0, 0.5, 1]
            for l in range(len(legend)):
                c = np.argmin(np.abs(spectrum - legend[l]))
                axs[2].text(df_fra['size_x'].iloc[-1],df_fra['size_y'].iloc[-1]/len(legend)*(l), str(legend[l]), verticalalignment='bottom', horizontalalignment='left', color=colors[c], fontsize=5)

            c = np.argmin(np.abs(spectrum - df_loc['reward_epi'].iloc[-1]))

            # plot path in yz
            axs[0].set_title('yz')
            axs[0].set_aspect(1/render_ratio)
            axs[0].plot(df_loc['position_y'], df_loc['position_z'], color=colors[c])
            axs[0].scatter(df_loc['target_y'].dropna().iloc[-1], df_loc['target_z'].dropna().iloc[-1], s=20, facecolors='none', edgecolors='grey')
            axs[0].set_xlim(0,df_loc['size_y'].dropna().iloc[-1])
            axs[0].set_ylim(0,df_loc['size_z'].dropna().iloc[-1])

            # plot path in xz
            axs[1].set_title('xz')
            axs[1].set_aspect(1/render_ratio)
            axs[1].plot(df_loc['position_x'], df_loc['position_z'], color=colors[c])
            axs[1].scatter(df_loc['target_x'].dropna().iloc[-1], df_loc['target_z'].dropna().iloc[-1], s=20, facecolors='none', edgecolors='grey')
            axs[1].set_xlim(0,df_loc['size_x'].dropna().iloc[-1])
            axs[1].set_ylim(0,df_loc['size_z'].dropna().iloc[-1])

            # plot path in xy
            axs[2].set_title('xy')
            axs[2].set_aspect(1)
            axs[2].plot(df_loc['position_x'], df_loc['position_y'], color=colors[c])
            axs[2].scatter(df_loc['target_x'].dropna().iloc[-1], df_loc['target_y'].dropna().iloc[-1], s=20, facecolors='none', edgecolors='grey')
            axs[2].set_xlim(0,df_loc['size_x'].dropna().iloc[-1])
            axs[2].set_ylim(0,df_loc['size_y'].dropna().iloc[-1])

            step = df_loc['position_x'].index[0] + 1

        fig.suptitle(str(int(i/n_f*100)) + ' %')
        plt.subplots_adjust(wspace=0.5, hspace=1)

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

def plot_3d_path():
    # read in logger file as pandas
    path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/'
    name_list = os.listdir(path_logger)
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list)

    vmin = -1
    vmax = 0

    vn = 100
    spectrum = np.linspace(vmin, vmax, vn)
    colors = pl.cm.jet(np.linspace(0,1,vn))

    ax = plt.axes(projection='3d')

    d = 0
    for j in range(int(df['epi_n'].dropna().iloc[-1]) + 1):

        df_loc = df[df['epi_n'].isin([j])]
        #end = np.argmin(df_loc['min_proj_dist'])
        end = -2 #plot the whole trajectory
        df_loc_cut = df_loc.iloc[1:end+1]
        draw = True
        if len(df_loc) < 10:
            draw = False
        if (min(df_loc['position_x']) < 0) | (max(df_loc['position_x']) > yaml_p['size_x'] - 1):
            draw = False
        if len(df_loc['min_dist'].dropna()) == 0:
            draw = False
        if d > 9:
            draw = False

        draw = True #remove if only certain things should be plotted

        if draw:
            c = np.argmin(np.abs(spectrum + df_loc['min_dist'].iloc[-1]))

            # plot path in 3d
            ax.plot3D(df_loc_cut['position_x'], df_loc_cut['position_y'], df_loc_cut['position_z'], color=colors[c])
            ax.scatter(df_loc_cut['position_x'], df_loc_cut['position_y'], df_loc_cut['position_z'], s=0.01, color='black')
            #if yaml_p['3d']:
            #    ax.scatter3D(df_loc['target_x'], df_loc['target_y'], df_loc['target_z'], color='grey')
            #else:
            #    ax.plot3D(np.linspace(0,yaml_p['size_x'], 10), [df_loc['target_y'].iloc[-1]]*10, [df_loc['target_z'].iloc[-1]]*10, color='grey')

            # mark the border of the box
            ax.set_xlim3d(0, yaml_p['size_x'] - 1)
            ax.set_ylim3d(0, yaml_p['size_y'] - 1)
            ax.set_zlim3d(0, yaml_p['size_z'] - 1)

            d += 1

            #fig.suptitle(str(int(i/n_f*100)) + ' %')
            #plt.subplots_adjust(wspace=0.5, hspace=1)

    # Build folder structure if it doesn't exist yet
    path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/3dpath.png'
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close()

def plot_2d_path():
    # read in logger file as pandas
    path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/'
    name_list = os.listdir(path_logger)
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list)

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    d = 0
    for j in range(int(df['epi_n'].dropna().iloc[-1]) + 1):

        df_loc = df[df['epi_n'].isin([j])]
        end = np.argmin(df_loc['min_proj_dist'])

        df_loc_cut = df_loc.iloc[0:end+1]

        draw = True
        if len(df_loc) < 10:
            draw = False
        if (min(df_loc['position_x']) < 0) | (max(df_loc['position_x']) > yaml_p['size_x'] - 1):
            draw = False
        if len(df_loc['min_dist'].dropna()) == 0:
            draw = False
        if d > 9:
            draw = False

        if draw:
            dydx = np.linspace(0,1,int(yaml_p['T']/yaml_p['delta_t_logger']))  # first derivative

            #dense lines
            points = np.array([df_loc_cut['position_y'], df_loc_cut['position_z']]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(dydx.min(), dydx.max())
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(dydx)
            line = ax.add_collection(lc)

            """
            #transparent lines
            points_transp = np.array([df_loc['position_y'][end::], df_loc['position_z'][end::]]).T.reshape(-1, 1, 2)
            segments_transp = np.concatenate([points_transp[:-1], points_transp[1:]], axis=1)
            norm_transp = plt.Normalize(dydx.max(),1)
            lc_transp = LineCollection(segments_transp, cmap='viridis', norm=norm, alpha=0.1)
            line_transp = ax.add_collection(lc_transp)
            """

            ax.scatter(df_loc_cut['position_y'].iloc[-1], df_loc_cut['position_z'].iloc[-1], c=end, vmin=0, vmax=int(yaml_p['T']/yaml_p['delta_t_logger']), cmap='viridis')
            ax.scatter(df_loc['target_y'],df_loc['target_z'], color='red', zorder=1000) #zorder so the target is always above everything else

            ax.set_xlim(0,yaml_p['size_y'] - 1)
            ax.set_ylim(0,yaml_p['size_z'] - 1)

            d += 1

        #fig.suptitle(str(int(i/n_f*100)) + ' %')
        #plt.subplots_adjust(wspace=0.5, hspace=1)

    # Build folder structure if it doesn't exist yet
    path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/2dpath.png'
    plt.savefig(path, dpi=150)
    plt.close()

def make_2d_gif():
    Path('temp').mkdir(parents=True, exist_ok=True)

    # read in logger file as pandas
    path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/'
    name_list = os.listdir(path_logger)
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list)

    df['position_x'] *= yaml_p['unit_xy']
    df['position_y'] *= yaml_p['unit_xy']
    df['position_z'] *= yaml_p['unit_z']
    df['target_x'] *= yaml_p['unit_xy']
    df['target_y'] *= yaml_p['unit_xy']
    df['target_z'] *= yaml_p['unit_z']

    d = 0
    for j in range(int(df['epi_n'].dropna().iloc[-1]) + 1):
        if j == 12: #2,4,(6),10,12!,13,(14)
            df_loc = df[df['epi_n'].isin([j])]
            for g in range(len(df_loc)):
                fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
                min_proj_dist = np.sqrt(((df_loc['target_x'].iloc[-1] - df_loc['position_x'])*yaml_p['unit_xy'])**2 + ((df_loc['target_y'].iloc[-1] - df_loc['position_y'])*yaml_p['unit_xy'])**2 + ((df_loc['target_z'].iloc[-1] - df_loc['position_z'])*yaml_p['unit_z'])**2)
                end = np.argmin(df_loc['min_proj_dist'])
                end = np.argmin(min_proj_dist)+1
                df_loc_cut = df_loc.iloc[0:min(g,end)+1]

                if d != 0:
                    draw = False

                dydx = np.linspace(0,1,int(yaml_p['T']/yaml_p['delta_t']))  # first derivative

                #dense lines
                points = np.array([df_loc_cut['position_y'], df_loc_cut['position_z']]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                norm = plt.Normalize(dydx.min(), dydx.max())
                lc = LineCollection(segments, cmap='viridis', norm=norm)
                lc.set_array(dydx)
                line = ax.add_collection(lc)

                if g >= end:
                    ax.scatter(df_loc['position_y'].iloc[end], df_loc['position_z'].iloc[end], c=end, vmin=0, vmax=int(yaml_p['T']/yaml_p['delta_t']), cmap='viridis')

                #transparent lines
                if g >= end:
                    points_transp = np.array([df_loc['position_y'][0:g+1], df_loc['position_z'][0:g+1]]).T.reshape(-1, 1, 2)
                    segments_transp = np.concatenate([points_transp[:-1], points_transp[1:]], axis=1)
                    norm_transp = plt.Normalize(dydx.max(),1)
                    lc_transp = LineCollection(segments_transp, cmap='viridis', norm=norm, alpha=0.1)
                    line_transp = ax.add_collection(lc_transp)

                ax.scatter(df_loc['target_y'],df_loc['target_z'], color='red', zorder=1000) #zorder so the target is always above everything else

                ax.set_xlabel('position [m]')
                ax.set_ylabel('height [m]')
                ax.set_xlim(0,(yaml_p['size_y'] - 1)*yaml_p['unit_xy'])
                ax.set_ylim(0,(yaml_p['size_z'] - 1)*yaml_p['unit_z'])

                #fig.suptitle(str(int(i/n_f*100)) + ' %')
                #plt.subplots_adjust(wspace=0.5, hspace=1)

                path = 'temp/' + str(g).zfill(2) + '.png'
                plt.savefig(path, dpi=150)
                plt.close()

    # Build GIF
    with imageio.get_writer('debug_2d.gif', mode='I', fps=1/yaml_p['delta_t']) as writer:
        path = 'temp/'
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

def tuning(directory_compare=None):
    # read in logger file as pandas
    path_logger_list = []
    if directory_compare is not None:
        path_logger_list.append(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/' + directory_compare + '/')
    path_logger_list.append(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/')

    fig, axs = plt.subplots(5, 1)
    for p in range(len(path_logger_list)):
        path_logger = path_logger_list[p]
        name_list = os.listdir(path_logger)
        for i in range(len(name_list)):
            name_list[i] = path_logger + name_list[i]
        df = many_logs2pandas(name_list)
        cmap0 = plt.cm.get_cmap('winter')
        cmap1 = plt.cm.get_cmap('autumn')

        iter_max = int(df['epi_n'].dropna().iloc[-1]) + 1

        for j in range(iter_max):
            df_loc = df[df['epi_n'].isin([j])]
            end = np.argmin(df_loc['action'])
            end = np.argmin(abs(np.gradient(df_loc['position_x'])))
            end = 1000
            df_loc_cut = df_loc.iloc[0:end]
            time = yaml_p['T'] - df_loc_cut['t']

            if p == 0:
                if (j == 0) & (yaml_p['mode'] == 'tuning'): #beacuse when tuning it's always the same action cycle
                    axs[4].plot(time, df_loc_cut['action'], color='grey', linewidth=0.5)
                color=cmap0(1-j/iter_max)
            else:
                color=cmap1(1-j/iter_max)
            axs[0].plot(time, df_loc_cut['position_x']*yaml_p['unit_xy'], color=color, linewidth=0.2)
            axs[1].plot(time, df_loc_cut['position_y']*yaml_p['unit_xy'], color=color, linewidth=0.2)
            axs[2].plot(time, df_loc_cut['position_z']*yaml_p['unit_z'], color=color, linewidth=0.2)
            axs[3].plot(time, df_loc_cut['velocity_z']*yaml_p['unit_z'], color=color, linewidth=0.2)
            axs[4].plot(time, df_loc_cut['rel_pos_est'], color=color, linewidth=0.2)

            #fig.suptitle(str(int(i/n_f*100)) + ' %')
            #plt.subplots_adjust(wspace=0.5, hspace=1)

    axs[0].set_ylim(0,yaml_p['size_x']*yaml_p['unit_xy'])
    axs[1].set_ylim(0,yaml_p['size_y']*yaml_p['unit_xy'])
    axs[2].set_ylim(0,yaml_p['size_z']*yaml_p['unit_z'])
    axs[3].set_ylim(-3,3)
    axs[4].set_ylim(-0.1,1.1)

    for a in range(5):
        axs[a].set_xlim(0,yaml_p['T'])
        axs[a].set_xlabel('time [s]')

        axs[a].grid(which='minor', alpha=0.2, linewidth=0.5)
        axs[a].grid(which='major', alpha=0.5, linewidth=0.5)

    axs[0].set_ylabel('pos x [m]')
    axs[1].set_ylabel('pos y [m]')
    axs[2].set_ylabel('pos z [m]')
    axs[3].set_ylabel('vel z [m]')
    axs[4].set_ylabel('rel pos z')

    # Build folder structure if it doesn't exist yet
    path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/tuning.png'
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()

def dist_hist(abs_path_list=None):
    if abs_path_list is None:
        abs_path_list = [yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/']

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    legend_list = []
    for path_logger in abs_path_list:
        # read in logger file as pandas
        name_list = os.listdir(path_logger)
        for i in range(len(name_list)):
            name_list[i] = path_logger + name_list[i]
        df = many_logs2pandas(name_list)

        d = 0
        data = []
        for j in range(int(df['epi_n'].dropna().iloc[-1]) + 1):
            df_loc = df[df['epi_n'].isin([j])]
            draw = True
            if len(df_loc) < 10:
                draw = False
            if (min(df_loc['position_x']) < 0) | (max(df_loc['position_x']) > yaml_p['size_x'] - 1):
                draw = False
            if len(df_loc['min_dist'].dropna()) == 0:
                draw = False
            #if d > 9:
            #    draw = False

            if draw:
                data.append(df_loc['min_dist'].dropna().iloc[-1])
                d += 1

        # plot cummulative histogram
        data.sort()
        x = [0]
        y = [0]
        for i in range(len(data)):
            x.append(data[i])
            y.append((i+1)/len(data)*100)
        ax.plot(x,y)
        legend_list.append('median at ' + str(np.round(np.median(data),2)) + ' [m]')

    ax.set_xlabel('min radius to the target [m]')
    ax.set_ylabel('tries within that radius [%]')
    plt.legend(legend_list)

    path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/dist_hist.png'
    plt.savefig(path, dpi=150)
    plt.close()

def wind_est():
    # read in logger file as pandas
    path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/'
    name_list = os.listdir(path_logger)
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list)

    from scipy.interpolate import LinearNDInterpolator
    import seaborn as sns
    points = np.vstack([df['position_x'],df['position_y'],df['position_z']]).T
    interp = LinearNDInterpolator(points, df['measurement_y'])

    world_est = np.zeros((yaml_p['size_x'], yaml_p['size_y'], yaml_p['size_z']))
    for i in range(yaml_p['size_x']):
        for j in range(yaml_p['size_y']):
            for k in range(yaml_p['size_z']):
                world_est[i,j,k] = interp([i,j,k])

    fig, axs = plt.subplots(2)
    if yaml_p['balloon'] == 'outdoor_balloon':
        limit = 10
    elif yaml_p['balloon'] == 'indoor_balloon':
        limit = 1.5
    else:
        print('ERROR: Choose an existing balloon type')

    world_name = np.random.choice(os.listdir(yaml_p['data_path'] +'test/tensor'))
    h = 0
    world = torch.load(yaml_p['data_path'] + 'test/tensor/' + world_name)

    cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

    world_est_mean = np.nanmean(world_est,axis=0)
    world_gt_mean = np.nanmean(world[2],axis=0)

    #axs[0].imshow(world_est[int(yaml_p['size_x']/2),:,:].T, origin='lower', cmap=cmap, alpha=1, vmin=-limit, vmax=limit)
    axs[0].imshow(world_est_mean.T, origin='lower', cmap=cmap, alpha=1, vmin=-limit, vmax=limit)
    axs[1].imshow(world_gt_mean.T, origin='lower', cmap=cmap, alpha=1, vmin=-limit, vmax=limit)

    axs[0].set_aspect(yaml_p['unit_z']/yaml_p['unit_xy'])
    axs[1].set_aspect(yaml_p['unit_z']/yaml_p['unit_xy'])

    plt.savefig('debug_windest.png')
    plt.close()

def reachability_study():
    path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/reachability_study/'
    df = pd.read_csv(path + 'percentage.csv', header=None)
    df = df.sort_values(by=0)
    hours = []
    for h in df.iloc[:,0]:
        hours.append(h[-2::])
    plt.plot(hours, df.iloc[:,1]*100)
    plt.xlabel('hour')
    plt.ylabel('percentage of reachability')
    plt.savefig(path + 'graph.png')
    plt.close()

    # Build GIF
    with imageio.get_writer(path + 'timelaps.gif', mode='I', fps=3) as writer:
        name_list = os.listdir(path)

        # find png
        name_list_png = []
        for name in name_list:
            if ('.png' in name) & (not 'graph' in name):
                name_list_png.append(name)
        name_list_png.sort()

        n = 0
        for name in name_list_png:
            image = imageio.imread(path + '/' + name)
            writer.append_data(image)
            n += 1

def write_overview():
    # read in logger file as pandas
    path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/'
    name_list = os.listdir(path_logger)
    name_list.sort()
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list)
    rew_epi = np.array(df['reward_epi'].dropna())

    # maximum and mean
    N_epi = 1
    cumsum_rew = np.cumsum(np.insert(rew_epi, 0, 0))
    mean_reward_epi = (cumsum_rew[N_epi:] - cumsum_rew[:-N_epi]) / float(N_epi)

    maximum = max(mean_reward_epi)
    mean = np.mean(mean_reward_epi)

    # std of action
    action = np.array(df['action'].dropna())
    action_std = np.std(action)

    # mediam of last battery level
    lbl = np.array(df['last_battery_level'].dropna())
    lbl_median = np.median(lbl)

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
    df_reward.insert(len(df_reward.columns),'action_std', action_std, True)
    df_reward.insert(len(df_reward.columns),'last_battery_level_median', lbl_median, True)
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
                axs[i,j].grid(linewidth=0.3)
                # colors
                color_max='red'
                color_mean='Indigo'
                color_norm='MediumSlateBlue'
                color_success='violet'
                color_score='pink'
                color_action='green'

                # scatter
                """
                axs[i,j].scatter(df.iloc[:,x],df['rew_epi_max'], s=0.3, facecolors='none', edgecolors=color_max, alpha=0.2)
                axs[i,j].scatter(df.iloc[:,x],df['rew_epi_mean'], s=0.3, facecolors='none', edgecolors=color_mean, alpha=0.2)
                axs[i,j].scatter(df.iloc[:,x],df['linreg_intercept'], s=0.3, facecolors='none', edgecolors=color_intercept, alpha=0.2)
                axs[i,j].scatter(df.iloc[:,x],df['linreg_score'], s=0.3, facecolors='none', edgecolors=color_score, alpha=0.2)
                """

                #ax2 = axs[i,j].twinx()
                #ax2.tick_params(axis='y', colors='green')

                """
                ax2.scatter(df.iloc[:,x],df['linreg_slope'], s=0.3, facecolors='none', edgecolors=color_slope, alpha=0.2)
                """

                """
                # max
                df_mean_max = pd.concat([df.iloc[:,x], df['rew_epi_max']], axis=1)

                if df_mean_max.columns[0] != df_mean_max.columns[1]:
                    mean_rew_max = df_mean_max.groupby(df_mean_max.columns[0]).mean().reset_index()
                    axs[i,j].scatter(mean_rew_max.iloc[:,0], mean_rew_max.iloc[:,1], s=0.3, color=color_max)
                """

                # action_std
                df_action_std = pd.concat([df.iloc[:,x], df['action_std']], axis=1)

                if df_action_std.columns[0] != df_action_std.columns[1]:
                    action_std = df_action_std.groupby(df_action_std.columns[0]).mean().reset_index()
                    axs[i,j].scatter(action_std.iloc[:,0], action_std.iloc[:,1], s=0.3, color=color_action)

                # mean
                df_mean_mean = pd.concat([df.iloc[:,x], df['rew_epi_mean']], axis=1)

                if df_mean_mean.columns[0] != df_mean_mean.columns[1]:
                    mean_rew_mean = df_mean_mean.groupby(df_mean_mean.columns[0]).mean().reset_index()
                    axs[i,j].scatter(mean_rew_mean.iloc[:,0], mean_rew_mean.iloc[:,1], s=0.3, color=color_mean)

                # success
                df_mean_success = pd.concat([df.iloc[:,x], df['success_rate']], axis=1)

                if df_mean_success.columns[0] != df_mean_success.columns[1]:
                    mean_success = df_mean_success.groupby(df_mean_success.columns[0]).mean().reset_index()
                    axs[i,j].scatter(mean_success.iloc[:,0], mean_success.iloc[:,1], s=0.3, color=color_success)

                """
                df_mean_slope = pd.concat([df.iloc[:,x], df['linreg_slope']], axis=1)
                if df_mean_slope.columns[0] != df_mean_slope.columns[1]:
                    mean_linreg_slope = df_mean_slope.groupby(df_mean_slope.columns[0]).mean().reset_index()
                    ax2.scatter(mean_linreg_slope.iloc[:,0], mean_linreg_slope.iloc[:,1], s=0.3, color=color_slope)
                df_mean_intercept= pd.concat([df.iloc[:,x], df['linreg_intercept']], axis=1)
                if df_mean_intercept.columns[0] != df_mean_intercept.columns[1]:
                    mean_linreg_intercept = df_mean_intercept.groupby(df_mean_intercept.columns[0]).mean().reset_index()
                    axs[i,j].scatter(mean_linreg_intercept.iloc[:,0], mean_linreg_intercept.iloc[:,1], s=0.3, color=color_intercept)
                df_mean_score = pd.concat([df.iloc[:,x], df['linreg_score']], axis=1)
                if df_mean_score.columns[0] != df_mean_score.columns[1]:
                    mean_linreg_score = df_mean_score.groupby(df_mean_score.columns[0]).mean().reset_index()
                    axs[i,j].scatter(mean_linreg_score.iloc[:,0], mean_linreg_score.iloc[:,1], s=0.3, color=color_score)
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

    dirpath = Path(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/map_test')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath, ignore_errors=True)
        os.mkdir(dirpath) #recreate the folder I just deleted
