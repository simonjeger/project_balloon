from tkinter import E
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits import mplot3d
import seaborn as sns
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
import simplekml

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
    N_epi = 2000
    cumsum_rew = np.cumsum(np.insert(rew_epi, 0, 0))
    mean_reward_epi = (cumsum_rew[N_epi:] - cumsum_rew[:-N_epi]) / float(N_epi)

    # plot
    fig, ax = plt.subplots(1,1)
    ax.plot(mean_reward_epi, color='black')

    #ax.set_title('max. mean (' + str(N_epi) + '): ' + str(np.round(max(mean_reward_epi),5)) + '   avg. reward (' + str(N_epi) + '): ' + str(np.round(np.mean(rew_epi),5)))
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    ax.tick_params(axis='y')
    ax.set_aspect(5000)

    ax.legend(
        ['Running mean over ' + str(N_epi) + ' episodes'
        ])

    fig.tight_layout()
    plt.savefig(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/learning_curve.png', dpi=500)
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
    type = 'iso'
    res = 5
    fps = 20
    # read in logger file as pandas
    path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/'
    dfs = []
    name_list = os.listdir(path_logger)
    name_list.sort()
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
        if 'events' in name_list[i]:
            dfs.append(tflog2pandas(name_list[i]))

    dfs[1]['position_x'] = dfs[1]['position_x'].iloc[0:500]
    dfs[1]['position_y'] = dfs[1]['position_y'].iloc[0:500]
    dfs[1]['position_z'] = dfs[1]['position_z'].iloc[0:500]

    # finding scale of map (in a horrible way)
    max_x_0 = np.max(dfs[0]['position_x'])
    max_y_0 = np.max(dfs[0]['position_y'])
    max_x_1 = np.max(dfs[1]['position_x'])
    max_y_1 = np.max(dfs[1]['position_y'])
    min_x_0 = np.min(dfs[0]['position_x'])
    min_y_0 = np.min(dfs[0]['position_y'])
    min_x_1 = np.min(dfs[1]['position_x'])
    min_y_1 = np.min(dfs[1]['position_y'])

    max_x = int(np.ceil(np.max([max_x_0,max_x_1])))
    max_y = int(np.ceil(np.max([max_y_0,max_y_1])))
    min_x = int(np.floor(np.min([min_x_0,min_x_1])))
    min_y = int(np.floor(np.min([min_y_0,min_y_1])))

    len_x = (max_x - min_x)*yaml_p['unit_xy']
    len_y = (max_y - min_y)*yaml_p['unit_xy']

    ax = plt.axes(projection='3d', computed_zorder=False)

    #plot surface
    list_of_worlds = os.listdir(yaml_p['data_path'] + 'test/tensor')
    world_name = list_of_worlds[0]
    world = torch.load(yaml_p['data_path'] + 'test/tensor/' + world_name)
    #z = world[0,min_x:max_x,min_y:max_y,0]*yaml_p['unit_z']
    z = world[0,:,:,0]*yaml_p['unit_z']
    #x = np.outer(np.linspace((min_x-dfs[0]['position_x'].iloc[0])*yaml_p['unit_xy'], (max_x-dfs[0]['position_x'].iloc[0])*yaml_p['unit_xy'], len(z)), np.ones(len(z[0])))
    #y = np.outer(np.ones(len(z)), np.linspace(min_y-dfs[0]['position_y'].iloc[1]*yaml_p['unit_xy'], (max_y-dfs[0]['position_y'].iloc[1])*yaml_p['unit_xy'], len(z[0])))
    x = np.outer(np.linspace((-dfs[0]['position_x'].iloc[0])*yaml_p['unit_xy'], (yaml_p['size_x']-dfs[0]['position_x'].iloc[0])*yaml_p['unit_xy'], len(z)), np.ones(len(z[0])))
    y = np.outer(np.ones(len(z)), np.linspace(0-dfs[0]['position_y'].iloc[1]*yaml_p['unit_xy'], (yaml_p['size_y']-dfs[0]['position_y'].iloc[1])*yaml_p['unit_xy'], len(z[0])))
    #x = np.arange(min_x, max_x, 1)
    #y = np.arange(min_y, max_y, 1)
    #x, y = np.meshgrid(x, y)
    #r = np.sqrt(x**2 + y**2)
    #z = np.sin(r)

    #c_contour = sns.cubehelix_palette(start=1.28, rot=0, dark=0.2, light=0.7, reverse=True, as_cmap=True)
    c_contour = 'summer'
    ax.plot_surface(x, y, z,cmap=c_contour, edgecolor='none', zorder=0)

    path_iso = 'temp_3d_iso'
    Path(path_iso).mkdir(parents=True, exist_ok=True)
    path_bird = 'temp_3d_bird'
    Path(path_bird).mkdir(parents=True, exist_ok=True)

    d = 0
    for j in range(int(dfs[0]['epi_n'].dropna().iloc[-1]) + 1):
        df_loc = dfs[0][dfs[0]['epi_n'].isin([j])]

        for f in range(int(len(df_loc)/res)):
            f *= res
            #end = 700
            end = f
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

            draw = True #remove if only certain things should be plotted

            if draw:
                # plot path in 3d
                ax.plot3D((dfs[1]['position_x']-dfs[1]['position_x'].iloc[0])*yaml_p['unit_xy'], (dfs[1]['position_y']-dfs[1]['position_y'].iloc[0])*yaml_p['unit_xy'], dfs[1]['position_z']*yaml_p['unit_z'], color='lightsteelblue', zorder=1000, linewidth=0.75)
                ax.plot3D((df_loc_cut['position_x']-df_loc_cut['position_x'].iloc[0])*yaml_p['unit_xy'], (df_loc_cut['position_y']-df_loc_cut['position_y'].iloc[0])*yaml_p['unit_xy'], df_loc_cut['position_z']*yaml_p['unit_z'], color='midnightblue', zorder=1000, linewidth=0.75)

                pos_x = []
                pos_y = []
                for i in range(len(df_loc_cut)):
                    pos_x.append(df_loc_cut['position_x'].iloc[i])
                    pos_y.append(df_loc_cut['position_y'].iloc[i])

                if yaml_p['3d']:
                    ax.scatter3D(0, 0, df_loc_cut['position_z'].iloc[0]*yaml_p['unit_z'], color='darkgreen', zorder=5)
                    ax.scatter3D((yaml_p['target_test'][0]-df_loc_cut['position_x'].iloc[0])*yaml_p['unit_xy'], (yaml_p['target_test'][1]-df_loc_cut['position_y'].iloc[0])*yaml_p['unit_xy'], yaml_p['target_test'][2]*yaml_p['unit_z'], color='firebrick', zorder=5)
                else:
                    ax.plot3D(np.linspace(0,yaml_p['size_x']*yaml_p['unit_xy'], 10), [(yaml_p['target_test'][1].iloc[-1]-df_loc_cut['position_y'].iloc[0])*yaml_p['unit_xy']]*10, [yaml_p['target_test'][2].iloc[-1]*yaml_p['unit_z']]*10, color='firebrick', zorder=5)

                closest_idx = np.argmin(df_loc_cut['min_dist'])
                min_dist = np.sqrt((yaml_p['unit_xy']*(df_loc_cut['position_x'] - yaml_p['target_test'][0]))**2 + (yaml_p['unit_xy']*(df_loc_cut['position_y'] - yaml_p['target_test'][1]))**2 + (yaml_p['unit_z']*(df_loc_cut['position_z'] - yaml_p['target_test'][2]))**2)
                closest_idx = np.argmin(min_dist)

                #velocity = np.max(np.sqrt((np.gradient(pos_x)/yaml_p['delta_t']*yaml_p['unit_xy'])**2 + (np.gradient(pos_y)/yaml_p['delta_t']*yaml_p['unit_xy'])**2))
                #ax.set_title('min_distance: ' + str(int(min_dist[closest_idx])) + ' m after ' + str(int(yaml_p['T'] - df_loc_cut['t'].iloc[-1])) + ' s, reaching a top speed of ' + str(int(velocity)) + ' m/s')
                ax.set_title('min. distance to target: ' + str(int(min_dist[closest_idx])) + ' m')

                d += 1

                #fig.suptitle(str(int(i/n_f*100)) + ' %')
                #plt.subplots_adjust(wspace=0.5, hspace=1)

            # mark the border of the box
            if type != 'iso':
                ax.set_xlim3d((min_x-dfs[0]['position_x'].iloc[0])*yaml_p['unit_xy'],(max_x-dfs[0]['position_x'].iloc[0])*yaml_p['unit_xy'])
                ax.set_ylim3d((min_y-dfs[0]['position_y'].iloc[0])*yaml_p['unit_xy'],(max_y-dfs[0]['position_y'].iloc[0])*yaml_p['unit_xy'])
            ax.set_zlim3d(0, (yaml_p['size_z'] - 1)*yaml_p['unit_z'])

            # Find correct aspect ration
            if type == 'iso':
                len_x = yaml_p['size_x']*yaml_p['unit_xy']
                len_y = yaml_p['size_y']*yaml_p['unit_xy']
            len_z = yaml_p['size_z']*yaml_p['unit_z']
            ax.set_box_aspect(aspect = (1,len_y/len_x,len_z/len_x))
            ax.set_zticks(np.arange(0, yaml_p['size_z']*yaml_p['unit_z'], step=1500))

            # Build folder structure if it doesn't exist yet
            #path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/'
            if type == 'iso':
                ax.view_init(30, -90)
                plt.savefig(path_iso + '/3dpath' + str(f).zfill(5) + '.png', dpi=250)
            else:
                ax.view_init(90, -90)
                ax.set_zticks([])
                plt.savefig(path_bird + '/3dpath' + str(f).zfill(5) + '.png', dpi=250)
            print('3d_path is ' + str(int(100*f/len(dfs[0]))) + ' % done')

    if type == 'iso':
        # Build GIF
        with imageio.get_writer(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/3d_iso.gif', mode='I', fps=fps) as writer:
            name_list = os.listdir(path_iso)
            name_list.sort()
            n = 0
            for name in name_list:
                image = imageio.imread(path_iso + '/' + name)
                writer.append_data(image)
                print('generating gif: ' + str(int(n/len(name_list)*100)) + ' %')
                n += 1
        shutil.rmtree(path_iso)

    else:
        # Build GIF
        with imageio.get_writer(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/3d_bird.gif', mode='I', fps=fps) as writer:
            name_list = os.listdir(path_bird)
            name_list.sort()
            n = 0
            for name in name_list:
                image = imageio.imread(path_bird + '/' + name)
                writer.append_data(image)
                print('generating gif: ' + str(int(n/len(name_list)*100)) + ' %')
                n += 1
        shutil.rmtree(path_bird)

def plot_3d_projection():
    # read in logger file as pandas
    path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/'
    name_list = os.listdir(path_logger)
    name_list.sort(reverse=True)

    # find boundries
    name_list_bound = name_list[:]
    for i in range(len(name_list)):
        name_list_bound[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list_bound)

    x_low = min(df['position_x']*yaml_p['unit_xy'])
    x_high = max(df['position_x']*yaml_p['unit_xy'])
    y_low = min(df['position_y']*yaml_p['unit_xy'])
    y_high = max(df['position_y']*yaml_p['unit_xy'])
    z_low = 0
    z_high = yaml_p['size_z']*yaml_p['unit_z']

    x_range = x_high - x_low
    y_range = y_high - y_low
    z_range = z_high - z_low

    fig, axs = plt.subplots(3,1, gridspec_kw={'height_ratios': [1, z_range/y_range, z_range/y_range]})

    ax_0 = axs[0]
    ax_1 = axs[1]
    ax_2 = axs[2]

    i = 0
    for l in range(len(name_list)):
        name_list[l] = path_logger + name_list[l]
        if 'events' in name_list[l]:
            df = tflog2pandas(name_list[l])

            vmin = -1
            vmax = 0

            vn = 100
            spectrum = np.linspace(vmin, vmax, vn)
            colors = pl.cm.jet(np.linspace(0,1,vn))

            ax_0.grid()
            ax_1.grid()
            ax_2.grid()

            for j in range(int(df['epi_n'].dropna().iloc[-1]) + 1):
                df_loc = df[df['epi_n'].isin([j])]

                end = -2
                # when sim takes forever
                """
                if i == 0:
                    end = 650
                else:
                    end = -2 #plot the whole trajectory
                """

                df_loc_cut = df_loc.iloc[1:end+1]

                draw = True #remove if only certain things should be plotted

                if draw:
                    c = np.argmin(np.abs(spectrum + df_loc['min_dist'].iloc[-1]))

                    closest_idx = np.argmin(df_loc_cut['min_dist'])
                    min_dist = np.sqrt((yaml_p['unit_xy']*(df_loc_cut['position_x'] - yaml_p['target_test'][0]))**2 + (yaml_p['unit_xy']*(df_loc_cut['position_y'] - yaml_p['target_test'][1]))**2 + (yaml_p['unit_z']*(df_loc_cut['position_z'] - yaml_p['target_test'][2]))**2)
                    closest_idx = np.argmin(min_dist)

                    overwrite_idx = np.argmax(df_loc_cut['action_overwrite'])
                    if overwrite_idx == 0:
                        overwrite_idx = len(df_loc_cut['action_overwrite'])

                    if i == 0:
                        color_path = 'lightsteelblue'
                        color_point = 'lightsteelblue'

                        orgin_x = df_loc_cut['position_x'].iloc[0]
                        orgin_y = df_loc_cut['position_y'].iloc[0]
                        orgin_z = df_loc_cut['position_z'].iloc[0]
                    else:
                        color_path = 'midnightblue'
                        color_point = 'midnightblue'

                    # plot path
                    ax_0.plot((df_loc_cut['position_x'].iloc[0:overwrite_idx+1]-orgin_x)*yaml_p['unit_xy'], (df_loc_cut['position_y'].iloc[0:overwrite_idx+1]-orgin_y)*yaml_p['unit_xy'], color=color_path, zorder=-1)
                    ax_1.plot((df_loc_cut['position_x'].iloc[0:overwrite_idx+1]-orgin_x)*yaml_p['unit_xy'], df_loc_cut['position_z'].iloc[0:overwrite_idx+1]*yaml_p['unit_z'], color=color_path, zorder=-1)
                    ax_2.plot((df_loc_cut['position_y'].iloc[0:overwrite_idx+1]-orgin_y)*yaml_p['unit_xy'], df_loc_cut['position_z'].iloc[0:overwrite_idx+1]*yaml_p['unit_z'], color=color_path, zorder=-1)

                    ax_0.plot((df_loc_cut['position_x'].iloc[overwrite_idx::]-orgin_x)*yaml_p['unit_xy'], (df_loc_cut['position_y'].iloc[overwrite_idx::]-orgin_y)*yaml_p['unit_xy'], color='lightgrey', zorder=-1)
                    ax_1.plot((df_loc_cut['position_x'].iloc[overwrite_idx::]-orgin_x)*yaml_p['unit_xy'], df_loc_cut['position_z'].iloc[overwrite_idx::]*yaml_p['unit_z'], color='lightgrey', zorder=-1)
                    ax_2.plot((df_loc_cut['position_y'].iloc[overwrite_idx::]-orgin_y)*yaml_p['unit_xy'], df_loc_cut['position_z'].iloc[overwrite_idx::]*yaml_p['unit_z'], color='lightgrey', zorder=-1)

                    ax_0.scatter((yaml_p['target_test'][0]-orgin_x)*yaml_p['unit_xy'], (yaml_p['target_test'][1]-orgin_y)*yaml_p['unit_xy'], s=30, color='black', marker='x', zorder=-2)
                    ax_1.scatter((yaml_p['target_test'][0]-orgin_x)*yaml_p['unit_xy'], yaml_p['target_test'][2]*yaml_p['unit_z'], s=30, color='black', marker='x', zorder=-2)
                    ax_2.scatter((yaml_p['target_test'][1]-orgin_y)*yaml_p['unit_xy'], yaml_p['target_test'][2]*yaml_p['unit_z'], s=30, color='black', marker='x', zorder=-2)

                    drawObject = Circle(((yaml_p['target_test'][0]-orgin_x)*yaml_p['unit_xy'], (yaml_p['target_test'][1]-orgin_y)*yaml_p['unit_xy']), radius=10*yaml_p['unit_z'], fill=False, color='black', zorder=-2)
                    ax_0.add_patch(drawObject)
                    drawObject = Circle(((yaml_p['target_test'][0]-orgin_x)*yaml_p['unit_xy'], (yaml_p['target_test'][2])*yaml_p['unit_z']), radius=10*yaml_p['unit_z'], fill=False, color='black', zorder=-2)
                    ax_1.add_patch(drawObject)
                    drawObject = Circle(((yaml_p['target_test'][1]-orgin_y)*yaml_p['unit_xy'], (yaml_p['target_test'][2])*yaml_p['unit_z']), radius=10*yaml_p['unit_z'], fill=False, color='black', zorder=-2)
                    ax_2.add_patch(drawObject)

                    ax_0.set_xlabel('x [m]')
                    ax_0.set_ylabel('y [m]')
                    ax_1.set_xlabel('x [m]')
                    ax_1.set_ylabel('z [m]')
                    ax_2.set_xlabel('y [m]')
                    ax_2.set_ylabel('z [m]')
            i += 1

    ax_0.set_aspect(1)
    ax_1.set_aspect(1)
    ax_2.set_aspect(1)

    ax_1.set_ylim(0,yaml_p['size_z']*yaml_p['unit_z'])
    ax_2.set_ylim(0,yaml_p['size_z']*yaml_p['unit_z'])

    # Build folder structure if it doesn't exist yet
    path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/'

    # Get the standard angle and then rotate
    #plt.suptitle('minimum distance: ' + str(int(min_dist[closest_idx])) + ' m')
    plt.tight_layout()
    plt.savefig(path + '2dproj.png', dpi=400)
    plt.show()
    plt.close()

def plot_2d_path():
    #cmap = 'Reds_r'
    #cmap = 'YlOrBr_r'
    cmap = 'Blues_r'
    # read in logger file as pandas
    path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/'
    name_list = os.listdir(path_logger)
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list)

    fig, ax = plt.subplots(1, 1)

    d = 0
    for j in range(int(df['epi_n'].dropna().iloc[-1]) + 1):

        df_loc = df[df['epi_n'].isin([j])]
        end = np.argmin(df_loc['min_proj_dist'])

        df_loc_cut = df_loc.iloc[0:end+1]

        draw = True
        if len(df_loc) < 10:
            draw = False
        if (min(df_loc_cut['position_x']) < 0) | (max(df_loc_cut['position_x']) > yaml_p['size_x'] - 1):
            draw = False
        if (min(df_loc_cut['position_y']) < 0) | (max(df_loc_cut['position_y']) > yaml_p['size_y'] - 1):
            draw = False
        if len(df_loc['min_dist'].dropna()) == 0:
            draw = False
        if d > 9:
            draw = False

        if draw:
            dydx = np.linspace(0,1,int(yaml_p['T']/yaml_p['delta_t_logger']))  # first derivative

            #dense lines
            points = np.array([df_loc_cut['position_y']*yaml_p['unit_xy'], df_loc_cut['position_z']*yaml_p['unit_z']]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(dydx.min(), dydx.max())
            lc = LineCollection(segments, cmap=cmap, norm=norm)
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

            ax.scatter(df_loc_cut['position_y'].iloc[-1]*yaml_p['unit_xy'], df_loc_cut['position_z'].iloc[-1]*yaml_p['unit_z'], c=end, vmin=0, vmax=int(yaml_p['T']/yaml_p['delta_t_logger']), cmap=cmap)
            ax.scatter(df_loc['target_y']*yaml_p['unit_xy'],df_loc['target_z']*yaml_p['unit_z'], color='black', marker='x', zorder=1000) #zorder so the target is always above everything else

            drawObject = Circle((df_loc['target_y'].iloc[-1]*yaml_p['unit_xy'], df_loc['target_z'].iloc[-1]*yaml_p['unit_z']), radius=1*yaml_p['unit_z'], fill=False, color='black', zorder=1000)
            ax.add_patch(drawObject)

            d += 1

        #fig.suptitle(str(int(i/n_f*100)) + ' %')
        #plt.subplots_adjust(wspace=0.5, hspace=1)

    # Build folder structure if it doesn't exist yet
    ax.set_aspect(1)
    ax.set_xlim(0,(yaml_p['size_y'] - 1)*yaml_p['unit_xy'])
    ax.set_ylim(0,(yaml_p['size_z'] - 1)*yaml_p['unit_z'])

    path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/2dpath.png'
    plt.savefig(path, dpi=150)
    plt.close()

def kml_sphere(kml, center,radius,color):
    poly_sphere_x = []
    poly_sphere_y = []
    poly_sphere_z = []
    N = 100
    for i in range(N):
        deg_to_rad = np.pi/180
        m_to_lon = 1/(6.371*(10**6)*deg_to_rad)
        m_to_lat = m_to_lon*(1/np.sin(center[1]*deg_to_rad))

        poly_sphere_x.append((center[0]+radius*np.cos(i/N*2*np.pi)*m_to_lat, center[1]+radius*np.sin(i/N*2*np.pi)*m_to_lon, center[2]))
        poly_sphere_y.append((center[0]+radius*np.cos(i/N*2*np.pi)*m_to_lat, center[1], center[2]+radius*np.sin(i/N*2*np.pi)))
        poly_sphere_z.append((center[0], center[1]+radius*np.sin(i/N*2*np.pi)*m_to_lon, center[2]+radius*np.cos(i/N*2*np.pi)))

    sphere_x = kml.newpolygon(name="x-y", outerboundaryis=poly_sphere_x,
                 altitudemode="absolute")
    sphere_y = kml.newpolygon(name="x-z", outerboundaryis=poly_sphere_y,
                 altitudemode="absolute")
    sphere_z = kml.newpolygon(name="y-z", outerboundaryis=poly_sphere_z,
                 altitudemode="absolute")

    sphere_x.style.polystyle.color = color
    sphere_y.style.polystyle.color = color
    sphere_z.style.polystyle.color = color

    sphere_x.style.linestyle.color = color
    sphere_y.style.linestyle.color = color
    sphere_z.style.linestyle.color = color

def plot_kml():
    # read in logger file as pandas
    path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/'
    name_list = os.listdir(path_logger)
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list)

    center_lat = yaml_p['center_latlon'][0]
    center_lon = yaml_p['center_latlon'][1]

    for j in range(int(df['epi_n'].dropna().iloc[-1]) + 1):

        df_loc = df[df['epi_n'].isin([j])]

        #end = 700 #plot the whole trajectory
        end = -2
        df_loc = df_loc.iloc[0:end+1]

        coords = []
        for p in range(len(df_loc['position_x'])):
            pos_x = df_loc['position_x'].iloc[p]
            pos_y = df_loc['position_y'].iloc[p]
            pos_z = df_loc['position_z'].iloc[p]

            step_x = (pos_x-(yaml_p['size_x']-1)/2)*yaml_p['unit_xy']
            step_y = (pos_y-(yaml_p['size_y']-1)/2)*yaml_p['unit_xy']

            lat, lon = step(center_lat, center_lon, step_x, step_y)
            alt = pos_z*yaml_p['unit_z']
            coords.append([lon,lat,alt])

        kml = simplekml.Kml()
        lin = kml.newlinestring(name="balloon", description='flight duration: ' + str(int((yaml_p['T'] - df['t'].iloc[-1])/60)) + ' min', coords=coords, altitudemode="absolute", extrude=1)
        lin.style.linestyle.color = '00000000'
        lin.style.linestyle.width = 1

        lin2 = kml.newlinestring(name="balloon", description='flight duration: ' + str(int((yaml_p['T'] - df['t'].iloc[-1])/60)) + ' min', coords=coords, altitudemode="absolute", extrude=0)
        lin2.style.linestyle.color = 'FFFFFFFF'
        lin2.style.linestyle.width = 1

        tar_x = yaml_p['target_test'][0]
        tar_y = yaml_p['target_test'][1]
        tar_z = yaml_p['target_test'][2]
        step_x = (tar_x-(yaml_p['size_x']-1)/2)*yaml_p['unit_xy']
        step_y = (tar_y-(yaml_p['size_y']-1)/2)*yaml_p['unit_xy']
        step_z = tar_z*yaml_p['unit_z']
        lat, lon = step(center_lat, center_lon, step_x, step_y)
        center_target = [lon, lat, step_z]

        kml_sphere(kml, coords[0], 50, color='FF00FF00')
        kml_sphere(kml, center_target, 10*30.48, color='FF0000FF')

        path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/trajectory_' + str(j) + '.kml'
        kml.save(path)

def step(lat, lon, step_x, step_y):
    R = 6371*1000 #radius of earth in meters
    lat = lat + (step_y/R) * (180/np.pi)
    lon = lon + (step_x/R) * (180/np.pi) / np.cos(lat*np.pi/180)
    return lat, lon

def arrow_wind():
    res = 5
    fps = 20
    path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/'
    name_list = os.listdir(path_logger)
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list)

    path = 'temp_wind'
    Path(path).mkdir(parents=True, exist_ok=True)

    for i in range(int(len(df)/res)):
        i *= res
        pos_x = df['position_x'].iloc[i]
        pos_y = df['position_y'].iloc[i]
        pos_z = df['position_z'].iloc[i]

        t = yaml_p['T'] - df['t'].iloc[i]

        #takeoff_time = df['takeoff_time'].iloc[-1]
        takeoff_time = 12*60*60
        world = interpolate_world(t,takeoff_time)
        u,v,w = interpolate(world, [pos_x, pos_y, pos_z])

        fig, ax = plt.subplots()
        ax.arrow(0,0,u*yaml_p['unit_xy'],v*yaml_p['unit_xy'],head_width=1,color='lightsteelblue')
        #ax.arrow(0,0,df['measurement_x'].iloc[i]*yaml_p['unit_xy'],df['measurement_y'].iloc[i]*yaml_p['unit_xy'],head_width=1,color='blue')
        ax.arrow(0,0,df['velocity_x'].iloc[i]*yaml_p['unit_xy'],df['velocity_y'].iloc[i]*yaml_p['unit_xy'],head_width=1,color='midnightblue')
        ax.set_xlim(-15,15)
        ax.set_ylim(-15,15)
        ax.set_aspect(1)
        plt.savefig(path +'/' + str(i).zfill(5) + '.png', dpi=200)
        plt.close()

        print('plot_wind is ' + str(int(100*i/len(df))) + '% done')

    # Build GIF
    with imageio.get_writer(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/wind.gif', mode='I', fps=fps) as writer:
        name_list = os.listdir(path)
        name_list.sort()
        n = 0
        for name in name_list:
            image = imageio.imread(path + '/' + name)
            writer.append_data(image)
            print('generating gif: ' + str(int(100*n/len(name_list))) + ' %')
            n += 1
    shutil.rmtree(path)

def plot_wind():
    path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/'
    name_list = os.listdir(path_logger)
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list)

    #end = 1055
    #end = -1
    end = 700
    df = df.iloc[0:end]

    sim_phi = []
    sim_r = []

    real_phi = []
    real_r = []
    t = []

    for i in range(len(df)):
        pos_x = df['position_x'].iloc[i]
        pos_y = df['position_y'].iloc[i]
        pos_z = df['position_z'].iloc[i]

        t.append(yaml_p['T'] - df['t'].iloc[i])

        #takeoff_time = df['takeoff_time'].iloc[-1]
        takeoff_time = 8*24*60
        world = interpolate_world(t[-1],takeoff_time)
        u,v,w = interpolate(world, [pos_x, pos_y, pos_z])

        sim_phi.append(np.arctan2(v,u))
        sim_r.append(np.linalg.norm([v,u])*yaml_p['unit_xy'])

        real_phi.append(np.arctan2(df['velocity_y'].iloc[i],df['velocity_x'].iloc[i]))
        real_r.append(np.linalg.norm([df['velocity_x'].iloc[i],df['velocity_y'].iloc[i]])*yaml_p['unit_xy'])

    fig, axs = plt.subplots(3)
    axs[0].set_ylim(-np.pi,np.pi)
    axs[2].set_ylim(0,max(df['min_dist'])*1.2)

    axs[0].plot(t, sim_phi, color='lightgrey')
    axs[0].plot(t, real_phi, color='midnightblue')
    axs[1].plot(t, sim_r, color='lightgrey')
    axs[1].plot(t, real_r, color='midnightblue')
    axs[2].plot(t[1::], df['min_dist'].iloc[1::], color='midnightblue')

    axs[0].set_title('wind direction')
    axs[1].set_title('wind magnitude')
    axs[2].set_title('minimum distance to target')

    axs[0].set_ylabel('[rad]')
    axs[1].set_ylabel('[m/s]')
    axs[2].set_ylabel('[m]')
    axs[2].set_xlabel('[s]')

    fig.tight_layout()
    plt.savefig(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/wind.png', dpi=500)
    plt.close()

def measurement_vs_velocity():
    # read in logger file as pandas
    path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/'
    name_list = os.listdir(path_logger)
    name_list.sort()
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list)
    end = 1000
    df = df.iloc[0:end]

    fig, axs = plt.subplots(2)
    axs[0].plot(yaml_p['T'] - df['t'], df['measurement_x']*yaml_p['unit_xy'], color='darkgrey')
    axs[0].plot(yaml_p['T'] - df['t'], df['velocity_x']*yaml_p['unit_xy'], color='black')
    axs[1].plot(yaml_p['T'] - df['t'], df['measurement_y']*yaml_p['unit_xy'], color='darkgrey')
    axs[1].plot(yaml_p['T'] - df['t'], df['velocity_y']*yaml_p['unit_xy'], color='black')

    axs[0].legend(['measurement in x', 'velocity in x'])
    axs[1].legend(['measurement in y', 'velocity in y'])

    axs[0].set_ylabel('[m/s]')
    axs[1].set_ylabel('[m/s]')
    axs[1].set_xlabel('[s]')

    fig.tight_layout()
    #plt.savefig(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/learning_curve.pdf')
    plt.savefig(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/measurement_vs_velocity.png', dpi=500)
    plt.close()

def interpolate_world(t,takeoff_time):
    tss = takeoff_time + yaml_p['T'] - t #time since start

    list_of_worlds = os.listdir(yaml_p['data_path'] + 'test/tensor')
    world_name = list_of_worlds[0]
    world_name = world_name[:-5]

    h = int(tss/60/60)
    p = (tss - h*60*60)/60/60
    if yaml_p['time_dependency']:
        world_0 = torch.load(yaml_p['data_path'] + 'test/tensor/' + world_name  + str(h).zfill(2) + '.pt')
        world_1 = torch.load(yaml_p['data_path'] + 'test/tensor/' + world_name  + str(h+1).zfill(2) + '.pt')

        world = p*(world_1 - world_0) + world_0
    else:
        world = torch.load(yaml_p['data_path'] + 'test/tensor/' + world_name + str(h).zfill(2) + '.pt')

    return world

def interpolate(world, position):
    coord_x = int(np.clip(position[0],0,len(world[0,:,0,0]) - 1))
    coord_y = int(np.clip(position[1],0,len(world[0,0,:,0]) - 1))
    coord_z = int(np.clip(position[2],0,len(world[0,0,0,:]) - 1))

    x = np.clip(position[0] - coord_x,0,1)
    y = np.clip(position[1] - coord_y,0,1)
    z = np.clip(position[2] - coord_z,0,1)

    # I detect runnning out of bounds in a later stage
    i_x = 1
    i_y = 1
    i_z = 1

    if coord_x == len(world[0,:,0,0])-1:
        i_x = 0
    if coord_y == len(world[0,0,:,0])-1:
        i_y = 0
    if coord_z == len(world[0,0,0,:])-1:
        i_z = 0

    f_000 = world[-4::,coord_x,coord_y,coord_z]
    f_001 = world[-4::,coord_x,coord_y,coord_z+i_z]
    f_010 = world[-4::,coord_x,coord_y+i_y,coord_z]
    f_011 = world[-4::,coord_x,coord_y+i_y,coord_z+i_z]
    f_100 = world[-4::,coord_x+i_x,coord_y,coord_z]
    f_101 = world[-4::,coord_x+i_x,coord_y,coord_z+i_z]
    f_110 = world[-4::,coord_x+i_x,coord_y+i_y,coord_z]
    f_111 = world[-4::,coord_x+i_x,coord_y+i_y,coord_z+i_z]

    interp = f_000*(1-x)*(1-y)*(1-z) + f_001*(1-x)*(1-y)*z + f_010*(1-x)*y*(1-z) + f_011*(1-x)*y*z + f_100*x*(1-y)*(1-z) + f_101*x*(1-y)*z + f_110*x*y*(1-z) + f_111*x*y*z

    w_x, w_y, w_z = interp[0:3] #don't care about the sigma from meteo swiss
    w_x /= yaml_p['unit_xy']
    w_y /= yaml_p['unit_xy']
    w_z /= yaml_p['unit_z']

    return np.array([w_x, w_y, w_z])

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

def plot_action():
    # read in logger file as pandas
    path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/'
    name_list = os.listdir(path_logger)
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list)

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    stop = 3000
    action = df['action'].iloc[0:stop]
    rel_pos_est = df['rel_pos_est'].iloc[0:stop]
    time = (yaml_p['T'] - df['t'].iloc[0:stop]) / 60 #makes more sense to plot this in minutes
    ax.plot(time, action, color='black')
    ax.plot(time, rel_pos_est, color='lightsteelblue')

    ax.set_xlabel('minutes')
    ax.set_ylabel('relative z position')

    # Build folder structure if it doesn't exist yet
    path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/action.png'
    plt.savefig(path, dpi=150)
    plt.close()

def dist_hist(abs_path_list=None):
    if abs_path_list is None:
        abs_path_list = [yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/']

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    legend_list = []

    color=['red', 'saddlebrown', 'midnightblue']
    for p in range(len(abs_path_list)):
        # read in logger file as pandas
        name_list = os.listdir(abs_path_list[p])
        for i in range(len(name_list)):
            name_list[i] = abs_path_list[p] + name_list[i]
        df = many_logs2pandas(name_list)
        d = 0
        data = []
        for j in range(int(df['epi_n'].dropna().iloc[-1]) + 1):
            df_loc = df[df['epi_n'].isin([j])]

            end = np.argmin(df_loc['min_proj_dist'])
            df_loc_cut = df_loc.iloc[0:end+1]

            draw = True
            if len(df_loc) < 10:
                draw = False
            if (min(df_loc_cut['position_x']) < 0) | (max(df_loc_cut['position_x']) > yaml_p['size_x'] - 1):
                draw = False
            if (min(df_loc_cut['position_y']) < 0) | (max(df_loc_cut['position_y']) > yaml_p['size_y'] - 1):
                draw = False
            if len(df_loc['min_dist'].dropna()) == 0:
                draw = False
            if d > 9:
                draw = False

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
        ax.plot(x,y,color=color[p])
        names = ['Benchmark', 'RL alternative Wind Source', 'RL pre-computed wind model']
        legend_list.append(names[p] + ': median at ' + str(np.round(np.median(data),2)) + ' [m]')

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

    from scipy.interpolate import NearestNDInterpolator
    import seaborn as sns
    points = np.vstack([df['position_x'],df['position_y'],df['position_z']]).T
    interp = NearestNDInterpolator(points, df['measurement_y']*yaml_p['unit_xy'])

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

def save_csv():
    path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test/'
    name_list = os.listdir(path_logger)
    for i in range(len(name_list)):
        name_list[i] = path_logger + name_list[i]
    df = many_logs2pandas(name_list)
    df_save = pd.DataFrame()

    min_dist_overview = []
    d = 0
    for j in range(int(df['epi_n'].dropna().iloc[-1]) + 1):
        df_loc = df[df['epi_n'].isin([j])]
        min_dist = np.sqrt((yaml_p['unit_xy']*(df_loc['position_x'] - yaml_p['target_test'][0]))**2 + (yaml_p['unit_xy']*(df_loc['position_y'] - yaml_p['target_test'][1]))**2 + (yaml_p['unit_z']*(df_loc['position_z'] - yaml_p['target_test'][2]))**2)
        end = np.argmin(min_dist)
        df_loc_cut = df_loc.iloc[0:end+1]
        min_dist = min_dist[0:end+1]

        draw = True
        if len(df_loc) < 10:
            draw = False
        if (min(df_loc_cut['position_x']) < 0) | (max(df_loc_cut['position_x']) > yaml_p['size_x'] - 1):
            draw = False
        if (min(df_loc_cut['position_y']) < 0) | (max(df_loc_cut['position_y']) > yaml_p['size_y'] - 1):
            draw = False
        if len(df_loc['min_dist'].dropna()) == 0:
            draw = False
        if d > 9:
            draw = False

        if draw:
            # min dist
            #min_dist = np.sqrt((yaml_p['unit_xy']*(df_loc_cut['position_y'] - yaml_p['target_test'][1]))**2 + (yaml_p['unit_z']*(df_loc_cut['position_z'] - yaml_p['target_test'][2]))**2)
            dic = {'min_dist': min_dist}
            df_min_dist = pd.DataFrame(dic)
            df_loc_cut = df_loc_cut.drop(["min_dist"], axis=1)
            df_loc_cut = pd.concat([df_loc_cut, df_min_dist], axis=1)

            # convert into meter
            df_loc_cut["position_x"] *= yaml_p['unit_xy']
            df_loc_cut["position_y"] *= yaml_p['unit_xy']
            df_loc_cut["position_z"] *= yaml_p['unit_z']

            # idx
            df_loc_cut["epi_n"] = d
            #df_save = pd.concat([df_save, df_loc_cut.loc[:,["epi_n","t","position_x","position_y","position_z","min_dist","action"]]])
            df_save = pd.concat([df_save, df_loc_cut.loc[:,["t","position_x","position_y","position_z","min_dist","action"]]])
            d += 1

            min_dist_overview.append(min_dist.iat[-1])
    
    print("median min dist: " + str(np.median(min_dist_overview)))
    df_save = df_save.dropna(axis=1)
    df_save.to_csv(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/data.csv')

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
