import numpy as np
import os

path = 'yaml'
os.makedirs(path, exist_ok=True)

def write(process_nr, time, type, autoencoder, vae_nr, window_size, bottleneck, time_train, cherry_pick, agent_type, width, depth, lr, temperature_optimizer_lr, replay_start_size, data_path, continuous, start_train, curriculum_dist, curriculum_rad, curriculum_rad_dry, step, action, min_proj_dist, balloon, wind_info, measurement_info):
    name = 'config_' + str(process_nr).zfill(5)

    # Write submit command
    file = open(path + '/submit.txt', "a")
    #file.write('bsub -W 23:55 -R "rusage[mem=50000]" python3 setup.py ' + path + '/' + name + '.yaml' + '\n')
    file.write('bsub -n 2 -W 23:55 -R "rusage[mem=8000, ngpus_excl_p=1]" python3 setup.py ' + path + '/' + name + '.yaml' + '\n')
    #file.write('bsub -n 2 -W 23:55 -R "rusage[mem=8000, ngpus_excl_p=1]" python3 agent_test.py ' + path + '/' + name + '.yaml' + '\n')
    file.close()

    # Clear file
    file = open(path + '/' + name + '.yaml', "w")
    file.close()
    os.remove(path + '/' + name + '.yaml')

    # Write file
    file = open(path + '/' + name + '.yaml', "w")
    text = ''

    text = text + '# general' + '\n'
    text = text + 'process_nr: ' + str(process_nr) + '\n'

    text = text + '\n' + '# setup' + '\n'
    text = text + 'size_x: 14' + '\n'
    text = text + 'size_y: 12' + '\n'
    text = text + 'size_z: 105' + '\n'
    text = text + 'unit_xy: 1100' + '\n'
    text = text + 'unit_z: 30.48' + '\n'
    text = text + 'time: ' + str(time) + '\n'
    text = text + 'type: ' + str(type) + '\n'

    text = text + '\n' + '# autoencoder' + '\n'
    text = text + 'autoencoder: ' + autoencoder + '\n'
    text = text + 'vae_nr: ' + str(vae_nr) + '\n'
    text = text + 'window_size: ' + str(window_size) + '\n'
    text = text + 'bottleneck: '+ str(bottleneck) + '\n'

    text = text + '\n' + '# model_train' + '\n'
    text = text + 'time_train: ' + str(time_train) + '\n'
    text = text + 'num_epochs_test: 1000' + '\n'
    text = text + 'phase: 1' + '\n'
    text = text + 'cherry_pick: ' + str(cherry_pick) + '\n'

    text = text + '\n' + '# build_agent' + '\n'
    text = text + 'rl: True' + '\n'
    text = text + 'explorer_type: LinearDecayEpsilonGreedy' + '\n'
    text = text + 'agent_type: ' + agent_type + '\n'
    text = text + 'width: ' + str(width) + '\n'
    text = text + 'depth: ' + str(depth) + '\n'
    text = text + 'epsi_high: 0.9' + '\n'
    text = text + 'epsi_low: 0.1' + '\n'
    text = text + 'decay: 300000' + '\n'
    text = text + 'gamma: 0.95' + '\n'
    text = text + 'buffer_size: 100000000' + '\n'
    text = text + 'lr: ' + f'{lr:.10f}' + '\n' #to avoid scientific notation (e.g. 1e-5)
    text = text + 'lr_scheduler: 999999999999' + '\n'
    text = text + 'temperature_optimizer_lr: ' + f'{temperature_optimizer_lr:.10f}' + '\n'
    text = text + 'max_grad_norm: 1' + '\n'
    text = text + 'replay_start_size: ' + str(replay_start_size) + '\n'
    text = text + 'update_interval: 20' + '\n'
    text = text + 'target_update_interval: 20' + '\n'
    text = text + 'minibatch_size: 800' + '\n'
    text = text + 'n_times_update: 100' + '\n'

    text = text + '\n' + '# build_environment' + '\n'
    text = text + 'data_path: ' + data_path + '\n'
    text = text + 'continuous: ' + str(continuous) + '\n'
    text = text + 'T: 100' + '\n'
    text = text + 'start_train: ' + str(start_train) + '\n'
    text = text + 'start_test: [7,6,0]' + '\n'
    text = text + 'target_train: "random"' + '\n'
    text = text + 'target_test: "random"' + '\n'
    text = text + 'radius_start_xy: 15' + '\n'
    text = text + 'radius_stop_xy: 15' + '\n'
    text = text + 'radius_start_ratio: 1' + '\n'
    text = text + 'curriculum_dist: ' + str(curriculum_dist) + '\n'
    text = text + 'curriculum_rad: ' + str(curriculum_rad) + '\n'
    text = text + 'curriculum_rad_dry: ' + str(curriculum_rad_dry) + '\n'
    text = text + 'min_space: 0.5' + '\n'
    text = text + 'hit: 1' + '\n'
    text = text + 'step: ' + f'{step:.10f}' + '\n'
    text = text + 'action: ' + f'{action:.10f}' + '\n'
    text = text + 'overtime: -1' + '\n'
    text = text + 'min_proj_dist: ' + str(min_proj_dist) + '\n'
    text = text + 'bounds: -1' + '\n'

    text = text + '\n' + '# build_character' + '\n'
    text = text + 'balloon: ' + balloon + '\n'
    text = text + 'measurement_info: ' + str(measurement_info) + '\n'
    text = text + 'wind_info: ' + str(wind_info) + '\n'
    text = text + 'short_sighted: False' + '\n'

    text = text + '\n' + '# logger' + '\n'
    text = text + "process_path: '/cluster/scratch/sjeger/'" + '\n'
    text = text + "reuse_weights: True" + '\n'
    text = text + "log_frequency: 3" + '\n'
    text = text + 'duration: 30' + '\n'
    text = text + 'fps: 15' + '\n'
    text = text + 'overview: True' + '\n'
    text = text + 'clear: True' + '\n'
    text = text + 'render: False' + '\n'

    file.write(text)
    file.close()

balloon = '"small"'
time_train = 20*60*60
step = -0.00003
action = -0.005
wind_info = True
measurement_info = True

process_nr = 3210

for data_path in ['"data_big/"']:
    for min_proj_dist in [1]:
        for cherry_pick in [0]:
            for agent_type in ['"SoftActorCritic"']:
                if agent_type == '"SoftActorCritic"':
                    time = 230
                    type = '"squished"'
                    continuous = True
                elif agent_type == '"DoubleDQN"':
                    time = 100
                    type = '"regular"'
                    continuous = False
                for autoencoder in ['"HAE_avg"']:
                    for width in [512]:
                        for depth in [2]:
                            for window_size in [4]:
                                for bottleneck in [2]:
                                    if autoencoder == '"HAE_avg"':
                                        vae_nr = 11111
                                    elif autoencoder == '"HAE_ext"':
                                        vae_nr = 11111
                                    elif autoencoder == '"VAE"':
                                        if window_size == 0:
                                            if bottleneck == 4:
                                                vae_nr = 11102
                                            elif bottleneck == 10:
                                                vae_nr = 11103
                                            elif bottleneck == 15:
                                                vae_nr = 11104
                                            elif bottleneck == 30:
                                                vae_nr = 11105
                                        if window_size == 1:
                                            if bottleneck == 4:
                                                vae_nr = 11112
                                            elif bottleneck == 10:
                                                vae_nr = 11113
                                            elif bottleneck == 15:
                                                vae_nr = 11114
                                            elif bottleneck == 30:
                                                vae_nr = 11115
                                        elif window_size == 2:
                                            if bottleneck == 4:
                                                vae_nr = 11122
                                            elif bottleneck == 10:
                                                vae_nr = 11123
                                            elif bottleneck == 15:
                                                vae_nr = 11124
                                            elif bottleneck == 30:
                                                vae_nr = 11125
                                        elif window_size == 3:
                                            if bottleneck == 4:
                                                vae_nr = 11132
                                            elif bottleneck == 10:
                                                vae_nr = 11133
                                            elif bottleneck == 15:
                                                vae_nr = 11134
                                            elif bottleneck == 30:
                                                vae_nr = 11135
                                        elif window_size == 4:
                                            if bottleneck == 4:
                                                vae_nr = 11142
                                            elif bottleneck == 10:
                                                vae_nr = 11143
                                            elif bottleneck == 15:
                                                vae_nr = 11144
                                            elif bottleneck == 30:
                                                vae_nr = 11145
                                        elif window_size == 5:
                                            if bottleneck == 4:
                                                vae_nr = 11152
                                            elif bottleneck == 10:
                                                vae_nr = 11153
                                            elif bottleneck == 15:
                                                vae_nr = 11154
                                            elif bottleneck == 30:
                                                vae_nr = 11155
                                    for start_train in [[7,6,0]]:
                                        for curriculum_dist in [1,10000,50000,100000,500000,1000000]:
                                            for curriculum_rad in [1,2,3]:
                                                for curriculum_rad_dry in [1000]:
                                                    for lr in [0.006]:
                                                        for temperature_optimizer_lr in [0.00003]:
                                                            for repeat in range(3):
                                                                for replay_start_size in [1000]:
                                                                    write(process_nr, time, type, autoencoder, vae_nr, window_size, bottleneck, time_train, cherry_pick, agent_type, width, depth, lr, temperature_optimizer_lr, replay_start_size, data_path, continuous, start_train, curriculum_dist, curriculum_rad, curriculum_rad_dry, step, action, min_proj_dist, balloon, wind_info, measurement_info)
                                                                    process_nr += 1
