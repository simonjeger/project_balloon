import numpy as np
import os

path = 'yaml'
os.makedirs(path, exist_ok=True)

def write(process_nr, time, type, autoencoder, window_size, bottleneck, num_epochs, cherry_pick, buffer_size, lr, epsi_low, decay, replay_start_size,  minibatch_size, n_times_update, data_path, continuous, start_train, curriculum_dist, curriculum_rad, step, action, min_proj_dist, boundaries, short_sighted):
    name = 'config_' + str(process_nr).zfill(5)

    # Write submit command
    file = open(path + '/submit.txt', "a")
    #file.write('bsub -W 23:55 -R "rusage[mem=50000]" python3 setup.py ' + path + '/' + name + '.yaml' + '\n')
    file.write('bsub -n 2 -W 24:00 -R "rusage[mem=8000, ngpus_excl_p=1]" python3 setup.py ' + path + '/' + name + '.yaml' + '\n')
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
    text = text + 'window_size: ' + str(window_size) + '\n'
    text = text + 'bottleneck: '+ str(bottleneck) + '\n'

    text = text + '\n' + '# model_train' + '\n'
    text = text + 'num_epochs_train: ' + str(num_epochs) + '\n'
    text = text + 'num_epochs_test: 1000' + '\n'
    text = text + 'phase: 50' + '\n'
    text = text + 'cherry_pick: ' + str(cherry_pick) + '\n'

    text = text + '\n' + '# build_agent' + '\n'
    text = text + 'explorer_type: LinearDecayEpsilonGreedy' + '\n'
    text = text + 'agent_type: SoftActorCritic' + '\n'
    text = text + 'epsi_high: 0.9' + '\n'
    text = text + 'epsi_low: ' + str(epsi_low) + '\n'
    text = text + 'decay: ' + str(decay) + '\n'
    text = text + 'gamma: 0.95' + '\n'
    text = text + 'buffer_size: ' + str(buffer_size) + '\n'
    text = text + 'lr: ' + f'{lr:.10f}' + '\n' #to avoid scientific notation (e.g. 1e-5)
    text = text + 'max_grad_norm: 1' + '\n'
    text = text + 'replay_start_size: ' + str(replay_start_size) + '\n'
    text = text + 'update_interval: 20' + '\n'
    text = text + 'target_update_interval: 20' + '\n'
    text = text + 'minibatch_size: ' + str(minibatch_size) + '\n'
    text = text + 'n_times_update: ' + str(n_times_update) + '\n'

    text = text + '\n' + '# build_environment' + '\n'
    text = text + 'data_path: ' + data_path + '\n'
    text = text + 'continuous: ' + str(continuous) + '\n'
    text = text + 'T: 100' + '\n'
    text = text + 'start_train: ' + str(start_train) + '\n'
    text = text + 'start_test: [7,6,0]' + '\n'
    text = text + 'target_train: "random"' + '\n'
    text = text + 'target_test: "random"' + '\n'
    text = text + 'radius_start_xy: 15' + '\n'
    text = text + 'radius_stop_xy: 10' + '\n'
    text = text + 'radius_start_ratio: 2' + '\n'
    text = text + 'curriculum_dist: ' + str(curriculum_dist) + '\n'
    text = text + 'curriculum_rad: ' + str(curriculum_rad) + '\n'
    text = text + 'hit: 1' + '\n'
    text = text + 'step: ' + f'{step:.10f}' + '\n'
    text = text + 'action: ' + f'{action:.10f}' + '\n'
    text = text + 'overtime: -1' + '\n'
    text = text + 'min_proj_dist: ' + str(min_proj_dist) + '\n'
    text = text + 'bounds: -1' + '\n'
    text = text + 'physics: True' + '\n'

    text = text + '\n' + '# build_character' + '\n'
    text = text + 'boundaries: ' + str(boundaries) + '\n'
    text = text + 'short_sighted: ' + str(short_sighted) + '\n'

    text = text + '\n' + '# logger' + '\n'
    text = text + "process_path: '/cluster/scratch/sjeger/'" + '\n'
    text = text + "qfunction: False" + '\n'
    text = text + "log_frequency: 3" + '\n'
    text = text + 'duration: 30' + '\n'
    text = text + 'fps: 15' + '\n'
    text = text + 'overview: True' + '\n'
    text = text + 'clear: True' + '\n'

    file.write(text)
    file.close()

time = 360
step = -0.00003
action = -0.05
start_train = [7,6,0]

process_nr = 1490
for data_path in ['"data/"', '"data_small/"', '"data_constant/"']:
    for type in ['"regular"', '"squished"']:
        for boundaries in ['"short"', '"long"']:
            if type == '"regular"':
                num_epochs = 20000
            if type == '"squished"':
                num_epochs = 10000
            for min_proj_dist in [1]:
                for autoencoder in ['"HAE_avg"']:
                    for cherry_pick in [0]:
                        for short_sighted in [False]:
                            for window_size in [1]:
                                for bottleneck in [1,4]:
                                    for buffer_size in [100000000]:
                                        for curriculum_dist in [1,10000]:
                                            for curriculum_rad in [1]:
                                                for epsi_low in [0.1]:
                                                    for decay in [300000]:
                                                        for minibatch_size in [800]:
                                                            for n_times_update in [100]:
                                                                for lr in [0.0005]:
                                                                    for repeat in range(2):
                                                                        for replay_start_size in [1000]:
                                                                            continuous = True

                                                                            write(process_nr, time, type, autoencoder, window_size, bottleneck, num_epochs, cherry_pick, buffer_size, lr, epsi_low, decay, replay_start_size, minibatch_size, n_times_update, data_path, continuous, start_train, curriculum_dist, curriculum_rad, step, action, min_proj_dist, boundaries, short_sighted)
                                                                            process_nr += 1
