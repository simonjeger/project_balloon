import numpy as np
import os

path = 'yaml'
os.makedirs(path, exist_ok=True)

def write(process_nr, type, autoencoder, num_epochs, buffer_size, lr, explorer_type, epsi_low, max_grad_norm, replay_start_size, update_interval, minibatch_size, n_times_update, data_path, continuous, start_train, curriculum_dist, curriculum_rad, step, action, min_proj_dist, balloon, short_sighted, qfunction):
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
    text = text + 'size_z: 105' + '\n'
    text = text + 'unit_xy: 1100' + '\n'
    text = text + 'unit_z: 30.48' + '\n'
    text = text + 'time: 360' + '\n'
    text = text + 'type: ' + type + '\n'

    text = text + '\n' + '# autoencoder' + '\n'
    text = text + 'autoencoder: ' + autoencoder + '\n'
    text = text + 'window_size: 3' + '\n'
    text = text + 'bottleneck: 4' + '\n'

    text = text + '\n' + '# model_train' + '\n'
    text = text + 'num_epochs_train: ' + str(num_epochs) + '\n'
    text = text + 'num_epochs_test: 1000' + '\n'
    text = text + 'phase: 5' + '\n'
    text = text + 'cherry_pick: 0' + '\n'

    text = text + '\n' + '# build_agent' + '\n'
    text = text + 'explorer_type: ' + str(explorer_type) + '\n'
    text = text + 'agent_type: "SoftActorCritic"' + '\n'
    text = text + 'epsi_high: 0.9' + '\n'
    text = text + 'epsi_low: ' + str(epsi_low) + '\n'
    text = text + 'decay: 150000' + '\n'
    text = text + 'gamma: 0.95' + '\n'
    text = text + 'buffer_size: ' + str(buffer_size) + '\n'
    text = text + 'lr: ' + f'{lr:.10f}' + '\n' #to avoid scientific notation (e.g. 1e-5)
    text = text + 'lr_actor: 0.0003' + '\n'
    text = text + 'lr_critic: 0.0003' + '\n'
    text = text + 'max_grad_norm: ' + str(max_grad_norm) + '\n'
    text = text + 'replay_start_size: ' + str(replay_start_size) + '\n'
    text = text + 'update_interval: ' + str(update_interval) + '\n'
    text = text + 'target_update_interval: ' + str(update_interval) + '\n'
    text = text + 'minibatch_size: ' + str(minibatch_size) + '\n'
    text = text + 'n_times_update: ' + str(n_times_update) + '\n'

    text = text + '\n' + '# build_environment' + '\n'
    text = text + 'data_path: ' + data_path + '\n'
    text = text + 'continuous: ' + str(continuous) + '\n'
    text = text + 'T: 100' + '\n'
    text = text + 'start_train: ' + str(start_train) + '\n'
    text = text + 'start_test: [7,0]' + '\n'
    text = text + 'target_train: "random"' + '\n'
    text = text + 'target_test: "random"' + '\n'
    text = text + 'radius_start_x: 10' + '\n'
    text = text + 'radius_stop_x: 2' + '\n'
    text = text + 'radius_start_ratio: 1' + '\n'
    text = text + 'curriculum_dist: ' + str(curriculum_dist) + '\n'
    text = text + 'curriculum_rad: ' + str(curriculum_rad) + '\n'
    text = text + 'hit: 1' + '\n'
    text = text + 'step: ' + f'{step:.10f}' + '\n'
    text = text + 'action: ' + f'{action:.10f}' + '\n'
    text = text + 'overtime: -1' + '\n'
    text = text + 'min_proj_dist: ' + str(min_proj_dist) + '\n'
    text = text + 'bounds: -1' + '\n'

    text = text + '\n' + '# build_character' + '\n'
    text = text + 'memory_size: 3' + '\n'
    text = text + 'memory_interval: 5' + '\n'
    text = text + 'balloon: ' + balloon + '\n'
    text = text + 'boundaries: "short"' + '\n'
    text = text + 'short_sighted: ' + str(short_sighted) + '\n'

    text = text + '\n' + '# logger' + '\n'
    text = text + "process_path: '/cluster/scratch/sjeger/'" + '\n'
    text = text + "reuse_weights: False" + '\n'
    text = text + "qfunction: " + str(qfunction) + '\n'
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
balloon = '"small"'

process_nr = 4000

for data_path in ['"data/"']:
    for start_train in [[7,0],'"random"']:
        for type in ['"regular"', '"squished"']:
            if type == '"regular"':
                num_epochs = 15000
            if type == '"squished"':
                num_epochs = 8000
            for qfunction in [False]:
                for short_sighted in [False]:
                    for min_proj_dist in [0]:
                        for autoencoder in ['"HAE_avg"']:
                            for buffer_size in [100000000]:
                                for lr in [0.0005]:
                                    for explorer_type in ['"LinearDecayEpsilonGreedy"']:
                                        for curriculum_dist in [1,40000]:
                                            for curriculum_rad in [1,2,3]:
                                                for epsi_low in [0.1]:
                                                    for max_grad_norm in [1]:
                                                        for update_interval in [300]:
                                                            for minibatch_size in [800]:
                                                                for n_times_update in [100]:
                                                                    for repeat in range(2):
                                                                        continuous = True
                                                                        replay_start_size = 10000

                                                                        write(process_nr, type, autoencoder, num_epochs, buffer_size, lr, explorer_type, epsi_low, max_grad_norm, replay_start_size, update_interval, minibatch_size, n_times_update, data_path, continuous, start_train, curriculum_dist, curriculum_rad, step, action, min_proj_dist, balloon, short_sighted, qfunction)
                                                                        process_nr += 1
