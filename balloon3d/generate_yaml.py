import numpy as np
import os

path = 'yaml'
os.makedirs(path, exist_ok=True)

def write(process_nr, autoencoder, num_epochs, buffer_size, lr, explorer_type, epsi_low, decay, max_grad_norm, update_interval, minibatch_size, n_times_update, data_path, step, action, min_distance, short_sighted, qfunction):
    name = 'config_' + str(process_nr).zfill(5)

    # Write submit command
    file = open(path + '/submit.txt', "a")
    file.write('bsub -W 23:55 -R "rusage[mem=50000]" python3 setup.py ' + path + '/' + name + '.yaml' + '\n')
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
    text = text + 'size_x: 10' + '\n'
    text = text + 'size_y: 10' + '\n'
    text = text + 'size_z: 105' + '\n'
    text = text + 'unit_xy: 1100' + '\n'
    text = text + 'unit_z: 30.48' + '\n'
    text = text + 'time: 30' + '\n'

    text = text + '\n' + '# autoencoder' + '\n'
    text = text + 'autoencoder: ' + autoencoder + '\n'
    text = text + 'window_size: 3' + '\n'
    text = text + 'bottleneck: 2' + '\n'

    text = text + '\n' + '# model_train' + '\n'
    text = text + 'num_epochs: ' + str(num_epochs) + '\n'
    text = text + 'phase: 10' + '\n'

    text = text + '\n' + '# build_agent' + '\n'
    text = text + 'explorer_type: ' + str(explorer_type) + '\n'
    text = text + 'agent_type: "DoubleDQN"' + '\n'
    text = text + 'epsi_high: 0.9' + '\n'
    text = text + 'epsi_low: ' + str(epsi_low) + '\n'
    text = text + 'decay: ' + str(decay) + '\n'
    text = text + 'scale: 1' + '\n'
    text = text + 'gamma: 0.95' + '\n'
    text = text + 'buffer_size: ' + str(buffer_size) + '\n'
    text = text + 'lr: ' + f'{lr:.10f}' + '\n' #to avoid scientific notation (e.g. 1e-5)
    text = text + 'max_grad_norm: ' + str(max_grad_norm) + '\n'
    text = text + 'replay_start_size: ' + str(minibatch_size) + '\n'
    text = text + 'update_interval: ' + str(update_interval) + '\n'
    text = text + 'target_update_interval: ' + str(update_interval) + '\n'
    text = text + 'minibatch_size: ' + str(minibatch_size) + '\n'
    text = text + 'n_times_update: ' + str(n_times_update) + '\n'

    text = text + '\n' + '# build_environment' + '\n'
    text = text + 'data_path: ' + data_path + '\n'
    text = text + 'T: 600' + '\n'
    text = text + 'start: [5,5,0]' + '\n'
    text = text + 'target: [9,9,50]' + '\n'
    text = text + 'radius: 10' + '\n'
    text = text + 'hit: 1' + '\n'
    text = text + 'step: ' + f'{step:.10f}' + '\n'
    text = text + 'action: ' + f'{action:.10f}' + '\n'
    text = text + 'overtime: -1' + '\n'
    text = text + 'min_distance: ' + str(min_distance) + '\n'
    text = text + 'bounds: -1' + '\n'
    text = text + 'physics: True' + '\n'

    text = text + '\n' + '# build_character' + '\n'
    text = text + 'short_sighted: ' + str(short_sighted) + '\n'

    text = text + '\n' + '# logger' + '\n'
    text = text + "process_path: '/cluster/scratch/sjeger/'" + '\n'
    text = text + "qfunction: " + str(qfunction) + '\n'
    text = text + "log_frequency: 3" + '\n'
    text = text + 'duration: 30' + '\n'
    text = text + 'fps: 15' + '\n'
    text = text + 'overview: True' + '\n'
    text = text + 'clear: True' + '\n'

    file.write(text)
    file.close()

process_nr = 30
for data_path in ['"data/"']:
    for qfunction in [False]:
        for short_sighted in [True, False]:
            for min_distance in [0,1]:
                for autoencoder in ['"HAE"']:
                    for num_epochs in [30000]:
                        for buffer_size in [100000000]:
                            for lr in [0.0005]:
                                for explorer_type in ['"LinearDecayEpsilonGreedy"', '"Boltzmann"']:
                                    for epsi_low in [0.1]:
                                        for decay in [300000]:
                                            for max_grad_norm in [1]:
                                                for update_interval in [300]:
                                                    for minibatch_size in [100]:
                                                        for n_times_update in [100]:
                                                            for step in [-0.01, -0.001]:
                                                                for action in [-0.03, -0.003]:
                                                                    for repeat in range(3):
                                                                        write(process_nr, autoencoder, num_epochs, buffer_size, lr, explorer_type, epsi_low, decay, max_grad_norm, update_interval, minibatch_size, n_times_update, data_path, step, action, min_distance, short_sighted, qfunction)
                                                                        process_nr += 1
