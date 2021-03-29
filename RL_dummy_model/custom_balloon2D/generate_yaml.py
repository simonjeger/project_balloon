import numpy as np
import os

path = 'yaml'
os.makedirs(path, exist_ok=True)

def write(process_nr, num_epochs, buffer_size, lr, explorer_type, epsi_low, decay, max_grad_norm, replay_start_size, epi_update_interval, epi_target_update_interval, min_distance):
    name = 'config_' + str(process_nr).zfill(5)

    # Write submit command
    file = open(path + '/submit.txt', "a")
    file.write('bsub -W 24:00 -R "rusage[mem=50000]" python3 setup.py ' + path + '/' + name + '.yaml' + '\n')
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
    text = text + 'size_x: 30' + '\n'
    text = text + 'size_z: 10' + '\n'

    text = text + '\n' + '# autoencoder' + '\n'
    text = text + 'window_size: 3' + '\n'
    text = text + 'bottleneck: 2' + '\n'

    text = text + '\n' + '# model_train' + '\n'
    text = text + 'num_epochs: ' + str(num_epochs) + '\n'
    text = text + 'phase: 5' + '\n'

    text = text + '\n' + '# build_agent' + '\n'
    text = text + 'explorer_type: ' + str(explorer_type) + '\n'
    text = text + 'agent_type: "DoubleDQN"' + '\n'
    text = text + 'epsi_high: 0.9' + '\n'
    text = text + 'epsi_low: ' + str(epsi_low) + '\n'
    text = text + 'decay: ' + str(decay) + '\n'
    text = text + 'scale: 1' + '\n'
    text = text + 'gamma: 0.95' + '\n'
    text = text + 'buffer_size: ' + str(buffer_size) + '\n'
    text = text + 'lr: ' + str(lr) + '\n'
    text = text + 'max_grad_norm: ' + str(max_grad_norm) + '\n'
    text = text + 'replay_start_size: ' + str(replay_start_size) + '\n'
    text = text + 'epi_update_interval: ' + str(epi_update_interval) + '\n'
    text = text + 'epi_target_update_interval: ' + str(epi_target_update_interval) + '\n'

    text = text + '\n' + '# build_environment' + '\n'
    text = text + 'T: 300' + '\n'
    text = text + 'start: [2,0]' + '\n'
    text = text + 'target: [25,7]' + '\n'
    text = text + 'radius: 1' + '\n'
    text = text + 'hit: 1' + '\n'
    text = text + 'step: -0.001' + '\n'
    text = text + 'action: -0.01' + '\n'
    text = text + 'overtime: 0' + '\n'
    text = text + 'min_distance: ' + str(min_distance) + '\n'
    text = text + 'bounds: -1' + '\n'

    text = text + '\n' + '# logger' + '\n'
    text = text + 'duration: 30' + '\n'
    text = text + 'fps: 20' + '\n'
    text = text + 'overview: True' + '\n'
    text = text + 'clear: True' + '\n'

    file.write(text)
    file.close()

process_nr = 960
for num_epochs in [50000]:
    for buffer_size in [1000000]:
        for lr in [0.0001]:
            for explorer_type in ['"LinearDecayEpsilonGreedy"']:
                for epsi_low in [0.05,0.01]:
                    for decay in [100000, 50000, 150000]:
                        for max_grad_norm in [1]:
                            for replay_start_size in [1000]:
                                for epi_update_interval in [8]:
                                    for epi_target_update_interval in [1]:
                                        for min_distance in [0.9]:
                                            for repeat in range(5):
                                                write(process_nr, num_epochs, buffer_size, lr, explorer_type, epsi_low, decay, max_grad_norm, replay_start_size, epi_update_interval, epi_target_update_interval, min_distance)
                                                process_nr += 1
