import numpy as np
import os

path = 'yaml'
os.makedirs(path, exist_ok=True)

def write(process_nr, num_epochs, decay, epsi_low, min_distance):
    name = 'config_' + str(process_nr).zfill(5)

    # Write submit command
    file = open(path + '/submit.txt', "a")
    file.write('bsub -W 24:00 -R "rusage[mem=100000]" python3 setup.py ' + path + '/' + name + '.yaml' + '\n')
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

    text = text + '\n' + '# build_agent' + '\n'
    text = text + 'gamma: 0.95' + '\n'
    text = text + 'buffer_size: 10000' + '\n'
    text = text + 'lr: 0.005' + '\n'
    text = text + 'epsi_high: 0.9' + '\n'
    text = text + 'epsi_low: ' + str(epsi_low) + '\n'
    text = text + 'decay: ' + str(decay) + '\n'

    text = text + '\n' + '# build_environment' + '\n'
    text = text + 'T: 300' + '\n'
    text = text + 'start: [2,0]' + '\n'
    text = text + 'target: [20,7]' + '\n'
    text = text + 'radius: 1' + '\n'
    text = text + 'hit: 1' + '\n'
    text = text + 'step: -0.005' + '\n'
    text = text + 'action: -0.01' + '\n'
    text = text + 'overtime: 0' + '\n'
    text = text + 'min_distance: ' + str(min_distance) + '\n'
    text = text + 'bounds: -1' + '\n'

    text = text + '\n' + '# logger' + '\n'
    text = text + 'duration: 30' + '\n'
    text = text + 'fps: 20' + '\n'
    text = text + 'clear: True' + '\n'

    file.write(text)
    file.close()

process_nr = 280
for num_epochs in [50000]:
    for decay in [10000, 50000, 1000000]:
        for epsi_low in [0.1, 0.05, 0.01]:
            for min_distance in [0.85]:
                for repeat in range(2):
                    write(process_nr, num_epochs, decay, epsi_low, min_distance)
                    process_nr += 1
