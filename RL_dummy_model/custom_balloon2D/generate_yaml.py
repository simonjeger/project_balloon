import numpy as np
import os

path = 'yaml'
os.makedirs(path, exist_ok=True)

def write(process_nr, decay, update_target_step, lr):
    name = 'config_' + str(process_nr).zfill(5)

    # Write submit command
    file = open(path + '/submit.txt', "a")
    file.write('python3 setup.py ' + path + '/' + name + '.yaml' + '\n')
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
    text = text + 'bottleneck: 10' + '\n'

    text = text + '\n' + '# model_train' + '\n'
    text = text + 'num_epochs: 800' + '\n'

    text = text + '\n' + '# build_agent' + '\n'
    text = text + 'gamma: 0.95' + '\n'
    text = text + 'buffer_size: 10000' + '\n'
    text = text + 'lr: ' + str(lr) + '\n'
    text = text + 'epsi_high: 0.9' + '\n'
    text = text + 'epsi_low: 0.05' + '\n'
    text = text + 'decay: ' + str(decay) + '\n'
    text = text + 'update_target_step: ' + str(update_target_step) + '\n'

    text = text + '\n' + '# build_environment' + '\n'
    text = text + 'T: 200' + '\n'
    text = text + 'start: [2,0]' + '\n'
    text = text + 'target: "random"' + '\n'
    text = text + 'radius: 1' + '\n'
    text = text + 'ramp: 2.5' + '\n'
    text = text + 'exp: 1' + '\n'
    text = text + 'lam: -0.1' + '\n'
    text = text + 'overtime: -1' + '\n'

    file.write(text)
    file.close()

process_nr = 170
for decay in [300, 400, 500]:
    for update_target_step in [200, 300, 400]:
        for lr in [0.005]:
            for repeat in range(2):
                write(process_nr, decay, update_target_step, lr)
                process_nr += 1
