import numpy as np
import os

path = 'yaml'
os.makedirs(path, exist_ok=True)

def write(process_nr, ramp, exp, lam):
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
    text = text + 'nb_steps: 30000' + '\n'

    text = text + '\n' + '# build_agent' + '\n'
    text = text + 'eps: 0.1' + '\n'
    text = text + 'target_model_update: 0.01' + '\n'

    text = text + '\n' + '# build_environment' + '\n'
    text = text + 'T: 200' + '\n'
    text = text + 'start: [15,0]' + '\n'
    text = text + 'target: [2,4]' + '\n'
    text = text + 'radius: 1' + '\n'
    text = text + 'ramp: 15' + '\n'
    text = text + 'exp: 2' + '\n'
    text = text + 'lam: 0.2' + '\n'

    file.write(text)
    file.close()

process_nr = 0
for ramp in [0,5,15]:
    for exp in [1,2,3]:
        for lam in [0.1, 0.5, 1]:
            write(process_nr, ramp, exp, lam)
            process_nr += 1
