import numpy as np
import os

path = 'yaml'
os.makedirs(path, exist_ok=True)

def write(process_nr, delta_t, delta_t_physics, autoencoder, window_size, bottleneck, time_train, HER, width_depth, lr, temperature_optimizer_lr, replay_start_size, update_interval, minibatch_size, data_path, radius_xy, curriculum_dist, curriculum_rad, curriculum_rad_dry, step, action, gradient, min_proj_dist, balloon, W_20, wind_info, measurement_info):
    name = 'config_' + str(process_nr).zfill(5)

    # Write submit command
    file = open(path + '/submit.txt', "a")
    #file.write('bsub -W 23:55 -R "rusage[mem=50000]" python3 setup.py ' + path + '/' + name + '.yaml' + '\n')
    file.write('bsub -n 2 -W 23:55 -R "rusage[mem=15000, ngpus_excl_p=1]" python3 setup.py ' + path + '/' + name + '.yaml' + '\n')
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
    text = text + 'size_x: 10' + '\n'
    text = text + 'size_y: 10' + '\n'
    text = text + 'size_z: 105' + '\n'
    text = text + 'unit_xy: 1100' + '\n'
    text = text + 'unit_z: 30.48' + '\n'
    text = text + 'delta_t: ' + str(delta_t) + '\n'
    text = text + 'delta_t_physics: ' + str(delta_t_physics) + '\n'

    text = text + '\n' + '# autoencoder' + '\n'
    text = text + 'autoencoder: ' + autoencoder + '\n'
    text = text + 'vae_nr: 11111' + '\n'
    text = text + 'window_size: ' + str(window_size) + '\n'
    text = text + 'bottleneck: '+ str(bottleneck) + '\n'

    text = text + '\n' + '# model_train' + '\n'
    text = text + 'time_train: ' + str(time_train) + '\n'
    text = text + 'num_epochs_test: 300' + '\n'

    text = text + '\n' + '# build_agent' + '\n'
    text = text + 'alt_resample: 1200' + '\n'
    text = text + 'mode: reinforcement_learning' + '\n'
    text = text + 'HER: ' + str(HER) + '\n'
    text = text + 'width: ' + str(width_depth[0]) + '\n'
    text = text + 'depth: ' + str(width_depth[1]) + '\n'
    text = text + 'gamma: 0.95' + '\n'
    text = text + 'buffer_size: 100000000' + '\n'
    text = text + 'lr: ' + f'{lr:.10f}' + '\n' #to avoid scientific notation (e.g. 1e-5)
    text = text + 'lr_scheduler: 999999999999' + '\n'
    text = text + 'temperature_optimizer_lr: ' + f'{temperature_optimizer_lr:.10f}' + '\n'
    text = text + 'replay_start_size: ' + str(replay_start_size) + '\n'
    text = text + 'update_interval: ' + str(update_interval) + '\n'
    text = text + 'minibatch_size: ' + str(minibatch_size) + '\n'

    text = text + '\n' + '# build_environment' + '\n'
    text = text + 'environment: python3' + '\n'
    text = text + 'data_path: ' + data_path + '\n'
    text = text + 'time_dependency: True' + '\n'
    text = text + 'T: 10000' + '\n'
    text = text + 'start_train: "center"' + '\n'
    text = text + 'start_test: "center"' + '\n'
    text = text + 'target_train: "random"' + '\n'
    text = text + 'target_test: "random"' + '\n'
    text = text + 'reachability_study: 0' + '\n'
    text = text + 'set_reachable_target: True' + '\n'
    text = text + 'radius_xy: ' + str(radius_xy) + '\n'
    text = text + 'radius_z: ' + str(radius_xy) + '\n'
    text = text + 'min_space: 0.5' + '\n'
    text = text + 'hit: 1' + '\n'
    text = text + 'step: ' + f'{step:.10f}' + '\n'
    text = text + 'action: ' + f'{action:.10f}' + '\n'
    text = text + 'overtime: -1' + '\n'
    text = text + 'min_proj_dist: ' + str(min_proj_dist) + '\n'
    text = text + 'gradient: ' + str(gradient) + '\n'
    text = text + 'bounds: -1' + '\n'

    text = text + '\n' + '# build_character' + '\n'
    text = text + 'balloon: ' + balloon + '\n'
    text = text + 'ceiling_width: 0.1' + '\n'
    text = text + 'noise_path: "/cluster/scratch/sjeger/noise_14x12/"' + '\n'
    text = text + 'W_20: 0' + '\n'
    text = text + 'measurement_info: ' + str(measurement_info) + '\n'
    text = text + 'wind_info: ' + str(wind_info) + '\n'

    text = text + '\n' + '# logger' + '\n'
    text = text + "process_path: '/cluster/scratch/sjeger/'" + '\n'
    text = text + "reuse_weights: True" + '\n'
    text = text + "log_frequency: 3" + '\n'
    text = text + 'duration: 30' + '\n'
    text = text + 'fps: 15' + '\n'
    text = text + 'overview: True' + '\n'
    text = text + 'render: False' + '\n'

    text = text + '\n' + '# temp' + '\n'
    text = text + 'delta_f: 1' + '\n'

    file.write(text)
    file.close()

balloon = '"outdoor_balloon"'
delta_t = 230
delta_t_physics = 20
time_train = 20*60*60
step = -0.00003
action = -0.005
measurement_info = True

process_nr = 6500

for data_path in ["/cluster/scratch/sjeger/data_single/", "/cluster/scratch/sjeger/data_single/", "/cluster/scratch/sjeger/data_10x10/", "/cluster/scratch/sjeger/data_20x20/"]:
    for radius_xy in [10]:
        for HER in [False]:
            for lr in [0.006]:
                for width_depth in [[20,4], [100,4], [500,4], [500,3], [1000,3], [1000,2], [5000,2]]:
                    for min_proj_dist in [1]:
                        for autoencoder in ['"HAE_bidir"']:
                            for window_size in [1]:
                                for bottleneck in [8]:
                                    for W_20 in [0]:
                                        for wind_info in [True]:
                                            #for gradient in np.array([0.1, 1, 10])*abs(step + action):
                                            for gradient in [0]:
                                                for curriculum_dist in [1]:
                                                    for curriculum_rad in [1]:
                                                        for curriculum_rad_dry in [1000]:
                                                            for temperature_optimizer_lr in [0.00003]:
                                                                for replay_start_size in [100000]:
                                                                    for update_interval in [1]:
                                                                        for minibatch_size in [1000]:
                                                                            for repeat in range(3):
                                                                                write(process_nr, delta_t, delta_t_physics, autoencoder, window_size, bottleneck, time_train, HER, width_depth, lr, temperature_optimizer_lr, replay_start_size, update_interval, minibatch_size, data_path, radius_xy, curriculum_dist, curriculum_rad, curriculum_rad_dry, step, action, gradient, min_proj_dist, balloon, W_20, wind_info, measurement_info)
                                                                                process_nr += 1
