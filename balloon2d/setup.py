import matplotlib
matplotlib.use('Agg') # this needs to be called at the very beginning on cluster server

from generate_world import generate_world
from visualize_world import visualize_world

from pathlib import Path
import shutil

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

# Build folder structure if it doesn't exist yet
Path(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5)).mkdir(parents=True, exist_ok=True)
Path(yaml_p['process_path'] + 'process' +  str(yaml_p['process_nr']).zfill(5) + '/weights_autoencoder').mkdir(parents=True, exist_ok=True)
Path(yaml_p['process_path'] + 'process' +  str(yaml_p['process_nr']).zfill(5) + '/weights_agent').mkdir(parents=True, exist_ok=True)
Path(yaml_p['data_path']).mkdir(parents=True, exist_ok=True)
Path(yaml_p['data_path'] + 'test').mkdir(parents=True, exist_ok=True)
Path(yaml_p['data_path'] + 'test/image').mkdir(parents=True, exist_ok=True)
Path(yaml_p['data_path'] + 'test/tensor').mkdir(parents=True, exist_ok=True)
Path(yaml_p['data_path'] + 'test/tensor_comp').mkdir(parents=True, exist_ok=True)
Path(yaml_p['data_path'] + 'train').mkdir(parents=True, exist_ok=True)
Path(yaml_p['data_path'] + 'train/image').mkdir(parents=True, exist_ok=True)
Path(yaml_p['data_path'] + 'train/tensor').mkdir(parents=True, exist_ok=True)

shutil.copy(args.yaml_file, yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5))

size_x = yaml_p['size_x']
size_z = yaml_p['size_z']

#generate_world(size_x, size_z, 50, 'train')
#generate_world(size_x, size_z, 15, 'test')

#import autoencoder_train
#import autoencoder_test

#visualize_world('train')
#visualize_world('test')

import agent_train
