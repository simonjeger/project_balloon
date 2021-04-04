import matplotlib
matplotlib.use('Agg') # this needs to be called at the very beginning on cluster server

from generate_wind_map import generate_wind_map
from visualize_wind_map import visualize_wind_map

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
Path(yaml_p['path'] + 'process' + str(yaml_p['process_nr']).zfill(5)).mkdir(parents=True, exist_ok=True)
Path(yaml_p['path'] + 'process' +  str(yaml_p['process_nr']).zfill(5) + '/weights_autoencoder').mkdir(parents=True, exist_ok=True)
Path(yaml_p['path'] + 'process' +  str(yaml_p['process_nr']).zfill(5) + '/weights_agent').mkdir(parents=True, exist_ok=True)
Path(yaml_p['path'] + 'data').mkdir(parents=True, exist_ok=True)
Path(yaml_p['path'] + 'data/test').mkdir(parents=True, exist_ok=True)
Path(yaml_p['path'] + 'data/test/image').mkdir(parents=True, exist_ok=True)
Path(yaml_p['path'] + 'data/test/tensor').mkdir(parents=True, exist_ok=True)
Path(yaml_p['path'] + 'data/test/tensor_comp').mkdir(parents=True, exist_ok=True)
Path(yaml_p['path'] + 'data/train').mkdir(parents=True, exist_ok=True)
Path(yaml_p['path'] + 'data/train/image').mkdir(parents=True, exist_ok=True)
Path(yaml_p['path'] + 'data/train/tensor').mkdir(parents=True, exist_ok=True)

shutil.copy(args.yaml_file, yaml_p['path'] + 'process' + str(yaml_p['process_nr']).zfill(5))

size_x = yaml_p['size_x']
size_z = yaml_p['size_z']

#generate_wind_map(size_x, size_z, 1000, 'train')
#generate_wind_map(size_x, size_z, 100, 'test')

#import autoencoder_train
#import autoencoder_test

#visualize_wind_map('train')
#visualize_wind_map('test')

import agent_train