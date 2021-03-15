from generate_wind_map import generate_wind_map
from visualize_wind_map import visualize_wind_map
from autoencoder_train import autoencoder_train

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
Path('process' + str(yaml_p['process_nr']).zfill(5)).mkdir(parents=True, exist_ok=True)
Path('process' +  str(yaml_p['process_nr']).zfill(5) + '/weights_autoencoder').mkdir(parents=True, exist_ok=True)
Path('process' +  str(yaml_p['process_nr']).zfill(5) + '/weights_model').mkdir(parents=True, exist_ok=True)
Path('data').mkdir(parents=True, exist_ok=True)
Path('data/test').mkdir(parents=True, exist_ok=True)
Path('data/test/image').mkdir(parents=True, exist_ok=True)
Path('data/test/tensor').mkdir(parents=True, exist_ok=True)
Path('data/test/tensor_comp').mkdir(parents=True, exist_ok=True)
Path('data/train').mkdir(parents=True, exist_ok=True)
Path('data/train/image').mkdir(parents=True, exist_ok=True)
Path('data/train/tensor').mkdir(parents=True, exist_ok=True)
Path('data/train/tensor_comp').mkdir(parents=True, exist_ok=True)

shutil.copy(args.yaml_file, 'process' + str(yaml_p['process_nr']).zfill(5))

size_x = yaml_p['size_x']
size_z = yaml_p['size_z']

generate_wind_map(size_x, size_z, 100, 'train')
generate_wind_map(size_x, size_z, 100, 'test')

visualize_wind_map('train')
visualize_wind_map('test')

autoencoder_train()

import model_train
