import matplotlib
matplotlib.use('Agg') # this needs to be called at the very beginning on cluster server

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
Path(yaml_p['process_path'] + 'process' +  str(yaml_p['process_nr']).zfill(5) + '/weights_agent').mkdir(parents=True, exist_ok=True)
Path(yaml_p['process_path'] + 'process' +  str(yaml_p['process_nr']).zfill(5) + '/reachability_study').mkdir(parents=True, exist_ok=True)
Path(yaml_p['process_path'] + 'process' +  str(yaml_p['process_nr']).zfill(5) + '/map_test').mkdir(parents=True, exist_ok=True)
Path(yaml_p['process_path'] + 'process' +  str(yaml_p['process_nr']).zfill(5) + '/render').mkdir(parents=True, exist_ok=True)

shutil.copy(args.yaml_file, yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5))

size_x = yaml_p['size_x']
size_y = yaml_p['size_y']
size_z = yaml_p['size_z']

#import autoencoder_train
#import autoencoder_test

import agent_train
import agent_test
#import agent_importance
