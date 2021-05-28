from torch import nn, optim
from pathlib import Path

from build_autoencoder import VAE

from torch.utils.tensorboard import SummaryWriter

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

Path('autoencoder').mkdir(parents=True, exist_ok=True)
Path('autoencoder/results').mkdir(parents=True, exist_ok=True)

# initialize logger
writer = SummaryWriter('autoencoder/logger_' + str(yaml_p['process_nr']))

model = VAE(writer)

EPOCHS = 10000
for epoch in range(1, EPOCHS + 1):
    model.model_train(epoch)
    model.save_weights('autoencoder')
