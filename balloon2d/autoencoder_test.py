import torch

from build_autoencoder import VAE

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

model = VAE()
model.to(device)
model.load_weights('autoencoder' + '/model_' + str(yaml_p['process_nr']) + '.pt')

EPOCHS = 1
for epoch in range(1, EPOCHS + 1):
    model.model_test(epoch)
