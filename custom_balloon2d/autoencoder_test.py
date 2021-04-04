from build_autoencoder import VAE

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

model = VAE()
model.load_weights(yaml_p['path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/weights_autoencoder')

EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    model.model_test(epoch)
