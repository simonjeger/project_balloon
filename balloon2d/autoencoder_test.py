from build_autoencoder import VAE

import yaml
import argparse

model = VAE()
model.load_weights('autoencoder' + '/model_' + str(yaml_p['process_nr']) + '.pt')

EPOCHS = 1
for epoch in range(1, EPOCHS + 1):
    model.model_test(epoch)
