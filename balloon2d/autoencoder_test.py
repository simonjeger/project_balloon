from build_autoencoder import VAE

import yaml
import argparse

model = VAE()
model.load_weights('autoencoder')

EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    model.model_test(epoch)
