from torch import nn, optim

from build_autoencoder import VAE

model = VAE()

EPOCHS = 100
for epoch in range(1, EPOCHS + 1):
    model.model_train(epoch)
model.save_weights('weights_autoencoder')
