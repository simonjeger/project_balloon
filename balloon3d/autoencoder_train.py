from torch import nn, optim

from build_autoencoder import VAE

from torch.utils.tensorboard import SummaryWriter

# initialize logger
writer = SummaryWriter('autoencoder')

model = VAE(writer)

EPOCHS = 40
for epoch in range(1, EPOCHS + 1):
    model.model_train(epoch)
model.save_weights('autoencoder')
