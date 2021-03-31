from build_autoencoder import VAE

model = VAE()
model.load_weights('weights_autoencoder')

EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    model.model_test(epoch)
