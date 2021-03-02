from build_autoencoder import Autoencoder
from build_autoencoder import load_tensor

import numpy as np

def autoencoder_test(train_or_test):
    x_test_data, x_test_norm = load_tensor('data/' + train_or_test+ '/tensor/')
    ae = Autoencoder(len(x_test_data[0]), len(x_test_data[0][0]), len(x_test_data[0][0][0]))
    ae.autoencoder_model.load_weights('weights_autoencoder/ae_weights.h5f')
    ae.compress(x_test_data, x_test_norm, train_or_test)
