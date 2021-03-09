from build_autoencoder import Autoencoder
from build_autoencoder import load_tensor
from visualize_wind_map import visualize_wind_map

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
matplotlib.use("Agg") # is needed for processing on cluster
from sklearn.model_selection import train_test_split

def autoencoder_train():
    ae = Autoencoder()

    x_train_data = load_tensor('data/train/tensor/')

    x_train_window = []
    for _ in range(5): #number of samples we take from the same
        for i in range(len(x_train_data)):
            data = ae.window(x_train_data[i])
            x_train_window.append(data)
    x_train_window = np.array(x_train_window)

    y_train_window = x_train_window
    x_train_window, x_val, y_train_window, y_val = train_test_split(x_train_window, y_train_window, test_size=0.2, random_state=42)

    ae.train_model(x_train_window, y_train_window, x_val, y_val, epochs=200, batch_size=200, plot=False)
    ae.autoencoder_model.save_weights('weights_autoencoder/ae_weights.h5f', overwrite=True)

    # visualizing for decoding purpouses
    """
    pred = ae.eval_model(x_val)
    visualize_wind_map('train', x_val[0:3])
    visualize_wind_map('train', pred[0:3])
    """
