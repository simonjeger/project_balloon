import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Reshape, Conv3D
from keras.layers import MaxPooling3D, Dropout, UpSampling3D

class Autoencoder():
    def __init__(self):
        # define size
        self.window_size = 3
        name_list = os.listdir('data/train/tensor/')
        tensor = torch.load('data/train/tensor/' + name_list[0])
        size_z = len(tensor[0][0])
        size_c = len(tensor[0][0][0])

        self.img_shape = (self.window_size*2, self.window_size*2, size_z, size_c)

        optimizer = Adam(lr=0.001)
        self.autoencoder_model = self.build_model()
        self.autoencoder_model.compile(loss='mse', optimizer=optimizer)
        self.autoencoder_model.summary()

        self.half_model = self.build_half_model()
        self.half_model.compile(loss='mse', optimizer=optimizer)

    def build_model(self):
        input_layer = Input(shape=self.img_shape)
        self.bottleneck = 10

        # encoder
        self.ec_c = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(input_layer)
        self.ec_p = MaxPooling3D((2, 2, 2), padding='same')(self.ec_c)
        self.ec_f = Flatten()(self.ec_p)
        self.ec_bn = Dense(self.bottleneck)(self.ec_f)

        # decoder
        self.dc_d = Dense(self.ec_f.shape[1])(self.ec_bn)
        self.dc_r = Reshape(self.ec_p.shape[1::])(self.dc_d)
        self.dc_c = Conv3D(self.ec_c.shape[-1], (3, 3, 3), activation='relu', padding='same')(self.dc_r)
        self.dc_u = UpSampling3D((2, 2, 2))(self.dc_c)
        output_layer = Conv3D(self.img_shape[-1], (3, 3, 3), activation='linear', padding='same')(self.dc_u) #activation used to be sigmoid

        return Model(input_layer, output_layer)

    def build_half_model(self):
        input_layer = Input(shape=self.img_shape)

        # only encode this time, last layer is compressed version
        self.ec_c = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(input_layer)
        self.ec_p = MaxPooling3D((2, 2, 2), padding='same')(self.ec_c)
        self.ec_f = Flatten()(self.ec_p)
        output_layer = Dense(self.bottleneck)(self.ec_f)

        return Model(input_layer, output_layer)

    def train_model(self, x_train, y_train, x_val, y_val, epochs, batch_size=20, plot=False):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')
        history = self.autoencoder_model.fit(x_train, y_train,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             validation_data=(x_val, y_val),
                                             callbacks=[early_stopping])
        if plot:
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()
            plt.close()

    def eval_model(self, x_test):
        preds = self.autoencoder_model.predict(x_test)
        return preds

    def compress(self, data):
        pred = self.half_model.predict(data)
        return pred[0]

    def window(self, data, center):
        size_x = len(data)
        size_y = len(data[0])
        size_z = len(data[0][0])
        size_c = len(data[0][0][0])

        start_x = int(max(center[0] - self.window_size, 0))
        start_y = int(max(center[1] - self.window_size, 0))
        end_x = int(min(center[0] + self.window_size, size_x))
        end_y = int(min(center[1] + self.window_size, size_x))

        window = np.zeros((self.window_size*2,self.window_size*2,size_z,size_c))
        fill_in = data[start_x:end_x,start_y:end_y,:]
        # touching the left border
        if (start_x == 0) & (start_y == 0):
            #print('both left')
            window[0:end_x,0:end_y,:] = fill_in
        elif (start_x == 0) & (start_y != 0) & (end_y != size_y):
            #print('x left')
            window[0:end_x,:,:] = fill_in
        elif (start_x != 0) & (end_x != size_x) & (start_y == 0):
            #print('y left')
            window[:,0:end_y,:] = fill_in

        # overlapping cases
        elif (start_x == 0) & (end_y == size_y):
            #print('x left, y right')
            window[0:end_x,self.window_size*2-(end_y-start_y):self.window_size*2,:] = fill_in
        elif (end_x == size_x) & (start_y == 0):
            #print('x right, y left')
            window[self.window_size*2-(end_x-start_x):self.window_size*2,0:end_y,:] = fill_in

        # touching the right border
        elif (end_x == size_x) & (end_y == size_y):
            #print('both right')
            window[self.window_size*2-(end_x-start_x):self.window_size*2,self.window_size*2-(end_y-start_y):self.window_size*2,:] = fill_in
        elif (end_x == size_x) & (start_y != 0) & (end_y != size_y):
            #print('x right')
            window[self.window_size*2-(end_x-start_x):self.window_size*2,:,:] = fill_in
        elif (start_x != 0) & (end_x != size_x) & (end_y == size_y):
            #print('y right')
            window[:,self.window_size*2-(end_y-start_y):self.window_size*2,:] = fill_in

        # if not touching anything¨
        else:
            #print('no touch')
            window = fill_in
        return window

def load_tensor(path):
    name_list = os.listdir(path)
    name_list.sort()
    tensor_list = []
    for name in name_list:
        tensor = torch.load(path + name)
        tensor_list.append(tensor)
    return np.array(tensor_list)
