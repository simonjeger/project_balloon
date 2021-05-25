import os
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, ConvTranspose2d, Sigmoid, Upsample
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import seaborn as sns

from utils.import_data import custom_data, wind_data
from  visualize_world import visualize_world

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

class VAE(nn.Module):
    def __init__(self, writer=None):
        super(VAE, self).__init__()
        # logger
        self.writer = writer

        # variables
        self.bottleneck_terrain = 2
        self.bottleneck_wind = 10
        self.bottleneck = self.bottleneck_terrain + self.bottleneck_wind

        self.window_size = 1
        self.window_size_total = 2*self.window_size + 1
        self.batch_size = 10

        #read in data
        train_dataset = wind_data('data/train/tensor/')
        test_dataset = wind_data('data/test/tensor/')

        self.size_c = len(train_dataset[0])
        self.size_x = len(train_dataset[0][0])
        self.size_y = len(train_dataset[0][0][0])
        self.size_z = len(train_dataset[0][0][0][0])

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=1) # I'll build my own batches through the window function
        self.test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)

        ngf = 64 #64
        ndf = 64 #64
        nc = self.size_c
        # encoder
        self.encoder_wind = nn.Sequential( # kernel_size = H_in - H_out - 1 #for basic case with padding=0, stride=1, dialation=1
            nn.Conv3d(nc, ndf, (3,3,53), bias=False), #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf, ndf * 2, (1,1,33), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 2, ndf * 4, (1,1,15), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 4, 1024, (1,1,7), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder_wind = nn.Sequential(
            nn.ConvTranspose3d(1024, ngf * 4, (1,1,6), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose3d(ngf * 4, ngf * 2, (2,2,16), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose3d(ngf * 2, ngf, (2,2,34), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose3d(ngf, nc, (1,1,52), bias=False),
            )

        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, self.bottleneck_wind)
        self.fc22 = nn.Linear(512, self.bottleneck_wind)

        self.fc3 = nn.Linear(self.bottleneck_wind, 512)
        self.fc4 = nn.Linear(512, 1024)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4) #used to be 1e-3

    def encode(self, x):
        #print("initial size", x.size())
        conv = self.encoder_wind(x);
        #print("encode conv", conv.size())
        h1 = self.fc1(conv.view(len(conv), -1))
        #print("encode h1", h1.size())
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu,logvar):
        if self.training:
            # multiply log variance with 0.5, then in-place exponent yielding the standard deviation
            std = logvar.mul(0.5).exp_()  # type: Variable
            eps = Variable(std.data.new(std.size()).normal_())
            # sample from a normal distribution with standard deviation = std and mean = mu
            return eps.mul(std).add_(mu)

        else:
            # mu has highest probability
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        deconv_input = deconv_input.view(len(deconv_input),-1,1,1,1)
        #print("deconv_input", deconv_input.size())
        #print("deconv_output", self.decoder_wind(deconv_input).size())
        return self.decoder_wind(deconv_input)
        #return self.fake_decoder(deconv_input).view(len(deconv_input),3, 30, 30)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        self.have_cuda = False
        if self.have_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # print("x", x.size())
        mu, logvar = self.encode(x)
        # print("mu, logvar", mu.size(), logvar.size())
        z = self.reparametrize(mu, logvar)
        # print("z", z.size())
        decoded = self.decode(z)
        # print("decoded", decoded.size())
        return decoded, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # how well do input x and output recon_x agree?
        BCE = nn.functional.mse_loss(recon_x, x)

        # how close is the distribution to mean = 0, std = 1?
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= len(x)*self.size_x*self.size_y*self.size_z*self.size_c #normalise by same number of elements as in reconstruction

        return BCE + KLD

    def model_train(self, epoch):
        # toggle model to train mode
        self.train()
        train_loss = 0

        for batch_idx, data in enumerate(self.train_loader):
            data = data.type(torch.FloatTensor) #numpy uses doubles, so just to be save

            data_window = torch.zeros([self.batch_size, self.size_c, self.window_size_total, self.window_size_total, self.size_z])
            for i in range(self.batch_size): #number of samples we take from the same world
                center_x = np.random.randint(0,self.size_x)
                center_y = np.random.randint(0,self.size_y)
                center_z = np.random.randint(0,self.size_z)
                data_window[i,:,:,:,:] = self.window(data[0], [center_x, center_y, center_z])
            data = data_window #to keep naming convention

            data = Variable(data)
            self.optimizer.zero_grad()

            # push whole batch of data through VAE.forward() to get recon_loss
            recon_batch, mu, logvar = self(data)

            # calculate scalar loss
            loss = self.loss_function(recon_batch, data, mu, logvar)
            # calculate the gradient of the loss w.r.t. the graph leaves i.e. input variables
            loss.backward()
            train_loss += loss.data.item()
            self.optimizer.step()
            log_interval = 100
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset)*self.batch_size, #because I do the batches manually
                    100. * batch_idx / len(self.train_loader),
                    loss.data.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(self.train_loader.dataset)))

        # logger
        if type(self.writer) is not None:
            self.writer.add_scalar('autoencoder training loss', train_loss / len(self.train_loader.dataset) , epoch * len(self.train_loader.dataset))

        sample = Variable(torch.randn(self.batch_size, self.bottleneck_wind))
        sample = self.decode(sample).cpu()

        self.visualize(sample, 'autoencoder/results/sample_')

    def model_test(self, epoch):
        # toggle model to test / inference mode
        self.eval()
        test_loss = 0

        # each data is of self.batch_size (default 128) samples
        for i, data in enumerate(self.test_loader):
            data = data.type(torch.FloatTensor) #numpy uses doubles, so just to be save

            data_window = torch.zeros([self.batch_size, self.size_c, self.window_size_total, self.window_size_total, self.size_z])
            for j in range(self.batch_size): #number of samples we take from the same world
                center_x = np.random.randint(0,self.size_x)
                center_y = np.random.randint(0,self.size_y)
                center_z = np.random.randint(0,self.size_z)
                data_window[j,:,:,:,:] = self.window(data[0], [center_x, center_y, center_z])
            data = data_window #to keep naming convention

            # we're only going to infer, so no autograd at all required: volatile=True
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = self(data)
            test_loss += self.loss_function(recon_batch, data, mu, logvar).data.item()

            self.visualize(data, 'autoencoder/results/' + str(i).zfill(5) + '_real_')
            self.visualize(recon_batch, 'autoencoder/results/' + str(i).zfill(5) + '_recon_')

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def save_weights(self, path):
        torch.save(self.state_dict(), path + '/model.pt')

    def load_weights(self, path):
        self.load_state_dict(torch.load(path + '/model.pt'))
        self.eval()

    def window(self, data, position):
        window = np.zeros((len(data),self.window_size_total,self.size_z))
        data_padded = np.zeros((len(data),self.size_x+2*self.window_size,self.size_y+2*self.window_size,self.size_z))

        data_padded[:,self.window_size:-self.window_size,self.window_size:-self.window_size,:] = data

        for i in range(self.window_size):
            data_padded[:,i,:,:] = data_padded[:,self.window_size,:,:]
            data_padded[:,:,i,:] = data_padded[:,:,self.window_size,:]
            data_padded[:,-(i+1),:,:] = data_padded[:,-(self.window_size+1),:,:]
            data_padded[:,:,-(i+1),:] = data_padded[:,:,-(self.window_size+1),:]

        start_x = int(position[0])
        start_y = int(position[1])
        end_x = int(position[0] + self.window_size_total)
        end_y = int(position[1] + self.window_size_total)

        window = data_padded[:,start_x:end_x,start_y:end_y,:]
        window = torch.tensor(window)
        return window

    def visualize(self, data, path):
        n = 0 #which port of the batch should be visualized

        for dim in ['xz', 'yz']:
            if dim == 'xz':
                mean_1 = data[n][-4,:,self.window_size,:].detach().numpy()
                mean_2 = data[n][-2,:,self.window_size,:].detach().numpy()
            if dim == 'yz':
                mean_1 = data[n][-4,self.window_size,:,:].detach().numpy()
                mean_2 = data[n][-2,self.window_size,:,:].detach().numpy()

            size_1 = len(mean_1)
            size_2 = len(mean_1[0])

            z,x = np.meshgrid(np.arange(0, size_2, 1),np.arange(0, size_1, 1))
            fig, ax = plt.subplots(frameon=False)
            render_ratio = int(yaml_p['unit_xy'] / yaml_p['unit_z'])

            #cmap = sns.diverging_palette(220, 20, as_cmap=True)
            cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
            ax.imshow(mean_1.T, origin='lower', extent=[0, size_1, 0, size_2], cmap=cmap, alpha=0.5, vmin=-5, vmax=5, interpolation='bilinear')

            #cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)
            cmap = sns.diverging_palette(145, 300, s=50, center="dark", as_cmap=True)
            ax.imshow(mean_2.T, origin='lower', extent=[0, size_1, 0, size_2], cmap=cmap, alpha=0.5, vmin=-5, vmax=5, interpolation='bilinear')

            ax.set_axis_off()
            ax.set_aspect(1/render_ratio)
            plt.savefig(path + dim + '.png')
            plt.close()

    def compress(self, data, position):
        data = self.window(data, position) #to keep naming convention

        # terrain
        #terrain = self.compress_terrain(data, position)

        # wind
        data = data[-3::,:,:] #we only autoencode wind
        data = data.view(1,3,self.window_size_total,self.size_z)
        data = data.type(torch.FloatTensor)
        self.eval() #toggle model to test / inference mode

        # we're only going to infer, so no autograd at all required: volatile=True
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = self(data)

        #comp = np.concatenate([terrain, mu[0].detach().numpy(), logvar[0].detach().numpy()])
        comp = np.concatenate([mu[0].detach().numpy()])
        return comp

    def compress_terrain(self, data, position):
        rel_x = self.window_size + position[0] - int(position[0]) #to accound for the rounding that occured in window function
        rel_z = position[1]
        terrain = data[0,:,0]

        x = np.linspace(0,self.window_size_total,len(terrain))
        distances = []
        res = 100
        for i in range(len(terrain)*res):
            #distances.append(np.sqrt((rel_x-i)**2 + (rel_z-terrain[i])**2))
            distances.append(np.sqrt((rel_x-i/res)**2 + (rel_z-np.interp(i/res,x,terrain))**2))

        distance = np.min(distances)
        bearing = np.arctan2(np.argmin(distances)/res - rel_x, rel_z - np.interp(np.argmin(distances)/res,x,terrain))

        return [distance, bearing]
