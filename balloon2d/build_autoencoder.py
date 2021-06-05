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

car = False

class VAE(nn.Module):
    def __init__(self, writer=None):
        super(VAE, self).__init__()

        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        # logger
        self.writer = writer

        # variables
        self.bottleneck_terrain = 2
        self.bottleneck_wind = yaml_p['bottleneck']

        if yaml_p['type'] == 'regular':
            self.bottleneck = self.bottleneck_wind
        elif yaml_p['type'] == 'squished':
            self.bottleneck = self.bottleneck_wind + 2

        self.window_size = yaml_p['window_size']
        self.window_size_total = 2*self.window_size + 1
        self.batch_size = 10

        #read in data
        if car:
            train_dataset = custom_data('data_cars/train/')
            test_dataset = custom_data('data_cars/test/')
        else:
            train_dataset = wind_data('data/train/tensor/')
            test_dataset = wind_data('data/test/tensor/')

        self.size_c = 2
        self.size_x = len(train_dataset[0][0])
        self.size_z = len(train_dataset[0][0][0])

        if car:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=self.batch_size, drop_last=True) # I'll build my own batches through the window function
            self.test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=self.batch_size, drop_last=True)
        else:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=self.batch_size, drop_last=True) # I'll build my own batches through the window function
            self.test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=self.batch_size, drop_last=True)

        ngf = 64 #64
        ndf = 64 #64
        nc = self.size_c

        # set kernel size of x direction
        if self.window_size_total == 3:
            k_enc = [3,1,1,1]
            k_dec = [1,2,2,1]
        elif self.window_size_total == 5:
            k_enc = [3,3,1,1]
            k_dec = [1,2,3,2]
        elif self.window_size_total == 7:
            k_enc = [3,3,2,2]
            k_dec = [1,3,3,3]
        elif self.window_size_total == 9:
            k_enc = [3,3,3,3]
            k_dec = [2,3,3,4]
        elif self.window_size_total == 11:
            k_enc = [5,3,3,3]
            k_dec = [2,4,4,4]
        elif self.window_size_total == 13:
            k_enc = [5,5,3,3]
            k_dec = [4,4,4,4]
        else:
            print('ERROR: This window_size is not supported')

        # encoder
        self.encoder = nn.Sequential( # kernel_size = H_in - H_out - 1 #for basic case with padding=0, stride=1, dialation=1
            nn.Conv2d(nc, ndf, 19, bias=False), #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 17, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 13, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, 1024, 4, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.encoder_wind = nn.Sequential( # kernel_size = H_in - H_out - 1 #for basic case with padding=0, stride=1, dialation=1
            nn.Conv2d(nc, ndf, (k_enc[0],53), bias=False), #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, (k_enc[1],33), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, (k_enc[2],15), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, 1024, (k_enc[3],7), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, ngf * 4, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 12, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf * 2, ngf, 18, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf, nc, 20, bias=False),
            nn.Sigmoid()
            )


        self.decoder_conv = nn.Sequential(
            #nn.Conv2d(1024, ngf * 8, 1, bias=False),
            nn.ConvTranspose2d(1024, ngf * 8, 1, bias=False),
            nn.ReLU(True),
            nn.Upsample([4, 4]),

            #nn.Conv2d(ngf * 8, ngf * 4, 4, bias=False),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, bias=False),
            nn.ReLU(True),
            nn.Upsample([8, 8]),

            #nn.Conv2d(ngf * 4, ngf * 2, 4, bias=False),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, bias=False),
            nn.ReLU(True),
            nn.Upsample([16, 16]),

            #nn.Conv2d(ngf * 2, ngf, 4, bias=False),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, bias=False),
            nn.ReLU(True),
            nn.Upsample([self.size_x, self.size_z]),

            #nn.Conv2d(ngf, nc, 1, bias=False),
            nn.ConvTranspose2d(ngf, nc, 1, bias=False),
            nn.Sigmoid()
            )

        self.decoder_wind = nn.Sequential(
            nn.ConvTranspose2d(1024, ngf * 4, (k_dec[0],6), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, (k_dec[1],16), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf * 2, ngf, (k_dec[2],34), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf, nc, (k_dec[3],52), bias=False),
            )

        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, self.bottleneck_wind)
        self.fc22 = nn.Linear(512, self.bottleneck_wind)

        self.fc3 = nn.Linear(self.bottleneck_wind, 512)
        self.fc4 = nn.Linear(512, 1024)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4) #used to be 1e-3

        self.step_n = 0

    def encode(self, x):
        #print("initial size", x.size())
        if car:
            conv = self.encoder(x);
        else:
            conv = self.encoder_wind(x);
        #print("encode conv", conv.size())
        h1 = self.fc1(conv.view(len(conv), -1))
        #print("encode h1", h1.size())
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu,logvar):
        if self.training:
            # multiply log variance with 0.5, then in-place exponent yielding the standard deviation
            std = logvar.mul(0.5).exp_()  # type: Variable
            if self.device == 'cuda:0':
                eps = torch.cuda.FloatTensor(std.size()).normal_()
            else:
                eps = torch.FloatTensor(std.size()).normal_()
            eps = Variable(std.data.new(std.size()).normal_())
            # sample from a normal distribution with standard deviation = std and mean = mu
            return eps.mul(std).add_(mu)

        else:
            # mu has highest probability
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        deconv_input = deconv_input.view(len(deconv_input),-1,1,1)
        #print("deconv_input", deconv_input.size())
        if car:
            return self.decoder(deconv_input)
        else:
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
        if car:
            BCE = F.binary_cross_entropy(recon_x, x)
        else:
            BCE = nn.functional.mse_loss(recon_x, x)

        # how close is the distribution to mean = 0, std = 1?
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= len(x)*self.size_x*self.size_z*self.size_c #normalise by same number of elements as in reconstruction

        if self.writer is not None:
            self.writer.add_scalar('BCE_loss', BCE, self.step_n)
            self.writer.add_scalar('KLD_loss', KLD, self.step_n)

        return BCE + KLD

    def model_train(self, epoch):
        # toggle model to train mode
        self.train()
        train_loss = 0

        for batch_idx, data in enumerate(self.train_loader):
            data = data.type(torch.FloatTensor) #numpy uses doubles, so just to be save

            if not car:
                data_window = torch.zeros([self.batch_size, self.size_c, self.window_size_total, self.size_z])
                for i in range(self.batch_size): #number of samples we take from the same world
                    center = np.random.randint(0,self.size_x)
                    if yaml_p['type'] == 'regular':
                        to_fill = self.window(data[i], center)
                    elif yaml_p['type'] == 'squished':
                        ceiling = [np.random.uniform(0.9, 1) * self.size_z]*self.size_x
                        to_fill = self.window_squished(data[i], center, ceiling)
                    data_window[i,:,:,:] = to_fill[-3:-1]
                data = data_window #to keep naming convention

            data = data.to(self.device)
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

            # logger
            if type(self.writer) is not None:
                self.writer.add_scalar('ae_training_loss', loss.data.item(), self.step_n)

            log_interval = 10
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data) * self.batch_size, len(self.train_loader.dataset)*self.batch_size, #because I do the batches manually
                    100. * batch_idx / len(self.train_loader),
                    loss.data.item() / len(data)))

            self.step_n += 1

        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(self.train_loader.dataset)))

        if car:
            # visualization of latent space
            sample = Variable(torch.randn(self.batch_size, self.bottleneck_wind))
            sample = sample.to(self.device)
            sample = self.decode(sample).cpu()

            save_image(sample.data.view(self.batch_size, self.size_c, self.size_x, self.size_z), 'autoencoder/results/sample.png')
        else:
            sample = Variable(torch.randn(self.batch_size, self.bottleneck_wind))
            sample = sample.to(self.device)
            sample = self.decode(sample).cpu()

            self.visualize(sample, 'autoencoder/results/sample_' + str(yaml_p['process_nr']) + '.png')

    def model_test(self, epoch):
        # toggle model to test / inference mode
        self.eval()
        test_loss = 0

        # each data is of self.batch_size (default 128) samples
        for batch_idx, data in enumerate(self.test_loader):
            data = data.type(torch.FloatTensor) #numpy uses doubles, so just to be save

            if not car:
                data_window = torch.zeros([self.batch_size, self.size_c, self.window_size_total, self.size_z])
                for i in range(self.batch_size): #number of samples we take from the same world
                    center = np.random.randint(0,self.size_x)
                    if yaml_p['type'] == 'regular':
                        to_fill = self.window(data[i], center)
                    elif yaml_p['type'] == 'squished':
                        ceiling = [np.random.uniform(0.9, 1) * self.size_z]*self.size_x
                        to_fill = self.window_squished(data[i], center, ceiling)
                    data_window[i,:,:,:] = to_fill[-3:-1]
                data = data_window #to keep naming convention

            data = data.to(self.device)
            # we're only going to infer, so no autograd at all required: volatile=True
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = self(data)
            test_loss += self.loss_function(recon_batch, data, mu, logvar).data.item()

            # get data back to plot it on cpu
            data = data.cpu()
            recon_batch = recon_batch.cpu()

            if car:
                if batch_idx == 0:
                    n = min(data.size(0), 8)

                    comparison = torch.cat([data[:n], recon_batch[:n]])

                    save_image(comparison.data.cpu(), 'autoencoder/results/reconstruction.png', nrow=n)

            else:
                self.visualize(data, 'autoencoder/results/' + str(i).zfill(5) + '_real.png')
                self.visualize(recon_batch, 'autoencoder/results/' + str(i).zfill(5) + '_recon.png')
                self.visualize(abs(data-recon_batch), 'autoencoder/results/' + str(i).zfill(5) + '_error_', error=True)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def save_weights(self, path):
        torch.save(self.state_dict(), path + '/model_' + str(yaml_p['process_nr']) + '.pt')

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def window(self, data, center):
        window = np.zeros((len(data),self.window_size_total,self.size_z))
        data_padded = np.zeros((len(data),self.size_x+2*self.window_size,self.size_z))
        data_padded[:,self.window_size:-self.window_size,:] = data

        for i in range(self.window_size):
            data_padded[:,i,:] = data_padded[:,self.window_size,:]
            data_padded[:,-(i+1),:] = data_padded[:,-(self.window_size+1),:]

        start_x = int(np.clip(center,0,self.size_x-1))
        end_x = int(start_x + self.window_size_total)

        window = data_padded[:,start_x:end_x,:]
        window = torch.tensor(window)
        return window

    def window_squished(self, data, center, ceiling):
        res = self.size_z
        data_squished = np.zeros((len(data),self.size_x,res))
        for i in range(self.size_x):
            bottom = data[0,i,0]
            top = ceiling[i]

            x_old = np.arange(0,self.size_z,1)
            x_new = np.linspace(bottom,top,res)
            data_squished[0,:,:] = data[0,:,:] #terrain stays the same

            for j in range(1,len(data)):
                data_squished[j,i,:] = np.interp(x_new,x_old,data[j,i,:])

        data_padded = np.zeros((len(data_squished),self.size_x+2*self.window_size,res))
        data_padded[:,self.window_size:-self.window_size,:] = data_squished

        for i in range(self.window_size):
            data_padded[:,i,:] = data_padded[:,self.window_size,:]
            data_padded[:,-(i+1),:] = data_padded[:,-(self.window_size+1),:]

        start_x = int(np.clip(center,0,self.size_x-1))
        end_x = int(start_x + self.window_size_total)

        window = data_padded[:,start_x:end_x,:]
        window = torch.tensor(window)
        return window

    def visualize(self, data, path, error=False):
        n = 0 #which port of the batch should be visualized
        mean_x = data[n][-2,:,:].detach().numpy()
        mean_z = data[n][-1,:,:].detach().numpy()
        #sig_xz = data[n][-1,:,:].detach().numpy()

        size_x = len(mean_x)
        size_z = len(mean_x[0])

        z,x = np.meshgrid(np.arange(0, size_z, 1),np.arange(0, size_x, 1))
        fig, ax = plt.subplots(frameon=False)
        render_ratio = int(yaml_p['unit_xy'] / yaml_p['unit_z'])

        if error == False:
            cmap = sns.diverging_palette(145, 300, s=50, center="dark", as_cmap=True)
            ax.imshow(mean_z.T, origin='lower', extent=[0, size_x, 0, size_z], cmap=cmap, alpha=0.5, vmin=-5, vmax=5, interpolation='bilinear')

            cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
            map = ax.imshow(mean_x.T, origin='lower', extent=[0, size_x, 0, size_z], cmap=cmap, alpha=0.5, vmin=-5, vmax=5, interpolation='bilinear')
            fig.colorbar(map)
        else:
            cmap = 'Blues'
            map = ax.imshow(mean_z.T, origin='lower', extent=[0, size_x, 0, size_z], cmap=cmap, alpha=0.5, interpolation='bilinear')
            fig.colorbar(map)

            cmap = 'Reds'
            map = ax.imshow(mean_x.T, origin='lower', extent=[0, size_x, 0, size_z], cmap=cmap, alpha=0.5, interpolation='bilinear')
            fig.colorbar(map)

        ax.set_aspect(1/render_ratio)
        plt.savefig(path)
        plt.close()

    def compress(self, data, position, ceiling):
        center = position[0]
        if yaml_p['type'] == 'regular':
            to_fill = self.window(data, center)

        elif yaml_p['type'] == 'squished':
            pos_x = np.clip(int(position[0]),0,self.size_x - 1)

            rel_pos = torch.tensor([(position[1]-data[0,self.window_size,0]) / (ceiling[pos_x] - data[0,self.window_size,0])])
            size = (ceiling[pos_x] - data[0,self.window_size,0])/self.size_z

            to_fill = self.window_squished(data, center, ceiling)
        data = to_fill[-3:-1]

        # wind
        data = data.view(1,self.size_c,self.window_size_total,self.size_z)
        data = data.type(torch.FloatTensor)
        self.eval() #toggle model to test / inference mode
        self.device = 'cpu'

        # we're only going to infer, so no autograd at all required: volatile=True
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = self(data)
        comp = mu[0].detach().numpy()

        if yaml_p['type'] == 'regular':
            result = comp
        elif yaml_p['type'] == 'squished':
            result = np.concatenate((comp,[rel_pos, size]))

        return result
