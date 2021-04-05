import os
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, ConvTranspose2d, Sigmoid, Upsample
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from import_data import custom_data, wind_data
from  visualize_world import visualize_world

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

car = True

class VAE(nn.Module):
    def __init__(self, writer):
        super(VAE, self).__init__()
        # logger
        self.writer = writer

        # variables
        self.bottleneck = 500
        self.batch_size = 5

        #read in data
        if car:
            train_dataset = custom_data('data_cars/train/')
            test_dataset = custom_data('data_cars/test/')
        else:
            train_dataset = wind_data('data/train/tensor/')
            test_dataset = wind_data('data/test/tensor/')

        self.size_c = len(train_dataset[0])
        self.size_x = len(train_dataset[0][0])
        self.size_z = len(train_dataset[0][0][0])

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=self.batch_size)
        self.test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=self.batch_size)

        if car:
            ngf = 64
            ndf = 64
            nc = 3
            # encoder
            self.encoder = nn.Sequential(
                # input is (nc) x 28 x 28
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 14 x 14
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 7 x 7
                nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 4 x 4
                nn.Conv2d(ndf * 4, 1024, 4, 1, 0, bias=False),
                # nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample([1, 1]) # I added that
                # nn.Sigmoid()
            )

            self.decoder = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(1024, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
                # nn.BatchNorm2d(ngf),
                # nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                # nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                # nn.Tanh()
                nn.Sigmoid(),
                nn.Upsample([self.size_x, self.size_z])
                # state size. (nc) x 64 x 64
            )

            self.fc1 = nn.Linear(1024, 512)
            self.fc21 = nn.Linear(512, self.bottleneck)
            self.fc22 = nn.Linear(512, self.bottleneck)

            self.fc3 = nn.Linear(self.bottleneck, 512)
            self.fc4 = nn.Linear(512, 1024)

            self.lrelu = nn.LeakyReLU()
            self.relu = nn.ReLU()

        else:
            # encoder
            self.cnn_layers = Sequential(
                Conv2d(3, 16, 3, padding=1),
                Conv2d(16, 4, 3, padding=1),
                MaxPool2d(2, 2)
            )

            # linear layers
            self.fc21 = Sequential(Linear(300, self.bottleneck)) #mu layer
            self.fc22 = Sequential(Linear(300, self.bottleneck)) #logvariance layer

            self.fc31 = Sequential(Linear(self.bottleneck, 300)) #logvariance layer

            # decoder
            self.ct_layers = nn.Sequential(
                ConvTranspose2d(2, 16, 2, stride=2),
                ConvTranspose2d(16, 3, 2, stride=2),
                Upsample([self.size_x, self.size_z]),
            )

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4) #used to be 1e-3

    def encode(self, x):
        if car:
            conv = self.encoder(x);
            # print("encode conv", conv.size())
            h1 = self.fc1(conv.view(-1, 1024))
            # print("encode h1", h1.size())
            return self.fc21(h1), self.fc22(h1)
        else:
            x = self.cnn_layers(x)
            x = x.view(self.batch_size, -1)
            return self.fc21(x), self.fc22(x)

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
        if car:
            h3 = self.relu(self.fc3(z))
            deconv_input = self.fc4(h3)
            # print("deconv_input", deconv_input.size())
            deconv_input = deconv_input.view(self.batch_size,-1,1,1)
            # print("deconv_input", deconv_input.size())
            return self.decoder(deconv_input)
        else:
            z = self.fc31(z)
            z = z.view(self.batch_size,2,2,-1)
            z = self.ct_layers(z)
            return z

    # added only for car
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
        if car:
            # print("x", x.size())
            mu, logvar = self.encode(x)
            # print("mu, logvar", mu.size(), logvar.size())
            z = self.reparametrize(mu, logvar)
            # print("z", z.size())
            decoded = self.decode(z)
            # print("decoded", decoded.size())
            return decoded, mu, logvar
        else:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # how well do input x and output recon_x agree?
        #BCE = F.binary_cross_entropy(recon_x, x)
        BCE = F.binary_cross_entropy_with_logits(recon_x, x)

        # how close is the distribution to mean = 0, std = 1?
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= self.batch_size * self.size_x*self.size_z*self.size_c #normalise by same number of elements as in reconstruction

        return BCE + KLD

    def model_train(self, epoch):
        # toggle model to train mode
        self.train()
        train_loss = 0

        for batch_idx, data in enumerate(self.train_loader):
            data = data.type(torch.FloatTensor) #numpy uses doubles, so just to be save
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
            log_interval = 10
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.data.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(self.train_loader.dataset)))

        # logger
        self.writer.add_scalar('autoencoder training loss', train_loss / len(self.train_loader.dataset) , epoch * len(self.train_loader.dataset))

        if car:
            # visualization of latent space
            sample = Variable(torch.randn(self.batch_size, self.bottleneck))
            sample = self.decode(sample).cpu()

            save_image(sample.data.view(self.batch_size, self.size_c, self.size_x, self.size_z),
                #           'results/sample_' + str(epoch) + '.png')
                          'results/sample.png')

    def model_test(self, epoch):
        # toggle model to test / inference mode
        self.eval()
        test_loss = 0

        # each data is of self.batch_size (default 128) samples
        for i, data in enumerate(self.test_loader):
            # we're only going to infer, so no autograd at all required: volatile=True
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = self(data)
            test_loss += self.loss_function(recon_batch, data, mu, logvar).data.item()

            if car:
                if i == 0:
                    n = min(data.size(0), 8)

                    comparison = torch.cat([data[:n],
                                              #recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
                                              recon_batch[:n]])

                    save_image(comparison.data.cpu(),
                    #            'results/reconstruction_' + str(epoch) + '.png', nrow=n)
                                'results/reconstruction.png', nrow=n)

            else:
                # save
                for b in range(self.batch_size):
                    # compressed
                    comp = torch.cat([mu[b], logvar[b]])
                    torch.save(comp, 'data/test/tensor_comp/wind_map_comp' + str(i*self.batch_size+b).zfill(5) + '.pt')

                    #validation
                    real = np.array(data[b]).transpose(1, -1, 0)
                    real = torch.tensor(real)
                    recon = recon_batch[b].detach().numpy().transpose(1, -1, 0)
                    recon = torch.tensor(recon)

                    visualize_world('test', real)
                    visualize_world('test', recon)


        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
