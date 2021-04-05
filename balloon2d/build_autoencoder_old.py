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

car = True

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # variables
        self.bottleneck = 20
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
            # encoder
            self.cnn_layers = Sequential(
                Conv2d(3, 16, 3, padding=1),
                Conv2d(16, 4, 3, padding=1),
                MaxPool2d(2, 2)
            )

            # linear layers
            self.fc21 = Sequential(Linear(19500, self.bottleneck)) #mu layer
            self.fc22 = Sequential(Linear(19500, self.bottleneck)) #logvariance layer

            self.fc31 = Sequential(Linear(self.bottleneck, 200)) #logvariance layer

            # decoder
            self.ct_layers = nn.Sequential(
                ConvTranspose2d(2, 16, 2, stride=2),
                ConvTranspose2d(16, 3, 2, stride=2),
                Upsample([self.size_x, self.size_z]),
            )

            self.ct_layers2 = nn.Sequential(
                Linear(200,self.size_c*self.size_x*self.size_z)
            )
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

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def encode(self, x):
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
        z = self.fc31(z)
        if car:
            z = self.ct_layers2(z)
            z = z.view(self.batch_size,self.size_c,self.size_x,self.size_z)
        else:
            z = z.view(self.batch_size,2,2,-1)
            z = self.ct_layers(z)
        return z

    def forward(self, x):
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

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(self.train_loader.dataset)))

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
