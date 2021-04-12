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

from import_data import custom_data, wind_data
from  visualize_world import visualize_world

car = False

class VAE(nn.Module):
    def __init__(self, writer='no_writer'):
        super(VAE, self).__init__()
        # logger
        self.writer = writer

        # variables
        self.bottleneck = 20
        self.window_size = 3
        self.batch_size = 10

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

        if car:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=self.batch_size) # I'll build my own batches through the window function
            self.test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=self.batch_size)
        else:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=1) # I'll build my own batches through the window function
            self.test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)

        ngf = 64 #64
        ndf = 64 #64
        nc = self.size_c
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
            nn.Conv2d(nc, ndf, (5,9), bias=False), #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, 1024, 1, bias=False),
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
            nn.ConvTranspose2d(1024, ngf * 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf * 2, ngf, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf, nc, (3,7), bias=False),
            )

        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, self.bottleneck)
        self.fc22 = nn.Linear(512, self.bottleneck)

        self.fc3 = nn.Linear(self.bottleneck, 512)
        self.fc4 = nn.Linear(512, 1024)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4) #used to be 1e-3

    def encode(self, x):
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
            eps = Variable(std.data.new(std.size()).normal_())
            # sample from a normal distribution with standard deviation = std and mean = mu
            return eps.mul(std).add_(mu)

        else:
            # mu has highest probability
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        #print("deconv_input", deconv_input.size())
        deconv_input = deconv_input.view(len(deconv_input),-1,1,1)
        #print("deconv_input", deconv_input.size())
        if car:
            return self.decoder(deconv_input)
        else:
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
        #BCE = F.binary_cross_entropy_with_logits(recon_x, x)

        # how close is the distribution to mean = 0, std = 1?
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= len(x)*self.size_x*self.size_z*self.size_c #normalise by same number of elements as in reconstruction

        return BCE + KLD

    def model_train(self, epoch):
        # toggle model to train mode
        self.train()
        train_loss = 0

        for batch_idx, data in enumerate(self.train_loader):
            data = data.type(torch.FloatTensor) #numpy uses doubles, so just to be save

            if not car:
                data_window = torch.zeros([self.batch_size, self.size_c, 2*self.window_size, self.size_z])
                for i in range(self.batch_size): #number of samples we take from the same world
                    center = np.random.randint(0,self.size_x)
                    data_window[i,:,:,:] = self.window(data[0], center)
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
        if type(self.writer) is not str:
            self.writer.add_scalar('autoencoder training loss', train_loss / len(self.train_loader.dataset) , epoch * len(self.train_loader.dataset))

        if car:
            # visualization of latent space
            sample = Variable(torch.randn(self.batch_size, self.bottleneck))
            sample = self.decode(sample).cpu()

            save_image(sample.data.view(self.batch_size, self.size_c, self.size_x, self.size_z),
                #           'results/sample_' + str(epoch) + '.png')
                          'autoencoder/results/sample.png')
        else:
            sample = Variable(torch.randn(self.batch_size, self.bottleneck))
            sample = self.decode(sample).cpu()

            self.visualize(sample, 'autoencoder/results/sample.png')

    def model_test(self, epoch):
        # toggle model to test / inference mode
        self.eval()
        test_loss = 0

        # each data is of self.batch_size (default 128) samples
        for i, data in enumerate(self.test_loader):
            data = data.type(torch.FloatTensor) #numpy uses doubles, so just to be save

            if not car:
                data_window = torch.zeros([self.batch_size, self.size_c, 2*self.window_size, self.size_z])
                for i in range(self.batch_size): #number of samples we take from the same world
                    center = np.random.randint(0,self.size_x)
                    data_window[i,:,:,:] = self.window(data[0], center)
                data = data_window #to keep naming convention

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
                                'autoencoder/results/reconstruction.png', nrow=n)

            else:
                #validation
                #real = np.array(data[b]).transpose(1, -1, 0)
                #real = torch.tensor(real)
                #recon = recon_batch[b].detach().numpy().transpose(1, -1, 0)
                #recon = torch.tensor(recon)

                self.visualize(data, 'autoencoder/results/real.png')
                self.visualize(recon_batch, 'autoencoder/results/recon.png')

                # save
                for b in range(self.batch_size):
                    # compressed
                    comp = torch.cat([mu[b], logvar[b]])
                    torch.save(comp, 'data/test/tensor_comp/wind_map_comp' + str(i*self.batch_size+b).zfill(5) + '.pt')


        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def save_weights(self, path):
        torch.save(self.state_dict(), path + '/model.pt')

    def load_weights(self, path):
        self.load_state_dict(torch.load(path + '/model.pt'))
        self.eval()

    def window(self, data, center):
        start_x = int(max(center - self.window_size, 0))
        end_x = int(min(center + self.window_size, self.size_x))
        window = np.zeros((self.size_c,self.window_size*2,self.size_z))
        fill_in = data[:,start_x:end_x,:]
        # touching the left border
        if start_x == 0:
            window[:,self.window_size*2-end_x::,:] = fill_in
            for i in range(2*self.window_size-len(fill_in[0])):
                window[:,i,:] = fill_in[:,0,:]

        # touching the right border
        elif end_x == self.size_x:
            window[:,0:end_x-start_x,:] = fill_in
            for i in range(2*self.window_size-len(fill_in[0])):
                window[:,2*self.window_size-i-1,:] = fill_in[:,-1,:]

        # if not touching anything¨
        else:
            #print('no touch')
            window = fill_in

        window = torch.tensor(window)
        return window

    def visualize(self, data, path):
        for n in range(min(len(data),3)):
            mean_x = data[n][-3,:,:].detach().numpy()
            mean_z = data[n][-2,:,:].detach().numpy()
            sig_xz = data[n][-1,:,:].detach().numpy()

            size_x = len(mean_x)
            size_z = len(mean_x[0])

            z,x = np.meshgrid(np.arange(0, size_z, 1),np.arange(0, size_x, 1))
            fig, ax = plt.subplots(frameon=False, figsize=(size_x,size_z))
            ax.set_axis_off()
            ax.set_aspect(1)

            # standardise color map for sig value
            floor = 0
            ceil = 1
            sig_xz = np.maximum(sig_xz, floor)
            sig_xz = np.minimum(sig_xz, ceil)
            sig_xz -= floor
            sig_xz /= ceil
            cm = matplotlib.cm.viridis
            colors = cm(sig_xz).reshape(size_x*size_z,4)

            # generate quiver
            q = ax.quiver(x, z, mean_x, mean_z, color=colors, scale=1, scale_units='inches')
            plt.savefig(path)
            plt.close()
