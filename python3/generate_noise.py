import torch
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2
import random
from pathlib import Path

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

def generate_noise(W_20, num, train_or_test):
    '''
    Prototyping Turbulent Wind Fields based on Spectral Domain Simulation
    Author: David Rohr, ASL, ETH Zurich, Switzerland, 2019
    Resources: - The Spatial Structure of Neutral Atmospheric Surface-Layer
              Turbulence, J. Mann, J. Fluid Mech., vol. 273, pp. 141-168, 1994
               - Wind Field Simulation, J. Mann, Prop. Engng. Mech,. vol.13
              No.4 pp.269-282, 1998
              - Simulation of Three-Dimensional Turbulent Velocity Fields,
              R. Frehlich & L.Cornman, J. of applied Meteorology, vol.40, 2000
              - Wikipedia: https://en.wikipedia.org/wiki/Continuous_gusts#Altitude_Dependence
    '''
    # set up folder StructurePath(yaml_p['data_path']).mkdir(parents=True, exist_ok=True)
    Path(yaml_p['noise_path'] + train_or_test).mkdir(parents=True, exist_ok=True)
    Path(yaml_p['noise_path'] + train_or_test + '/tensor_' + str(W_20)).mkdir(parents=True, exist_ok=True)

    # details from previous code
    check_statistics=False

    # the smallest element is of unit_z
    nx = int(np.round(yaml_p['size_x']*yaml_p['unit_xy']/yaml_p['unit_z']))
    ny = int(np.round(yaml_p['size_y']*yaml_p['unit_xy']/yaml_p['unit_z']))
    nz = int(np.round(yaml_p['size_z']))
    dx = yaml_p['unit_z']
    dy = yaml_p['unit_z']
    dz = yaml_p['unit_z']

    lambda_min = min([dx, dy, dz])
    lambda_min = max(lambda_min, 0.06) # minimal wavelength of turbulence to simulate, [m]  (min 6cm)

    x = np.arange(0, dx * (nx -1), dx).tolist()
    y = np.arange(0, dy * (ny -1), dy).tolist()
    z = np.arange(0, dz * (nz -1), dz).tolist()

    X, Y, Z = np.meshgrid(x, y, z)
    U = np.array(0 * X).astype(complex)
    V = np.array(0 * Y).astype(complex)
    W = np.array(0 * Z).astype(complex)

    # assemble spatial frequency components (wave vector)
    k_x = 2*np.pi/lambda_min/nx * np.array(np.arange(-(nx-1)/2, (nx+1)/2).tolist())
    k_y = 2*np.pi/lambda_min/ny * np.array(np.arange(-(ny-1)/2, (ny+1)/2).tolist())
    k_z = 2*np.pi/lambda_min/nz * np.array(np.arange(-(nz-1)/2, (nz+1)/2).tolist())

    # frequency spacing
    dk_x = k_x[1] - k_x[0]
    dk_y = k_y[1] - k_y[0]
    dk_z = k_z[1] - k_z[0]

    for n in range(num):
        ## Fourier Simulation

        # Create random phase for every frequency component
        xi = (np.random.randn(3, nx, ny, nz) + 1j*np.random.randn(3, nx, ny, nz))/np.sqrt(2)

        C_ij = np.zeros((3, 3, nx, ny, nz))
        # Phi_ij = np.zeros((3, 3, nx, ny, nz))
        # E_ij = np.zeros((nx, ny, nz))
        # E_sum = 0

        # sigma and L depend on height about ground
        for ikz in range(nz):
            print('generated ' + str(n) + ' of ' + str(num) + ' noise maps, next one at ' + str(int(ikz/nz*100)) + '%')

            h = ikz*dz/0.3048 #h is in ft
            if h < 1000: #low altitude
                L = h/(0.177 + 0.000823*h)**1.2
                sigma_w = 0.1*W_20
                sigma_uv = sigma_w/(0.177 + 0.000823*h)**0.4
                sigma = np.array([sigma_uv,sigma_uv,sigma_w])

            elif 1000 <= h < 2000: #low to medium altitude
                L_1000 = 1000/((0.177 + 0.000823*1000)**1.2)
                L_2000 = 2500
                L = L_1000 + (L_2000 - L_1000)*(h-1000)/(2000 - 1000)
                sigma_w = 0.1*W_20
                sigma_uv = sigma_w
                sigma = np.array([sigma_uv,sigma_uv,sigma_w])

            else: # medium and high altitude
             # The following values are taken from: MIL-STD-1797A (1990), Flying Qualities of Piloted Aircraft (PDF). U.S. Department of Defense, Figure 262 on page 673 and converted from [ft] to [m]
                L = L_2000
                if W_20 == 15:
                    h_list = [1600, 9200, 17600]
                    s_list = [0.1*W_20, 0.1*W_20, 0.914]
                elif W_20 == 30:
                    h_list = [2000, 11600, 43600]
                    s_list = [0.1*W_20, 0.1*W_20, 0.914]
                elif W_20 == 45:
                    h_list = [2800, 4400, 20000, 80000]
                    s_list = [0.1*W_20, 6.461, 6.461, 0.914]
                else:
                    print('ERROR: Please choose one of the following W_20 values: 15, 30, 45')
                sigma_w = np.interp(h, h_list, s_list)
                sigma_uv = sigma_w
                sigma = np.array([sigma_uv,sigma_uv,sigma_w])

            for ikx in range(nx):
                for iky in range(ny):
                    k = np.array([k_x[ikx], k_y[iky], k_z[ikz]])
                    k = np.transpose(k)
                    if 0 < np.linalg.norm(k) <= k_x[-1]:
                        # Phi_ij[:, :, ikx, iky, ikz] = spec_tens_iso_inc(k, L, sigma)
                        E = karman_E(k, L, sigma)
                        # E_ij[ikx, iky, ikz] = E
                        # E_sum = E_sum + E / (np.linalg.norm(k) ** 2 * 4 * np.pi) * dk_x * dk_y * dk_z

                        A_ij = np.sqrt(E / (4 * np.pi)) / (np.linalg.norm(k) ** 2) * np.array([[0, k[2], -k[1]],
                                                                                                     [- k[2], 0, k[0]],
                                                                                                     [k[1], -k[0], 0]])
                        C_ij[:, :, ikx, iky, ikz] = np.sqrt(dk_x * dk_y * dk_z) * np.array(A_ij)

        perc = 0
        N = nx * ny * nz

        # IFFT (inverse fast fourier transform)
        complex_field = np.zeros((3, nx, ny, nz), dtype=complex)
        for ikx in range(nx):
            for iky in range(ny):
                for ikz in range(nz):
                    complex_field[:, ikx, iky, ikz] = np.array(C_ij[:, :, ikx, iky, ikz]).dot(np.array(xi[:, ikx, iky, ikz]))

        U_c = np.squeeze(complex_field[0, :, :, :])
        V_c = np.squeeze(complex_field[1, :, :, :])
        W_c = np.squeeze(complex_field[2, :, :, :])

        U_c = np.roll(np.roll(np.roll(U_c, int(-(nx-1)/2), axis=0), int(-(ny-1)/2), axis=1), int(-(nz-1)/2), axis=2)
        V_c = np.roll(np.roll(np.roll(V_c, int(-(nx-1)/2), axis=0), int(-(ny-1)/2), axis=1), int(-(nz-1)/2), axis=2)
        W_c = np.roll(np.roll(np.roll(W_c, int(-(nx-1)/2), axis=0), int(-(ny-1)/2), axis=1), int(-(nz-1)/2), axis=2)

        U2 = np.fft.ifft(np.fft.ifft(np.fft.ifft(U_c, n=None, axis=0), n=None, axis=1), n=None, axis=2) * nx*ny*nz
        V2 = np.fft.ifft(np.fft.ifft(np.fft.ifft(V_c, n=None, axis=0), n=None, axis=1), n=None, axis=2) * nx*ny*nz
        W2 = np.fft.ifft(np.fft.ifft(np.fft.ifft(W_c, n=None, axis=0), n=None, axis=1), n=None, axis=2) * nx*ny*nz

        x2 = np.linspace(0, (nx-1)*lambda_min, nx)
        y2 = np.linspace(0, (ny-1)*lambda_min, ny)
        z2 = np.linspace(0, (nz-1)*lambda_min, nz)

        X2, Y2, Z2 = np.meshgrid(x2, y2, z2)

        if check_statistics:
            prsv_pred = 0
            for ikx in range(nx):
                for iky in range(ny):
                    prsv_pred = prsv_pred + np.array(np.transpose(complex_field[:, ikx, iky, ikz])).dot(complex_field[:, ikx, iky, ikz])

            # check statistics of field
            # turbulent component standard deviation
            std_real = [np.std(np.reshape(np.real(U2), (1, -1))),
                        np.std(np.reshape(np.real(V2), (1, -1))),
                        np.std(np.reshape(np.real(W2), (1, -1)))]
            # turbulent kinetic energy (of real valued field)
            tke_real = 0.5 * np.sum(np.multiply(std_real, std_real))
            # turbulent kinetic energy (of complex valued field)
            tke_complex = 0.5 / (nx*ny*nz) * (np.sum(np.reshape(np.multiply(np.abs(U2), np.abs(U2)), (1, -1)))
                                                        + np.sum(np.reshape(np.multiply(np.abs(V2), np.abs(V2)), (1, -1)))
                                                        + np.sum(np.reshape(np.multiply(np.abs(W2), np.abs(W2)), (1, -1))))
            prsv = 1 / (nx*ny*nz) * (np.sum(np.reshape(np.multiply(np.abs(U2), np.abs(U2)), (1, -1)))
                                              + np.sum(np.reshape(np.multiply(np.abs(V2), np.abs(V2)), (1, -1)))
                                              + np.sum(np.reshape(np.multiply(np.abs(W2), np.abs(W2)), (1, -1))))

        U = U2
        V = V2
        W = W2
        X = X2
        Y = Y2
        Z = Z2

        # turbulent velocity field matrix
        UVW = np.stack((np.real(U), np.real(V), np.real(W)), axis=0)
        XYZ = np.stack((X, Y, Z), axis=0)

        # save only at resolution that it will be actually used
        size_c = len(UVW)
        size_x = int(len(UVW[0])/2)
        size_y = int(len(UVW[0][0])/2)
        size_z = int(len(UVW[0][0][0]))

        UVW_interp = np.zeros((size_c, yaml_p['size_x'], yaml_p['size_y'], yaml_p['size_z']))
        for i in range(size_c):
            for j in range(yaml_p['size_x']):
                for k in range(yaml_p['size_y']):
                    UVW_interp[i,j,k] = UVW[i,int(yaml_p['size_x']/size_x)*j,int(yaml_p['size_y']/size_y)*k]
                    print(int(yaml_p['size_x']/size_x)*j)

        # save
        plot(UVW_interp,W_20,min(nx,ny,nz))
        torch.save(UVW_interp, yaml_p['noise_path'] + train_or_test + '/tensor_' + str(W_20) + '/noise_map' + str(n).zfill(5) + '.pt')

def karman_E(k, L, sigma):
    E = 1.4528 * L * sigma**2 * (L*np.linalg.norm(k))**4 / (1+(L*np.linalg.norm(k))**2)**(17/6)
    return E

def plot(UVW,t,N):
    U = UVW[0]
    V = UVW[1]
    W = UVW[2]

    fig, axs = plt.subplots(4)

    vmin = np.min(UVW)
    vmax = np.max(UVW)

    #imshow
    #img = axs[0].imshow(U[0].T, vmin=vmin, vmax=vmax, interpolation='bilinear')
    #axs[1].imshow(V[0].T, vmin=vmin, vmax=vmax, interpolation='bilinear')
    #axs[2].imshow(W[0].T, vmin=vmin, vmax=vmax, interpolation='bilinear')
    img = axs[0].imshow(U[0].T, vmin=vmin, vmax=vmax)
    axs[1].imshow(V[0].T, vmin=vmin, vmax=vmax)
    axs[2].imshow(W[0].T, vmin=vmin, vmax=vmax)

    #set xlabel
    axs[0].set_xlabel('U: y @x=0')
    axs[0].set_ylabel('U: z @x=0')
    axs[1].set_xlabel('V: y @x=0')
    axs[1].set_ylabel('V: z @x=0')
    axs[2].set_xlabel('W: y @x=0')
    axs[2].set_ylabel('W: z @x=0')

    axs[0].set_aspect(1)
    axs[1].set_aspect(1)
    axs[2].set_aspect(1)

    axs[3].axis('off')
    cbar = plt.colorbar(img, ax=axs[3], orientation='horizontal')

    axs[0].set_title(str(int(t/N*100)) + '%')

    plt.tight_layout()
    Path('temp').mkdir(parents=True, exist_ok=True)
    plt.savefig('temp/t_' + str(t).zfill(5) + '.png', dpi=1000)
    plt.close()


for W_20 in [15]:
    generate_noise(W_20, 50, 'train')
