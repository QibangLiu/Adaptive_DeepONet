# %%


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
# %%

Nx = 128
Ny = 128
m = Nx * Ny
fenics_data = scio.loadmat("../Adap_possion/TrainingData/poisson_gauss_cov20k.mat")

x_grid = fenics_data["x_grid"].astype(np.float32)  # shape (Ny, Nx)
y_grid = fenics_data["y_grid"].astype(np.float32)
source_terms_raw = fenics_data["source_terms"].astype(np.float32)  # shape (N, Ny, Nx)
source_terms_raw = source_terms_raw.reshape(-1, Nx * Ny)
solutions_raw = fenics_data["solutions"].astype(np.float32)  # shape (N, Ny, Nx)
solutions_raw = solutions_raw.reshape(-1, Nx * Ny)
# %%
source_mean= np.mean(solutions_raw, axis=1)
source_min= np.min(solutions_raw, axis=1)
source_max= np.max(solutions_raw, axis=1)
source_std = np.std(solutions_raw, axis=1)
# %%
plt.hist(source_mean, bins=100)
# %%
gap=1
idxs=np.argsort(source_mean)
fig=plt.figure()
ax=plt.subplot(1,1,1)
ax.plot(source_mean[idxs][::gap], label='mean')
ax.fill_between(range(len(source_mean[::gap])), (source_min[idxs])[::gap], (source_max[idxs])[::gap], color='gray', alpha=0.2)

# %%
gap=100
idxs=np.argsort(source_min)
plt.plot(source_min[idxs][::gap], label='min')
plt.plot(source_max[idxs][::gap], label='max')
plt.fill_between(range(len(source_mean[::gap])), (source_min[idxs])[::gap], (source_max[idxs])[::gap], color='gray', alpha=0.2)

plt.legend()
plt.xlabel('sample index')
plt.ylabel('source term max/min')
# %%
nr,nc=4,4
fig=plt.figure(figsize=(16,16))
for i in range(nr*nc):
    ax=fig.add_subplot(nr,nc,i+1)
    #im=ax.imshow(solutions_raw[i].reshape(Ny,Nx),cmap='jet')
    im=ax.contourf(x_grid, y_grid, solutions_raw[i].reshape(Ny,Nx), 20, cmap='jet')
    colorbar = ax.figure.colorbar(im, ax=ax)
    #ax.axis('off')
    ax.axis('equal')
    ax.get_tightbbox()
    #ax.set_title(f'{i}')
# %%
