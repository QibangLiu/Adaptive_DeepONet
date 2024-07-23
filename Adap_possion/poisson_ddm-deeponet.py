#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.io as scio
from deepxde import utils
import deepxde as dde
import torch
#from deepxde.backend import tf
#from myutils import find_checkpoint_2restore, EvaluateDerivatives, LaplaceOperator2D
import DeepONet
# dde.backend.set_default_backend("tensorflow")
# dde.config.set_default_float("float64")
# In[]
filebase = (
    "/scratch/bbpq/qibang/repository/Adap_data_driven_possion/saved_model/FenicsData"
)
os.makedirs(filebase, exist_ok=True)
restore_path=None
#restore_path = find_checkpoint_2restore(filebase)
# In[3]:

Nx = 128
Ny = 128
m = Nx * Ny
###  N number of samples 1000
###  m number of points 40
###  P number of output sensors (coordinates) 1600
### x_train is a tuple of u0(N,m) and output sensores, all coordinates xy_train_testing(P,2)
### y_train is target solutions (our s) u(N,P)

# tf.keras.utils.set_random_seed(seed)

# fenics_data = scio.loadmat(
#     "/scratch/bbpq/qibang/repository/Adap_data_driven_possion/TrainingData/poisson.mat"
# )

# x_grid = fenics_data["x_grid"].astype(np.float32)  # shape (Ny, Nx)
# y_grid = fenics_data["y_grid"].astype(np.float32)
# source_terms = fenics_data["source_terms"].astype(np.float32)  # shape (N, Ny, Nx)
# source_terms = source_terms.reshape(-1, Nx * Ny)
# solutions = fenics_data["solutions"].astype(np.float32)  # shape (N, Ny, Nx)
# solutions = solutions.reshape(-1, Nx * Ny)
# u0_train = source_terms[:5000]
# u0_testing = source_terms[5000:]
# s_train = solutions[:5000]
# s_testing = solutions[5000:]

# xy_train_testing = np.concatenate(
#     [x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)], axis=1
# )

fenics_data = scio.loadmat(
    "/scratch/bblv/qibang/repository/Adap_data_driven_possion/TrainingData/poisson_gauss_cov.mat"
)

x_grid = fenics_data["x_grid"].astype(np.float32)  # shape (Ny, Nx)
y_grid = fenics_data["y_grid"].astype(np.float32)
source_terms_raw = fenics_data["source_terms"].astype(np.float32)  # shape (N, Ny, Nx)
source_terms_raw = source_terms_raw.reshape(-1, Nx * Ny)
solutions_raw = fenics_data["solutions"].astype(np.float32)  # shape (N, Ny, Nx)
solutions_raw = solutions_raw.reshape(-1, Nx * Ny)
scaler_source = 0.5 * (np.max(source_terms_raw) - np.min(source_terms_raw))
scaler_solution = 0.5 * (np.max(solutions_raw) - np.min(solutions_raw))
solutions = solutions_raw / scaler_solution
source_terms = source_terms_raw / scaler_source
u0_train = source_terms[:5000]
u0_testing = source_terms[5000:6000]
s_train = solutions[:5000]
s_testing = solutions[5000:6000]

u0_testing_raw = source_terms_raw[5000:6000]
u0_train_raw = source_terms_raw[:5000]
s_testing_raw = solutions_raw[5000:6000]
s_train_raw = solutions_raw[:5000]

xy_train_testing = np.concatenate(
    [x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)], axis=1
)

print("u0_train.shape = ", u0_train.shape)
print("type of u0_train = ", type(u0_train))
print("u0_testing.shape = ", u0_testing.shape)
print("s_train.shape = ", s_train.shape)
print("s_testing.shape = ", s_testing.shape)
print("xy_train_testing.shape", xy_train_testing.shape)
# %%
x_train = (u0_train, xy_train_testing)
y_train = s_train
x_test = (u0_testing, xy_train_testing)
y_test = s_testing

# %%
# x_train, y_train = get_data("train_IC2.npz")
# x_test, y_test = get_data("test_IC2.npz")


data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)
# %%
net = dde.maps.DeepONetCartesianProd(
    [m, 100, 100, 100, 100, 100, 100],
    [2, 100, 100, 100, 100, 100, 100],
    {"branch": "relu", "trunk": "tanh"},
    "Glorot normal",
)

model = dde.Model(data, net)
model.compile(
    "adam",
    lr=1e-3,
    decay=("inverse time", 1, 1e-4),
    metrics=["mean l2 relative error"],
)
# IC1
# losshistory, train_state = model.train(epochs=100000, batch_size=None)
# IC2

check_point_filename = os.path.join(filebase, "model.ckpt")
checkpointer = dde.callbacks.ModelCheckpoint(
    check_point_filename, verbose=1, save_better_only=True
)
if restore_path is not None:
    model.restore(restore_path)
losshistory, train_state = model.train(
    iterations=100000, batch_size=128, callbacks=[checkpointer]
)

y_pred_p = model.predict(data.test_x)
y_pred = scaler_solution * y_pred_p
y_test = s_testing_raw
print("y_pred.shape =", y_pred.shape)
##np.savetxt("y_pred_deeponet.dat", y_pred[0].reshape(nt, nx))
##np.savetxt("y_true_deeponet.dat", data.test_y[0].reshape(nt, nx))
##np.savetxt("y_error_deeponet.dat", (y_pred[0] - data.test_y[0]).reshape(nt, nx))

# %%

error_s = []
for i in range(len(y_test)):
    error_s_tmp = np.linalg.norm(y_test[i] - y_pred[i]) / np.linalg.norm(y_test[i])
    error_s.append(error_s_tmp)
error_s = np.stack(error_s)
_=plt.hist(error_s)


# %%

#### Plotting Results
# %%
import matplotlib.pyplot as plt
import pylab as py


# Defining custom plotting functions
def my_contourf(x, y, F, ttl, vmin=None, vmax=None):
    cnt = py.contourf(x, y, F, 12, cmap="jet", vmin=vmin, vmax=vmax)
    py.colorbar()
    py.xlabel("x")
    py.ylabel("y")
    py.title(ttl)
    return 0




min_index = np.argmin(error_s)
max_index = np.argmax(error_s)
median_index = np.median(error_s).astype(int)

# Print the indexes
print("Index for minimum element:", min_index)
print("Index for maximum element:", max_index)
print("Index for median element:", median_index)


min_median_max_index = np.array([min_index, median_index, max_index])


for index in min_median_max_index:

    u0_testing_nx_ny = u0_testing[index].reshape(Ny, Nx)
    s_testing_nx_ny = y_test[index].reshape(Ny, Nx)
    s_pred_nx_ny = y_pred[index].reshape(Ny, Nx)
    vmin,vmax=np.min(s_testing_nx_ny),np.max(s_testing_nx_ny)
    fig = plt.figure(figsize=(18, 5))
    ax=plt.subplot(1, 3, 1)
    # py.figure(figsize = (14,7))
    my_contourf(x_grid, y_grid, u0_testing_nx_ny, r"Source Distrubution",vmin=vmin, vmax=vmax)
    plt.tight_layout()
    ax=plt.subplot(1, 3, 2)
    # py.figure(figsize = (14,7))
    my_contourf(x_grid, y_grid, s_testing_nx_ny, r"Reference Solution",vmin=vmin, vmax=vmax)
    plt.tight_layout()
    ax=plt.subplot(1, 3, 3)
    # py.figure(figsize = (14,7))
    my_contourf(x_grid, y_grid, s_pred_nx_ny, r"Predicted Solution",vmin=vmin, vmax=vmax)
    plt.tight_layout()

    # if index == min_index:
    #     plt.savefig("temperature_min_error_5000_3.jpg", dpi=300)
    # if index == median_index:
    #     plt.savefig("temperature_median_error_5000_3.jpg", dpi=300)
    # if index == max_index:
    #     plt.savefig("temperature_max_error_5000_3.jpg", dpi=300)
    # plt.savefig("temperature_sample{}_5000_3.jpg".format(index), dpi=300)
    # plt.show()



# %%
def LaplaceOperator2D(x, y,aux=None):
    dy_dx = torch.autograd.grad(
        outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]
    dydx, dydy = dy_dx[:, 0:1], dy_dx[:, 1:2]
    dydx2 = torch.autograd.grad(
        dydx,
        x,
        grad_outputs=torch.ones_like(dydx),
        create_graph=True,
    )[0][:, 0:1]
    dydy2 = torch.autograd.grad(
        dydy, x, grad_outputs=torch.ones_like(dydy), create_graph=True
    )[0][:, 1:2]

    return  (dydx2 + dydy2) 


laplace_op = DeepONet.EvaluateDeepONetPDEs(model.net, LaplaceOperator2D)
# %%
x_plot=x_test
u0_plot_raw=u0_testing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_branch, input_trunk = x_plot[0], x_plot[1]
input_branch = torch.tensor(input_branch).to(device)
input_trunk = torch.tensor(input_trunk,requires_grad=True).to(device)

laplace_op_val = laplace_op((input_branch[min_median_max_index], input_trunk))
laplace_op_val = -0.01 *scaler_solution* laplace_op_val.detach().cpu().numpy()

# %%
nr, nc = 3, 2
i = 0
fig = plt.figure(figsize=(8, 10))

for i, index in enumerate(min_median_max_index):

    vmin = np.min(u0_plot_raw[index])
    vmax = np.max(u0_plot_raw[index])

    ax = plt.subplot(nr, nc, nc * i + 1)
    # py.figure(figsize = (14,7))
    c1 = ax.contourf(
        x_grid, y_grid, u0_plot_raw[index].reshape(Ny, Nx), 20, cmap="jet"
    )
    ax.set_title(r"Source Distrubution")
    cbar = fig.colorbar(c1, ax=ax)
    plt.tight_layout()
    ax = plt.subplot(nr, nc, nc * i + 2)
    # py.figure(figsize = (14,7))
    c2 = ax.contourf(x_grid, y_grid, laplace_op_val[i].reshape(Ny, Nx), 20, cmap="jet")
    ax.set_title(r"$-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$")
    cbar = fig.colorbar(c2, ax=ax)
    plt.tight_layout()
