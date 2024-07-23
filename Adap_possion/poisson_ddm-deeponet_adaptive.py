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
from deepxde.backend import tf
from myutils import find_checkpoint_2restore, EvaluateDerivatives, LaplaceOperator2D

# dde.backend.set_default_backend("tensorflow")
# dde.config.set_default_float("float64")
# In[]
prefix_filebase = "/scratch/bbpq/qibang/repository/Adap_data_driven_possion/saved_model"

# In[3]:

Nx = 128
Ny = 128
m = Nx * Ny
###  N number of samples 1000
###  m number of points 40
###  P number of output sensors (coordinates) 1600
### x_train is a tuple of u0(N,m) and output sensores, all coordinates xy_train_testing(P,2)
### y_train is target solutions (our s) u(N,P)

tf.keras.backend.clear_session()
# tf.keras.utils.set_random_seed(seed)
fenics_data = scio.loadmat(
    "/scratch/bbpq/qibang/repository/Adap_data_driven_possion/TrainingData/poisson.mat"
)

x_grid = fenics_data["x_grid"].astype(np.float32)  # shape (Ny, Nx)
y_grid = fenics_data["y_grid"].astype(np.float32)
source_terms = fenics_data["source_terms"].astype(np.float32)  # shape (N, Ny, Nx)
source_terms = source_terms.reshape(-1, Nx * Ny)
solutions = fenics_data["solutions"].astype(np.float32)  # shape (N, Ny, Nx)
solutions = solutions.reshape(-1, Nx * Ny)
u0_train = source_terms[:5500]
u0_testing = source_terms[5500:]
s_train = solutions[:5500]
s_testing = solutions[5500:]

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

x_test = (u0_testing, xy_train_testing)
y_test = s_testing

# %%
net = dde.maps.DeepONetCartesianProd(
    [m, 100, 100, 100, 100, 100, 100],
    [2, 100, 100, 100, 100, 100, 100],
    {"branch": "relu", "trunk": "tanh"},
    "Glorot normal",
)
data = None
model = dde.Model(data, net)
laplace_op = EvaluateDerivatives(model, LaplaceOperator2D)


def ErrorMeasure(X, rhs):
    lhs = -0.01 * (laplace_op.eval(X))
    lhs = np.squeeze(lhs)
    return np.linalg.norm(lhs - rhs, axis=1) / np.linalg.norm(rhs, axis=1)


# %%
numCase_init, numCase_add = 200, 200
iter_start, iter_end = 14, 15
k = 2
c = 0
project_name = (
    "adaptive_k" + str(int(k * 100)) + "c" + str(int(c * 100)) + "dN" + str(numCase_add)
)
all_data_idx = np.arange(len(u0_train))

for iter in range(iter_start, iter_end):
    print("====iter=======:%d" % iter)
    pre_filebase = os.path.join(prefix_filebase, project_name, "iter" + str(iter - 1))
    current_filebase = os.path.join(prefix_filebase, project_name, "iter" + str(iter))
    os.makedirs(current_filebase, exist_ok=True)
    restore_path = find_checkpoint_2restore(pre_filebase)
    if iter == 0:
        #np.random.seed(80)
        currTrainDataIDX = np.random.choice(
            a=all_data_idx, size=numCase_init, replace=False
        )
    else:
        pre_train_data_idx = np.genfromtxt(
            os.path.join(pre_filebase, "trainDataIDX.csv"), dtype=int, delimiter=","
        )
        # potential training data
        potential_train_data_idx = np.delete(all_data_idx, pre_train_data_idx)
        if restore_path is not None:
            model.restore(restore_path)
        LR = ErrorMeasure(
            (u0_train[potential_train_data_idx], xy_train_testing),
            s_train[potential_train_data_idx],
        )
        probility = np.power(LR, k) / np.power(LR, k).mean() + c
        probility_normalized = probility / np.sum(probility)
        new_training_data_idx = np.random.choice(
            a=potential_train_data_idx,
            size=numCase_add,
            replace=False,
            p=probility_normalized,
        )
        currTrainDataIDX = np.concatenate((pre_train_data_idx, new_training_data_idx))
    np.savetxt(
        os.path.join(current_filebase, "trainDataIDX.csv"),
        currTrainDataIDX,
        fmt="%d",
        delimiter=",",
    )

    curr_u_train = u0_train[currTrainDataIDX]
    x_train = (curr_u_train, xy_train_testing)
    curr_y_train = s_train[currTrainDataIDX]
    data = dde.data.TripleCartesianProd(x_train, curr_y_train, x_test, y_test)
    model.data = data
    model.compile(
        "adam",
        lr=1e-3,
        metrics=["mean l2 relative error"],
    )
    check_point_filename = os.path.join(current_filebase, "model.ckpt")
    checkpointer = dde.callbacks.ModelCheckpoint(
        check_point_filename, verbose=1, save_better_only=True
    )
    
    losshistory, train_state = model.train(
        iterations=20000, batch_size=64, callbacks=[checkpointer]
    )
    dde.saveplot(
        losshistory, train_state, issave=True, isplot=True, output_dir=current_filebase
    )
    y_pred = model.predict(data.test_x)
    print("y_pred.shape =", y_pred.shape)
    error_test = np.linalg.norm(y_test - y_pred, axis=1) / np.linalg.norm(
        y_test, axis=1
    )
    np.savetxt(
        os.path.join(current_filebase, "TestL2Error.csv"),
        error_test,
        fmt="%.4e",
        delimiter=",",
    )
    plt.hist(error_test)
    plt.xlabel('L2 relative error')
    plt.ylabel('frequency')
# IC1
# losshistory, train_state = model.train(epochs=100000, batch_size=None)
# IC2


##np.savetxt("y_pred_deeponet.dat", y_pred[0].reshape(nt, nx))
##np.savetxt("y_true_deeponet.dat", data.test_y[0].reshape(nt, nx))
##np.savetxt("y_error_deeponet.dat", (y_pred[0] - data.test_y[0]).reshape(nt, nx))


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


error_s = error_test

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
    vmin, vmax = np.min(s_testing_nx_ny), np.max(s_testing_nx_ny)
    fig = plt.figure(figsize=(18, 5))
    ax = plt.subplot(1, 3, 1)
    # py.figure(figsize = (14,7))
    my_contourf(
        x_grid, y_grid, u0_testing_nx_ny, r"Source Distrubution", vmin=vmin, vmax=vmax
    )
    plt.tight_layout()
    ax = plt.subplot(1, 3, 2)
    # py.figure(figsize = (14,7))
    my_contourf(
        x_grid, y_grid, s_testing_nx_ny, r"Reference Solution", vmin=vmin, vmax=vmax
    )
    plt.tight_layout()
    ax = plt.subplot(1, 3, 3)
    # py.figure(figsize = (14,7))
    my_contourf(
        x_grid, y_grid, s_pred_nx_ny, r"Predicted Solution", vmin=vmin, vmax=vmax
    )
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

# %%
laplace_op_val = laplace_op.eval((data.test_x[0][min_median_max_index], data.test_x[1]))
laplace_op_val = -0.01 * laplace_op_val

# %%
nr, nc = 3, 2
i = 0
fig = plt.figure(figsize=(8, 10))
ax = plt.subplot(nr, nc, 2 * i + 1)
# ax.contourf(x,y,data.train_x[0][min_median_max_index][i].reshape(Nx,Ny),12,cmap = 'jet')
vmin = np.min(data.test_x[0][min_median_max_index][i])
vmax = np.max(data.test_x[0][min_median_max_index][i])
my_contourf(
    x_grid,
    y_grid,
    data.test_x[0][min_median_max_index][i].reshape(Ny, Nx),
    r"Source Distrubution",
    vmin=vmin,
    vmax=vmax,
)
plt.tight_layout()
ax = plt.subplot(nr, nc, 2 * i + 2)
# ax.contourf(x,y,laplace_op_val[i].reshape(Nx,Ny),12,cmap = 'jet')
my_contourf(
    x_grid,
    y_grid,
    laplace_op_val[i].reshape(Ny, Nx),
    r"$-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$",
    vmin=vmin,
    vmax=vmax,
)
plt.tight_layout()
i = 1
vmin = np.min(data.test_x[0][min_median_max_index][i])
vmax = np.max(data.test_x[0][min_median_max_index][i])
ax = plt.subplot(nr, nc, 2 * i + 1)
# ax.contourf(x,y,data.train_x[0][min_median_max_index][i].reshape(Nx,Ny),12,cmap = 'jet')
my_contourf(
    x_grid,
    y_grid,
    data.test_x[0][min_median_max_index][i].reshape(Ny, Nx),
    r"Source Distrubution",
    vmin=vmin,
    vmax=vmax,
)
plt.tight_layout()
ax = plt.subplot(nr, nc, 2 * i + 2)
my_contourf(
    x_grid,
    y_grid,
    laplace_op_val[i].reshape(Ny, Nx),
    r"$-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$",
    vmin=vmin,
    vmax=vmax,
)
plt.tight_layout()
i = 2
vmin = np.min(data.test_x[0][min_median_max_index][i])
vmax = np.max(data.test_x[0][min_median_max_index][i])
ax = plt.subplot(nr, nc, 2 * i + 1)
# ax.contourf(x,y,data.train_x[0][min_median_max_index][i].reshape(Nx,Ny),12,cmap = 'jet')
my_contourf(
    x_grid,
    y_grid,
    data.test_x[0][min_median_max_index][i].reshape(Ny, Nx),
    r"Source Distrubution",
    vmin=vmin,
    vmax=vmax,
)
plt.tight_layout()
ax = plt.subplot(nr, nc, 2 * i + 2)
# ax.contourf(x,y,laplace_op_val[i].reshape(Nx,Ny),12,cmap = 'jet')
my_contourf(
    x_grid,
    y_grid,
    laplace_op_val[i].reshape(Ny, Nx),
    r"$-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$",
    vmin=vmin,
    vmax=vmax,
)
plt.tight_layout()
# %%
