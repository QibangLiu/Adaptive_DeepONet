#!/usr/bin/env python
# coding: utf-8

# In[1]:

import DeepONet as don
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.io as scio
import tensorflow as tf
from tensorflow import keras
from myutils import find_checkpoint_2restore, EvaluateDerivatives, LaplaceOperator2D
from myutils import (
    find_checkpoint_2restore,
    EvaluateDerivatives,
    LaplaceOperator2D,
    UNET,
)

# dde.backend.set_default_backend("tensorflow")
# dde.config.set_default_float("float64")
# In[]
filebase = (
    "/scratch/bblv/qibang/repository/Adap_data_driven_possion/saved_model/FenicsData"
)
os.makedirs(filebase, exist_ok=True)

restore_path = find_checkpoint_2restore(filebase)
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
trunk_model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(
            100, input_shape=(2,), activation="tanh"
        ),  # Flatten the input
        tf.keras.layers.Dense(
            100, activation="tanh"
        ),  # First dense layer with ReLU activation
        tf.keras.layers.Dense(
            100, activation="tanh"
        ),  # Dropout layer to prevent overfitting
        tf.keras.layers.Dense(100, activation="tanh"),
        tf.keras.layers.Dense(100, activation="tanh"),
        tf.keras.layers.Dense(100),
    ]
)

# branch_model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(100,input_shape=(m, ),activation='tanh'),  # Flatten the input
#     tf.keras.layers.Dense(100, activation='tanh'),  # First dense layer with ReLU activation
#     tf.keras.layers.Dense(100, activation='tanh'),                   # Dropout layer to prevent overfitting
#     tf.keras.layers.Dense(100, activation='tanh'),
#     tf.keras.layers.Dense(100, activation='tanh'),
#     tf.keras.layers.Dense(100)
# ])

branch_model = UNET(
    img_size=(128, 128),
    output_size=100,
    img_channels=1,
    widths=[2, 4, 8, 16],
    has_attention=[False, False, True, True],
    num_res_blocks=1,
    norm_groups=2,
)
# %%
data = don.TripleCartesianProd(x_train, y_train, x_test, y_test, batch_size=64)

model = don.DeepONetCartesianProd(
    [branch_model], [2, 100, 100, 100, 100, 100, 100], keras.activations.swish
)
optm = tf.keras.optimizers.Adam(learning_rate=5e-4)

model.compile(optimizer=optm, loss="mse")

# keras.backend.set_value(model.optimizer.lr, 5e-4)
# %%
# x_train, y_train = get_data("train_IC2.npz")
# x_test, y_test = get_data("test_IC2.npz")
checkpoint_fname = os.path.join(filebase, "model.ckpt")
if os.path.exists(checkpoint_fname):
    model.load_weights(checkpoint_fname)
    model.load_history(filebase)
model_checkpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_fname,
    save_weights_only=True,
    monitor="loss",
    verbose=1,
    save_freq="epoch",
    save_best_only=True,
    mode="min",
)
h = model.fit(
    data.train_dataset, validation_data=data.test_dataset, epochs=200, verbose=2,
    callbacks=[model_checkpoint]
)
model.save_history(filebase)
model.summary()
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["loss"], label="loss")
ax.plot(h["val_loss"], label="val_loss")
ax.legend()
ax.set_yscale("log")
# %%

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
fig = plt.figure()
_ = plt.hist(error_s)


# %%

#### Plotting Results
# %%
import matplotlib.pyplot as plt
import pylab as py
import matplotlib.colors as mcolors


# Defining custom plotting functions
def my_contourf(x, y, F, ttl, vmin=None, vmax=None):
    cnt = py.contourf(
        x, y, F, 20, cmap="jet", norm=mcolors.Normalize(vmin=vmin, vmax=vmax)
    )
    # py.colorbar()
    py.xlabel("x")
    py.ylabel("y")
    py.title(ttl)
    return 0


# %%
sort_idx = np.argsort(error_s)
min_index = sort_idx[0]
max_index = sort_idx[-1]
median_index = sort_idx[len(sort_idx) // 2]

# Print the indexes
print("Index for minimum element:", min_index)
print("Index for maximum element:", max_index)
print("Index for median element:", median_index)


min_median_max_index = np.array([min_index, median_index, max_index])

nr, nc = 3, 3
fig = plt.figure(figsize=(18, 15))
for i, index in enumerate(min_median_max_index):

    u0_testing_nx_ny = u0_testing_raw[index].reshape(Ny, Nx)
    s_testing_nx_ny = y_test[index].reshape(Ny, Nx)
    s_pred_nx_ny = y_pred[index].reshape(Ny, Nx)
    vmin = min(s_testing_nx_ny.min(), s_pred_nx_ny.min())
    vmax = max(s_testing_nx_ny.max(), s_pred_nx_ny.max())
    
    ax = plt.subplot(nr, nc, nc*i+1)
    # py.figure(figsize = (14,7))
    c = ax.contourf(x_grid, y_grid, u0_testing_nx_ny, 20, cmap="jet")
    ax.set_title(r"Source Distrubution")
    plt.colorbar(c)
    plt.tight_layout()
    ax = plt.subplot(nr, nc, nc*i+2)
    # py.figure(figsize = (14,7))
    c1 = ax.contourf(x_grid, y_grid, s_testing_nx_ny, 20, cmap="jet")
    ax.set_title(r"Reference Solution")
    cbar = fig.colorbar(c1, ax=ax)
    plt.tight_layout()
    ax = plt.subplot(nr, nc, nc*i+3)
    # py.figure(figsize = (14,7))
    c2 = ax.contourf(x_grid, y_grid, s_pred_nx_ny, 20, cmap="jet")
    ax.set_title(r"Predicted Solution")
    cbar = fig.colorbar(c1, ax=ax)
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
laplace_op = EvaluateDerivatives(model, LaplaceOperator2D)
# %%
laplace_op_val = laplace_op.eval((data.test_x[0][min_median_max_index], data.test_x[1]))
laplace_op_val = -0.01 * laplace_op_val * scaler_solution
# %%
nr, nc = 3, 2
i = 0
fig = plt.figure(figsize=(8, 10))

for i, index in enumerate(min_median_max_index):

    vmin = np.min(u0_testing_raw[index])
    vmax = np.max(u0_testing_raw[index])
        
    ax = plt.subplot(nr, nc, nc*i+1)
    # py.figure(figsize = (14,7))
    c1 = ax.contourf(x_grid, y_grid, u0_testing_raw[index].reshape(Ny, Nx), 20, cmap="jet")
    ax.set_title(r"Source Distrubution")
    cbar = fig.colorbar(c1, ax=ax)
    plt.tight_layout()
    ax = plt.subplot(nr, nc, nc*i+2)
    # py.figure(figsize = (14,7))
    c2 = ax.contourf(x_grid, y_grid, laplace_op_val[i].reshape(Ny, Nx), 20, cmap="jet")
    ax.set_title(r"$-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$")
    cbar = fig.colorbar(c2, ax=ax)
    plt.tight_layout()
  


# %%
