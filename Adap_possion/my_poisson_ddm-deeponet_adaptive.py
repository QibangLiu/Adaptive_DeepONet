#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import Adaptive_DeepONet.Adap_possion.DeepONet_torch as don
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.io as scio
from deepxde import utils
import deepxde as dde
from deepxde.backend import tf
from myutils import find_checkpoint_2restore, EvaluateDerivatives, LaplaceOperator2D
import timeit
# dde.backend.set_default_backend("tensorflow")
# dde.config.set_default_float("float64")
# In[]
prefix_filebase = "/scratch/bblv/qibang/repository/Adap_data_driven_possion/saved_model"

str_k, str_c, str_dN, str_start, str_end = sys.argv[1:]


# In[3]:

Nx = 128
Ny = 128
m = Nx * Ny
# N number of samples 1000
# m number of points 40
# P number of output sensors (coordinates) 1600
# x_train is a tuple of u0(N,m) and output sensores, all coordinates xy_train_testing(P,2)
# y_train is target solutions (our s) u(N,P)

tf.keras.backend.clear_session()
# tf.keras.utils.set_random_seed(seed)
fenics_data = scio.loadmat(
    "/scratch/bblv/qibang/repository/Adap_data_driven_possion/TrainingData/poisson_gauss_cov.mat"
)

x_grid = fenics_data["x_grid"].astype(np.float32)  # shape (Ny, Nx)
y_grid = fenics_data["y_grid"].astype(np.float32)
source_terms_raw = fenics_data["source_terms"].astype(
    np.float32)  # shape (N, Ny, Nx)
source_terms_raw = source_terms_raw.reshape(-1, Nx * Ny)
solutions_raw = fenics_data["solutions"].astype(
    np.float32)  # shape (N, Ny, Nx)
solutions_raw = solutions_raw.reshape(-1, Nx * Ny)
scaler_source = 0.5 * (np.max(source_terms_raw) - np.min(source_terms_raw))
scaler_solution = 0.5 * (np.max(solutions_raw) - np.min(solutions_raw))
solutions = solutions_raw / scaler_solution
source_terms = source_terms_raw / scaler_source
u0_train = source_terms[:-500]
u0_testing = source_terms[-500:]
s_train = solutions[:-500]
s_testing = solutions[-500:]

u0_testing_raw = source_terms_raw[-500:]
u0_train_raw = source_terms_raw[:-500]
s_testing_raw = solutions_raw[-500:]
s_train_raw = solutions_raw[:-500]

xy_train_testing = np.concatenate(
    [x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)], axis=1
)

# %%

x_test = (u0_testing, xy_train_testing)
y_test = s_testing

# %%

data = None
model = don.DeepONetCartesianProd(
    [m, 100, 100, 100, 100, 100, 100],
    [2, 100, 100, 100, 100, 100, 100],
    {"branch": "relu", "trunk": "tanh"},
)
initial_weights = model.get_weights()
laplace_op = EvaluateDerivatives(model, LaplaceOperator2D)


def ErrorMeasure(X, rhs):
    lhs = -0.01 * (laplace_op.eval(X)) * scaler_solution
    lhs = np.squeeze(lhs)
    return np.linalg.norm(lhs - rhs, axis=1) / np.linalg.norm(rhs, axis=1)


# %%
numCase_init, numCase_add = int(str_dN), int(str_dN)
k = float(str_k)
c = float(str_c)

iter_start, iter_end = int(str_start), int(str_end)
project_name = (
    "adaptive_k" + str_k + "c" + str_c + "dN" + str_dN
)
all_data_idx = np.arange(len(u0_train))
filebase = os.path.join(prefix_filebase, project_name)
start_time = timeit.default_timer()

for iter in range(iter_start, iter_end):
    print("====iter=======:%d" % iter)
    pre_filebase = os.path.join(filebase, "iter" + str(iter - 1))
    current_filebase = os.path.join(filebase, "iter" + str(iter))
    os.makedirs(current_filebase, exist_ok=True)
    # restore_path = find_checkpoint_2restore(pre_filebase)
    if iter == 0:
        # np.random.seed(80)
        currTrainDataIDX = np.random.choice(
            a=all_data_idx, size=numCase_init, replace=False
        )
    else:
        pre_train_data_idx = np.genfromtxt(
            os.path.join(pre_filebase, "trainDataIDX.csv"), dtype=int, delimiter=","
        )
        # potential training data
        potential_train_data_idx = np.delete(all_data_idx, pre_train_data_idx)

        LR = ErrorMeasure(
            (u0_train[potential_train_data_idx], xy_train_testing),
            u0_train[potential_train_data_idx],
        )
        probility = np.power(LR, k) / np.power(LR, k).mean() + c
        probility_normalized = probility / np.sum(probility)
        new_training_data_idx = np.random.choice(
            a=potential_train_data_idx,
            size=numCase_add,
            replace=False,
            p=probility_normalized,
        )
        currTrainDataIDX = np.concatenate(
            (pre_train_data_idx, new_training_data_idx))

    np.savetxt(
        os.path.join(current_filebase, "trainDataIDX.csv"),
        currTrainDataIDX,
        fmt="%d",
        delimiter=",",
    )

    curr_u_train = u0_train[currTrainDataIDX]
    x_train = (curr_u_train, xy_train_testing)
    curr_y_train = s_train[currTrainDataIDX]
    data = don.TripleCartesianProd(
        x_train, curr_y_train, x_test, y_test, batch_size=64)
    optm = tf.keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(optimizer=optm, loss="mse")
    checkpoint_fname = os.path.join(current_filebase, "model.ckpt")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_fname,
        save_weights_only=True,
        monitor="val_loss",
        verbose=1,
        save_freq="epoch",
        save_best_only=True,
        mode="min",
    )
    model.set_weights(initial_weights)
    h = model.fit(
        data.train_dataset, validation_data=data.test_dataset, epochs=800, verbose=2,
        callbacks=[model_checkpoint]
    )
    model.save_history(filebase)
    model.load_weights(checkpoint_fname)
    y_pred = model.predict(data.test_x)
    error_test = np.linalg.norm(y_test - y_pred, axis=1) / np.linalg.norm(
        y_test, axis=1
    )
    np.savetxt(
        os.path.join(current_filebase, "TestL2Error.csv"),
        error_test,
        fmt="%.4e",
        delimiter=",",
    )
    stop_time = timeit.default_timer()
    print('training Run time so far: ', round(
        stop_time - start_time, 2), '(s)')
    # fig=plt.figure()
    # plt.hist(error_test)
    # plt.xlabel("L2 relative error")
    # plt.ylabel("frequency")
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["loss"], label="loss")
ax.plot(h["val_loss"], label="val_loss")
ax.legend()
ax.set_yscale("log")

# %%

# Plotting Results
# %%


error_s = error_test
sort_idx = np.argsort(error_s)
min_index = sort_idx[0]
max_index = sort_idx[-1]
median_index = sort_idx[len(sort_idx) // 2]

# Print the indexes
print("Index for minimum element:", min_index)
print("Index for maximum element:", max_index)
print("Index for median element:", median_index)

min_median_max_index = np.array([min_index, median_index, max_index])

y_pred = scaler_solution * y_pred
y_test = s_testing_raw
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

# %%

# %%
laplace_op_val = laplace_op.eval(
    (data.test_x[0][min_median_max_index], data.test_x[1]))
laplace_op_val = -0.01 * laplace_op_val

# %%
laplace_op = EvaluateDerivatives(model, LaplaceOperator2D)
# %%
laplace_op_val = laplace_op.eval(
    (data.test_x[0][min_median_max_index], data.test_x[1]))
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
    c1 = ax.contourf(x_grid, y_grid, u0_testing_raw[index].reshape(
        Ny, Nx), 20, cmap="jet")
    ax.set_title(r"Source Distrubution")
    cbar = fig.colorbar(c1, ax=ax)
    plt.tight_layout()
    ax = plt.subplot(nr, nc, nc*i+2)
    # py.figure(figsize = (14,7))
    c2 = ax.contourf(x_grid, y_grid, laplace_op_val[i].reshape(
        Ny, Nx), 20, cmap="jet")
    ax.set_title(r"$-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$")
    cbar = fig.colorbar(c2, ax=ax)
    plt.tight_layout()
