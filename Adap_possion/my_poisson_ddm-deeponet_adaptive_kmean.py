#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import DeepONet_tf as don
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.io as scio
from deepxde import utils
import deepxde as dde
from deepxde.backend import tf
from myutils import find_checkpoint_2restore, EvaluateDerivatives, LaplaceOperator2D
import timeit
from sklearn.cluster import KMeans
# dde.backend.set_default_backend("tensorflow")
# dde.config.set_default_float("float64")
# In[]
prefix_filebase = "./saved_model"
method, str_dN, str_start, str_end = sys.argv[1:5]
if method == 'PDF':
    str_k, str_c = sys.argv[5:-1]
else:
    str_k, str_c = '0', "0"
str_caseID=sys.argv[-1]
##%run my_poisson_ddm-deeponet_adaptive_kmean.py 'PDF'  '400'  '0' '2' '1' '1' '0'
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
    "./TrainingData/poisson_gauss_cov20k.mat"
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
u0_train = source_terms[:-1000]
u0_testing = source_terms[-1000:]
s_train = solutions[:-1000]
s_testing = solutions[-1000:]

u0_testing_raw = source_terms_raw[-1000:]
u0_train_raw = source_terms_raw[:-1000]
s_testing_raw = solutions_raw[-1000:]
s_train_raw = solutions_raw[:-1000]

xy_train_testing = np.concatenate(
    [x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)], axis=1
)

y_kmeans = np.genfromtxt(
    './TrainingData/poisson_gauss_cov_kmean100.txt', dtype=int, delimiter=","
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


def get_cluster_index(clusters_all, potential_idx, n_clusters):
    cluster_indices = {}
    for cluster in range(n_clusters):
        cluster_indices[cluster] = potential_idx[np.where(
            clusters_all == cluster)[0]]
    return cluster_indices


def distribute_objects(N, m):
    quotient, remainder = divmod(N, m)
    group_sizes = [quotient] * m

    # Distribute the remainder (if any)
    for i in range(remainder):
        group_sizes[i] += 1
    np.random.shuffle(group_sizes)
    return group_sizes


def sampling(iter, dN, pre_filebase, method='random'):
    all_data_idx = np.arange(len(u0_train))
    currTrainDataIDX__=None
    if method == 'k-mean':
        if iter == 0:
            cluster_indices = get_cluster_index(
                y_kmeans[potential_train_data_idx], all_data_idx, 100)  # in [0, u0_train.shape[0]]
            num_addcase_per_cluster = distribute_objects(dN, 100)
            new_training_data_idx = np.array([])
            for i, (key, value) in enumerate(cluster_indices.items()):
                sorted_idx = np.arange(len(value))
                np.random.shuffle(sorted_idx)
                selected_idx = value[sorted_idx[:num_addcase_per_cluster[i]]]
                new_training_data_idx = np.concatenate(
                    (new_training_data_idx, selected_idx))
            currTrainDataIDX__ = new_training_data_idx
        else:
            LR_pool = ErrorMeasure(
                (u0_train, xy_train_testing),
                s_train,
            )
            pre_train_data_idx = np.genfromtxt(
                os.path.join(pre_filebase, "trainDataIDX.csv"), dtype=int, delimiter=","
            )
            # potential training data
            potential_train_data_idx = np.delete(
                all_data_idx, pre_train_data_idx)
            cluster_indices = get_cluster_index(
                y_kmeans[potential_train_data_idx], potential_train_data_idx, 100)  # in [0, u0_train.shape[0]]
            num_addcase_per_cluster = distribute_objects(dN, 100)
            new_training_data_idx = np.array([])
            for i, (key, value) in enumerate(cluster_indices.items()):
                # value is in [0, u0_train.shape[0]]
                sorted_idx = np.argsort(LR_pool[value])[::-1]
                selected_idx = value[sorted_idx[:num_addcase_per_cluster[i]]]
                new_training_data_idx = np.concatenate(
                    (new_training_data_idx, selected_idx))
            currTrainDataIDX__ = np.concatenate(
                (pre_train_data_idx, new_training_data_idx))
    else:#if method == 'PDF':
        if iter == 0:
            currTrainDataIDX__ = np.random.choice(
                a=all_data_idx, size=numCase_init, replace=False
            )
        else:
            pre_train_data_idx = np.genfromtxt(
                os.path.join(pre_filebase, "trainDataIDX.csv"), dtype=int, delimiter=","
            )
            # potential training data
            potential_train_data_idx = np.delete(
                all_data_idx, pre_train_data_idx)

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
            currTrainDataIDX__ = np.concatenate(
                (pre_train_data_idx, new_training_data_idx))
    return currTrainDataIDX__


# %%
numCase_init, numCase_add = int(str_dN), int(str_dN)
k = float(str_k)
c = float(str_c)

iter_start, iter_end = int(str_start), int(str_end)
if method == 'PDF':
    project_name = (
        "adapt_tl_PDF_k" + str_k + "c" + str_c + "dN" + str_dN+"case"+str_caseID
    )
elif method == 'k-mean':
    project_name = (
        "adapt_tl_kmean" + "dN" + str_dN+"case"+str_caseID
    )
filebase = os.path.join(prefix_filebase, project_name)
start_time = timeit.default_timer()

for iter in range(iter_start, iter_end):
    print("====iter=======:%d" % iter)
    pre_filebase = os.path.join(filebase, "iter" + str(iter - 1))
    current_filebase = os.path.join(filebase, "iter" + str(iter))
    os.makedirs(current_filebase, exist_ok=True)
    # restore_path = find_checkpoint_2restore(pre_filebase)
    currTrainDataIDX = sampling(iter, numCase_add, pre_filebase)

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
    #model.set_weights(initial_weights)
    h = model.fit(
        data.train_dataset, validation_data=data.test_dataset, epochs=80, verbose=2,
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
