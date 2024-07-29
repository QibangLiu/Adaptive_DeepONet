#!/usr/bin/env python
# coding: utf-8

# In[1]:
# %run ddm-deeponet_adaptive.py 200 0 2 1 1 1
# %%
import sys
import os
import DeepONet
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.io as scio
import timeit
from torch.utils.data import Dataset, DataLoader

# In[]
# %run ddm-deeponet_adaptive.py 2000 0 2 1 1 1
prefix_filebase = "./saved_model"
# str_dN, str_start, str_end = sys.argv[1:4]
# str_k, str_c = sys.argv[4:-1]
# str_caseID = sys.argv[-1]
str_dN, str_start, str_end = '200', '0', '2'
str_k, str_c = '1', '1'
str_caseID = '1'
# In[3]:

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
shift_solution, scaler_solution = np.mean(solutions_raw), np.std(solutions_raw)
shift_source, scaler_source = np.mean(source_terms_raw), np.std(source_terms_raw)
solutions = (solutions_raw - shift_solution) / scaler_solution
source_terms = (source_terms_raw - shift_source) / scaler_source

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

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_trunk = torch.tensor(xy_train_testing).to(device)
x_trunk.requires_grad_(True)

x_train_branch = torch.tensor(u0_train).to(device)
y_train = torch.tensor(s_train).to(device)
aux_train = torch.tensor(u0_train_raw).to(device)

x_test = (torch.tensor(u0_testing).to(device), x_trunk)
y_test = torch.tensor(s_testing).to(device)
dataset_test = DeepONet.TripleCartesianProd(x_test, y_test)
test_loader = DataLoader(
    dataset_test,
    batch_size=dataset_test.__len__(),
    collate_fn=dataset_test.custom_collate_fn,
)
# %%

mse = nn.MSELoss()
model = DeepONet.DeepONetCartesianProd(
    [m, 100, 100, 100, 100, 100, 100],
    [2, 100, 100, 100, 100, 100, 100],
    activation={"branch": nn.ReLU(), "trunk": nn.Tanh()},
)
# initial_weights = model.get_weights()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
model.compile(optimizer=optimizer, loss=mse)
model.to(device)


# %%
def ResidualError(y, x, aux=None,create_graph=False):
    dydx2 = DeepONet.laplacian(y, x,create_graph=create_graph)
    return -0.01 * (dydx2) * scaler_solution - aux


res_op = DeepONet.EvaluateDeepONetPDEs(ResidualError,model=model)


def ErrorMeasure(X, rhs):
    res = res_op(X, aux=rhs)
    res = res.detach().cpu().numpy()
    res = np.squeeze(res)
    rhs_np = rhs.detach().cpu().numpy()
    return np.linalg.norm(res, axis=1) / np.linalg.norm(rhs_np, axis=1)


def sampling(iter, dN, pre_filebase):
    all_data_idx = np.arange(len(x_train_branch))
    currTrainDataIDX__ = None
    if iter == 0:
        currTrainDataIDX__ = np.random.choice(
            a=all_data_idx, size=dN, replace=False
        )
    else:
        pre_train_data_idx = np.genfromtxt(
            os.path.join(pre_filebase, "trainDataIDX.csv"), dtype=int, delimiter=","
        )
        # potential training data
        potential_train_data_idx = np.delete(all_data_idx, pre_train_data_idx)
        print('befor error measure')
        torch.cuda.empty_cache()
        LR = ErrorMeasure(
            (x_train_branch[potential_train_data_idx], x_trunk),
            aux_train[potential_train_data_idx]
        )
        probility = np.power(LR, k) / np.power(LR, k).mean() + c
        probility_normalized = probility / np.sum(probility)
        new_training_data_idx = np.random.choice(
            a=potential_train_data_idx,
            size=dN,
            replace=False,
            p=probility_normalized,
        )
        currTrainDataIDX__ = np.concatenate((pre_train_data_idx, new_training_data_idx))
    return currTrainDataIDX__


# %%
numCase_add = int(str_dN)
k = float(str_k)
c = float(str_c)

iter_start, iter_end = int(str_start), int(str_end)
project_name = "adapt_k" + str_k + "c" + str_c + "dN" + str_dN + "case" + str_caseID
filebase = os.path.join(prefix_filebase, project_name)
start_time = timeit.default_timer()
# %%

if iter_start != 0:
    pre_filebase = os.path.join(filebase, "iter" + str(iter_start - 1))
    model.load_weights(os.path.join(pre_filebase, "model.ckpt"), device)
    model.load_logs(pre_filebase)

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

    curr_u_train = x_train_branch[currTrainDataIDX]
    x_train = (curr_u_train, x_trunk)
    curr_y_train = y_train[currTrainDataIDX]
    dataset_train = DeepONet.TripleCartesianProd(x_train, curr_y_train)
    train_loader = DataLoader(
        dataset_train,
        batch_size=64,
        shuffle=True,
        collate_fn=dataset_train.custom_collate_fn,
    )
    checkpoint_fname = os.path.join(current_filebase, "model.ckpt")
    checkpoint_callback = DeepONet.ModelCheckpoint(
        checkpoint_fname, monitor="val_loss", save_best_only=True
    )
    # model.set_weights(initial_weights)
    h = model.fit(
        train_loader,
        test_loader,
        epochs=1000,
        callbacks=checkpoint_callback,
    )
    model.save_logs(filebase)
    model.load_weights(checkpoint_fname, device)
    y_pred = model.predict(test_loader)
    error_test = np.linalg.norm(s_testing - y_pred, axis=1) / np.linalg.norm(
        s_testing, axis=1
    )
    np.savetxt(
        os.path.join(current_filebase, "TestL2Error.csv"),
        error_test,
        fmt="%.4e",
        delimiter=",",
    )
    stop_time = timeit.default_timer()
    print("training Run time so far: ", round(stop_time - start_time, 2), "(s)")
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
x_validate = x_test
y_validate = s_testing_raw
u0_validate = u0_testing_raw
x_validate[1].requires_grad_(True)
y_pred_out = model(x_validate)
y_pred = scaler_solution * y_pred_out.detach().cpu().numpy() + shift_solution


error_s = []
for i in range(len(y_validate)):
    error_s_tmp = np.linalg.norm(y_validate[i] - y_pred[i]) / np.linalg.norm(
        y_validate[i]
    )
    error_s.append(error_s_tmp)
error_s = np.stack(error_s)
fig = plt.figure()
_ = plt.hist(error_s)
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

    u0_validate_nx_ny = u0_validate[index].reshape(Ny, Nx)
    y_validate_nx_ny = y_validate[index].reshape(Ny, Nx)
    s_pred_nx_ny = y_pred[index].reshape(Ny, Nx)
    vmin = min(y_validate_nx_ny.min(), s_pred_nx_ny.min())
    vmax = max(y_validate_nx_ny.max(), s_pred_nx_ny.max())

    ax = plt.subplot(nr, nc, nc * i + 1)
    # py.figure(figsize = (14,7))
    c = ax.contourf(x_grid, y_grid, u0_validate_nx_ny, 20, cmap="jet")
    ax.set_title(r"Source Distrubution")
    plt.colorbar(c)
    plt.tight_layout()
    ax = plt.subplot(nr, nc, nc * i + 2)
    # py.figure(figsize = (14,7))
    c1 = ax.contourf(x_grid, y_grid, y_validate_nx_ny, 20, cmap="jet")
    ax.set_title(r"Reference Solution")
    cbar = fig.colorbar(c1, ax=ax)
    plt.tight_layout()
    ax = plt.subplot(nr, nc, nc * i + 3)
    # py.figure(figsize = (14,7))
    c2 = ax.contourf(x_grid, y_grid, s_pred_nx_ny, 20, cmap="jet")
    ax.set_title(r"Predicted Solution")
    cbar = fig.colorbar(c1, ax=ax)
    plt.tight_layout()


# %%
def LaplaceOperator2D(y, x, aux=None):
    dydx2 = DeepONet.laplacian(y, x)
    return -0.01 * (dydx2) * scaler_solution


# %%
laplace_op = DeepONet.EvaluateDeepONetPDEs(LaplaceOperator2D)
laplace_op_val = laplace_op(
    (x_validate[0][min_median_max_index], x_validate[1]), model=model
)
laplace_op_val = laplace_op_val.detach().cpu().numpy()

# %%
nr, nc = 3, 2
i = 0
fig = plt.figure(figsize=(8, 10))

for i, index in enumerate(min_median_max_index):

    vmin = np.min(u0_validate[index])
    vmax = np.max(u0_validate[index])

    ax = plt.subplot(nr, nc, nc * i + 1)
    # py.figure(figsize = (14,7))
    c1 = ax.contourf(
        x_grid,
        y_grid,
        u0_validate[index].reshape(Ny, Nx),
        20,
        vmax=vmax,
        vmin=vmin,
        cmap="jet",
    )
    ax.set_title(r"Source Distrubution")
    cbar = fig.colorbar(c1, ax=ax)
    plt.tight_layout()
    ax = plt.subplot(nr, nc, nc * i + 2)
    # py.figure(figsize = (14,7))
    c2 = ax.contourf(
        x_grid,
        y_grid,
        laplace_op_val[i].reshape(Ny, Nx),
        20,
        vmax=vmax,
        vmin=vmin,
        cmap="jet",
    )
    ax.set_title(r"$-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$")
    cbar = fig.colorbar(c2, ax=ax)
    plt.tight_layout()
