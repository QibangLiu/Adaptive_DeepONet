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
import timeit

# In[]

filebase = "./saved_model/poisson_tf"

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
# shift_solution, scaler_solution = np.min(solutions_raw), np.max(solutions_raw)-np.min(solutions_raw)
# shift_source, scaler_source = np.min(source_terms_raw), np.max(source_terms_raw)-np.min(source_terms_raw)
# shift_solution, scaler_solution = 0, (np.max(solutions_raw)-np.min(solutions_raw))*0.5
# shift_source, scaler_source = 0, (np.max(source_terms_raw)-np.min(source_terms_raw))*0.5
# shift_solution, scaler_solution = 0, 1
# shift_source, scaler_source = 0, 1
solutions = (solutions_raw - shift_solution) / scaler_solution
source_terms = (source_terms_raw - shift_source) / scaler_source

num_train = 5000

u0_train = source_terms[:num_train]
u0_testing = source_terms[-1000:]
s_train = solutions[:num_train]
s_testing = solutions[-1000:]

u0_testing_raw = source_terms_raw[-1000:]
u0_train_raw = source_terms_raw[:num_train]
s_testing_raw = solutions_raw[-1000:]
s_train_raw = solutions_raw[:num_train]

xy_train_testing = np.concatenate(
    [x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)], axis=1
)

# %%
x_train = (u0_train, xy_train_testing)
y_train = s_train
x_test = (u0_testing, xy_train_testing)
y_test = s_testing

# %%

data_train = don.TripleCartesianProd(x_train, y_train, batch_size=64)
data_test = don.TripleCartesianProd(x_test, y_test, shuffle=False)
model = don.DeepONetCartesianProd(
    [m, 100, 100, 100, 100, 100, 100],
    [2, 100, 100, 100, 100, 100, 100],
    {"branch": "relu", "trunk": "tanh"},
)
initial_weights = model.get_weights()
laplace_op = don.EvaluateDeepONetPDEs(model, don.laplacian)


def ErrorMeasure(X, rhs):
    lhs = -0.01 * (laplace_op(X)) * scaler_solution
    lhs = np.squeeze(lhs)
    return np.linalg.norm(lhs - rhs, axis=1) / np.linalg.norm(rhs, axis=1)


# %%
start_time = timeit.default_timer()

optm = tf.keras.optimizers.Adam(learning_rate=5e-4)
model.compile(optimizer=optm, loss="mse")
checkpoint_fname = os.path.join(filebase, "model.ckpt")
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_fname,
    save_weights_only=True,
    monitor="val_loss",
    verbose=0,
    save_freq="epoch",
    save_best_only=True,
    mode="min",
)
model.load_weights(checkpoint_fname)
h = model.load_history(filebase)
# h = model.fit(
#     data_train.dataset,
#     validation_data=data_test.dataset,
#     epochs=1000,
#     verbose=2,
#     callbacks=[model_checkpoint],
# )
# model.save_history(filebase)
# model.load_weights(checkpoint_fname)

y_pred = model.predict(data_test.X_data)
y_pred_inverse = y_pred * scaler_solution + shift_solution
error_test = np.linalg.norm(s_testing_raw - y_pred_inverse, axis=1) / np.linalg.norm(
    s_testing_raw, axis=1
)
stop_time = timeit.default_timer()
print("training Run time so far: ", round(stop_time - start_time, 2), "(s)")
fig = plt.figure()
plt.hist(error_test)
plt.xlabel("L2 relative error")
plt.ylabel("frequency")
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["loss"], label="loss")
ax.plot(h["val_loss"], label="val_loss")
ax.legend()
ax.set_yscale("log")


# Plotting Results
# %%
u0_validate = u0_train_raw
y_validate = s_train_raw
x_validate = (u0_train, xy_train_testing)
with tf.device('/GPU:0'):
    y_pred_out = model.predict(x_validate)
y_pred = y_pred_out * scaler_solution + shift_solution
# %%
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

# %%
laplace_op_val_ = laplace_op((x_validate[0][min_median_max_index], x_validate[1]))
# %%
laplace_op_val = -0.01 * laplace_op_val_ * scaler_solution


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


# %%
ErrorMeasure(
    (x_validate[0][min_median_max_index], x_validate[1]),
    u0_validate[min_median_max_index],
)

dx = x_grid[0, 1] - x_grid[0, 0]
dy = y_grid[1, 0] - y_grid[0, 0]
dx=(np.max(x_grid)-np.min(x_grid))/(Nx-1)/5
dy=(np.max(y_grid)-np.min(y_grid))/(Ny-1)/5
dx=dx.astype(np.float32)
dy=dy.astype(np.float32)
# %%
def laplacian_FD(X_data, dx, dy,model):
    """u shape=(batch_size,Ny,Nx)
    x shape=(Ny*Nx,2)
    return shape=(batch_size,Ny-2,Nx-2)"""
    x=X_data[1] # shape=(Ny*Nx,2) coordinates
    u0=X_data[0] # shape=(batch_size,Ny*Nx) source terms
    x_plus_dx=np.concatenate( (x[:,0:1]+dx, x[:,1:2]),axis=1)
    x_minus_dx=np.concatenate( (x[:,0:1]-dx, x[:,1:2]),axis=1)
    x_plus_dy=np.concatenate( (x[:,0:1], x[:,1:2]+dy),axis=1)
    x_minus_dy=np.concatenate( (x[:,0:1], x[:,1:2]-dy),axis=1)
    u=model((u0,x))
    u_plus_dx=model((u0,x_plus_dx))
    u_minus_dx=model((u0,x_minus_dx))
    u_plus_dy=model((u0,x_plus_dy))
    u_minus_dy=model((u0,x_minus_dy))
    du_dxx=(u_plus_dx-2*u+u_minus_dx)/dx**2
    du_dyy=(u_plus_dy-2*u+u_minus_dy)/dy**2
    lap = du_dxx + du_dyy
    return lap

X_data = (x_validate[0][min_median_max_index], x_validate[1])
x=X_data[1] # shape=(Ny*Nx,2) coordinates
u0=X_data[0] # shape=(batch_size,Ny*Nx) source terms
x_plus_dx=np.concatenate( (x[:,0:1]+dx, x[:,1:2]),axis=1)
u=model((u0,x)).numpy().reshape(-1,Ny,Nx)*scaler_solution+shift_solution
u_plus_dx=model((u0,x_plus_dx)).numpy().reshape(-1,Ny,Nx)*scaler_solution+shift_solution
# %%
# def laplacian_FD(u,dx,dy):
#     """u shape=(batch_size,Ny,Nx)
#     return shape=(batch_size,Ny-2,Nx-2)"""
#     du_dxx=(u[:,1:-1,2:]-2*u[:,1:-1,1:-1]+u[:,1:-1,:-2])/dx**2
#     du_dyy=(u[:,2:,1:-1]-2*u[:,1:-1,1:-1]+u[:,:-2,1:-1])/dy**2
#     return du_dxx+du_dyy

laplace_FD_val=-0.01*laplacian_FD((x_validate[0][min_median_max_index], x_validate[1]), dx, dy,model)*scaler_solution

#laplace_FD_val = -0.01 * laplacian_FD(y_pred[min_median_max_index].reshape(-1,Ny,Nx), dx, dy)
#laplace_FD_true= -0.01 * don.laplacian_FD(y_validate[min_median_max_index].reshape(-1,Ny,Nx), dx, dy)
laplace_FD_true=laplace_FD_val
# %%
nr, nc = 3, 4
i = 0
fig = plt.figure(figsize=(16, 10))


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

    # ax = plt.subplot(nr, nc, nc * i + 2)
    # # py.figure(figsize = (14,7))
    # c3 = ax.contourf(
    #     x_grid[1:-1, 1:-1],
    #     y_grid[1:-1, 1:-1],
    #     laplace_FD_true[i],
    #     20,
    #     vmax=vmax,
    #     vmin=vmin,
    #     cmap="jet",
    # )
    # ax.set_title(r"Source Distrubution by FD of Reference Solution")
    # cbar = fig.colorbar(c3, ax=ax)
    # plt.tight_layout()

    ax = plt.subplot(nr, nc, nc * i + 3)
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
    ax.set_title(r"AD: $-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$")
    cbar = fig.colorbar(c2, ax=ax)
    plt.tight_layout()

    ax = plt.subplot(nr, nc, nc * i + 4)
    # py.figure(figsize = (14,7))
    c3 = ax.contourf(
        x_grid,
        y_grid,
        laplace_FD_val[i].numpy().reshape(Ny, Nx),
        # x_grid[1:-1, 1:-1],
        # y_grid[1:-1, 1:-1],
        # laplace_FD_val[i],
        20,
        vmax=vmax,
        vmin=vmin,
        cmap="jet",
    )
    ax.set_title(r"FD: $-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$")
    cbar = fig.colorbar(c3, ax=ax)
    plt.tight_layout()


# # %%
# %%timeit
# laplace_op_val_ = laplace_op((source_terms, x_train[1]))

# %%
# %%timeit
# with tf.device('/CPU:0'):
#     y_pred = model((source_terms, x_train[1]))
# y_pred = tf.reshape(model((source_terms, x_train[1])),(-1, Ny, Nx))
# laplace_val_FD = -0.01 * don.laplacian_FD(y_pred, dx, dy)


# %%
