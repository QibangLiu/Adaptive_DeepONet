#!/usr/bin/env python
# coding: utf-8

# In[1]:

import scipy.integrate as sciint
import sys
import os
import DeepONet_tf as DeepONet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.io as scio
import tensorflow as tf
import h5py
import timeit
tf.keras.backend.set_floatx('float64')
# In[]
train=False
filebase = "./saved_model/test"

# In[3]:
# fenics_data = scio.loadmat("./TrainingData/poisson_gauss_cov20k.mat")
# x_grid = fenics_data["x_grid"].astype(np.float64)  # shape (Ny, Nx)
# y_grid = fenics_data["y_grid"].astype(np.float64)
# source_terms_raw = fenics_data["source_terms"].astype(np.float64).reshape(-1, Nx * Ny)  # shape (N, Ny* Nx)
# solutions_raw = fenics_data["solutions"].astype(np.float64).reshape(-1, Nx * Ny)  # shape (N, Ny* Nx)

hf=h5py.File("./TrainingData/poisson_gauss_fft40K.h5", 'r')
x_grid=hf['x_grid'][:].astype(np.float64)  # shape (Nx,)\
y_grid = hf["y_grid"][:].astype(np.float64)
Nx,Ny=x_grid.shape[1],x_grid.shape[0]
source_terms_raw = hf["source_terms"][:].astype(np.float64).reshape(-1, Nx * Ny)
solutions_raw = hf["solutions"][:].astype(np.float64).reshape(-1, Nx * Ny)
hf.close()
dx=x_grid[0,1]-x_grid[0,0]
dy=y_grid[1,0]-y_grid[0,0]

m = Nx * Ny

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

num_train = 500
num_test = 1000

u0_train = source_terms[:num_train]
u0_testing = source_terms[-num_test:]
s_train = solutions[:num_train]
s_testing = solutions[-num_test:]

u0_testing_raw = source_terms_raw[-num_test:]
u0_train_raw = source_terms_raw[:num_train]
s_testing_raw = solutions_raw[-num_test:]
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
num_nodes = 200
data_train = DeepONet.TripleCartesianProd(x_train, y_train, batch_size=64)
data_test = DeepONet.TripleCartesianProd(x_test, y_test, shuffle=False)
model = DeepONet.DeepONetCartesianProd(
    [m, num_nodes, num_nodes, num_nodes, num_nodes, num_nodes, num_nodes],
    [2, num_nodes, num_nodes, num_nodes, num_nodes, num_nodes, num_nodes],
    {"branch": "relu", "trunk": "tanh"},
)


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

if train:
    h = model.fit(
        data_train.dataset,
        validation_data=data_test.dataset,
        epochs=1000,
        verbose=2,
        callbacks=[model_checkpoint],
    )
    model.save_history(filebase)
    model.load_weights(checkpoint_fname)
else:
    model.load_weights(checkpoint_fname)
    h = model.load_history(filebase)
stop_time = timeit.default_timer()
print("training Run time so far: ", round(stop_time - start_time, 2), "(s)")

# fig = plt.figure()
# ax = plt.subplot(1, 1, 1)
# ax.plot(h["loss"], label="loss")
# ax.plot(h["val_loss"], label="val_loss")
# ax.legend()
# ax.set_yscale("log")
# %%
x_validate=(u0_testing, xy_train_testing)
y_validate=s_testing_raw
u0_validate=u0_testing_raw
def L2RelativeError(x_validate, y_validate):
    y_pred = model.predict(x_validate)
    y_pred = y_pred * scaler_solution + shift_solution
    error_s = np.linalg.norm(y_validate - y_pred, axis=1) / np.linalg.norm(
        y_validate, axis=1
    )
    return error_s, y_pred

error_s, y_pred = L2RelativeError(x_validate, y_validate)
fig = plt.figure()
_=plt.hist(error_s)
plt.xlabel("L2 relative error")
plt.ylabel("frequency")


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

laplace_op = DeepONet.EvaluateDeepONetPDEs(model, DeepONet.laplacian)
def ErrorMeasure(X, rhs,laplace_op):
    lhs = -0.01 * (laplace_op(X)) * scaler_solution
    lhs = np.squeeze(lhs)
    return np.linalg.norm(lhs - rhs, axis=1) / np.linalg.norm(rhs, axis=1)

res=ErrorMeasure(x_validate, u0_validate,laplace_op)

# %%
# jacob_op=DeepONet.EvaluateDeepONetPDEs(model, DeepONet.jacobian)
# def ErrorMeasure(X, rhs,jacob_op):
#     du_dX = -0.01 *jacob_op(X)* scaler_solution
#     du_dx, du_dy = du_dX[:,:, 0], du_dX[:,:, 1]
#     du_dx=du_dx.reshape(-1,Ny,Nx)
#     du_dy=du_dy.reshape(-1,Ny,Nx)
    
#     # left side
#     int_side_0 = -sciint.simpson(du_dx[:,:, 0], dx=dy)  # along y
#     # right side
#     int_side_1 = sciint.simpson(du_dx[:, :, -1], dx=dy)  # along y
#     # bot side
#     int_side_2 = -sciint.simpson(du_dy[:, 0, :], dx=dx)  # along x
#     # top side
#     int_side_3 = sciint.simpson(du_dy[:, -1, :], dx=dx)  # along x
    
#     int_side = int_side_0 + int_side_1 + int_side_2 + int_side_3
    
#     rhs=rhs.reshape(-1,Ny,Nx)
#     int1 = sciint.simpson(rhs, dx=dx, axis=-1)  # int along x
#     int_rhs=sciint.simpson(int1, dx=dy)  # int along y
    

#     return np.abs(int_side - int_rhs)#/np.abs(int_rhs)
    
# res=ErrorMeasure(x_validate, u0_validate,jacob_op)

# %%
gap=2
plt.plot(error_s[sort_idx][::gap],res[sort_idx][::gap],'o')
# Fit a straight line
coefficients = np.polyfit(error_s[sort_idx], res[sort_idx], 1)
line = np.poly1d(coefficients)

# Plot the line
plt.plot(error_s[sort_idx], res[sort_idx], 'o')
plt.plot(error_s[sort_idx], line(error_s[sort_idx]), color='red')

# Add labels and title
plt.xlabel('L2 Relative Error')
plt.ylabel('Residual Error')
plt.title('L2 Relative Error vs Residual Error')
# plt.xlim(0.005, 0.02)
# plt.ylim(0.00, 0.015)

correlation = np.corrcoef(error_s, res)[0, 1]
print("Pearson correlation coefficient:", correlation)

from scipy.stats import spearmanr
r_spearman, _ = spearmanr(error_s, res)
print(f"Spearman's rank correlation coefficient: {r_spearman}")

# %%
# check the derivatives
laplace_op_val_ = laplace_op((x_validate[0][min_median_max_index], x_validate[1]))
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
    ticks = cbar.get_ticks()
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
    cbar = plt.colorbar(c2)
    #cbar.set_ticks(ticks)
    plt.tight_layout()

sys.exit()
# %%
# below is for FD

ErrorMeasure(
    (x_validate[0][min_median_max_index], x_validate[1]),
    u0_validate[min_median_max_index],
)

dx = x_grid[0, 1] - x_grid[0, 0]
dy = y_grid[1, 0] - y_grid[0, 0]
dx=(np.max(x_grid)-np.min(x_grid))/(Nx-1)/5
dy=(np.max(y_grid)-np.min(y_grid))/(Ny-1)/5
dx=dx.astype(np.float64)
dy=dy.astype(np.float64)
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
