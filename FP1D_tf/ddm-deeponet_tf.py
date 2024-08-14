#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import DeepONet_tf as DeepONet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.io as scio
from deepxde import utils
import deepxde as dde
from deepxde.backend import tf
import timeit
from IPython.display import HTML

# In[]

filebase = "./saved_model/poisson_FP1D"

# In[3]:


fenics_data = scio.loadmat("./TrainingData/FP1D_all.mat")

x_grid_file=fenics_data["x_data"].squeeze()
T_data_file=fenics_data["Temp_data"].squeeze()
alpha_data_file=fenics_data["alpha_data"].squeeze()
t_data_file=fenics_data["t_data"].squeeze()
pcs_data_raw=fenics_data["process_condition"].astype(np.float32)
x_grid_raw=x_grid_file.astype(np.float32)
scaler_x=np.max(x_grid_raw)
x_grid=x_grid_raw/scaler_x
Nx=len(x_grid)
# %%
T_data_raw=[]
alpha_data=[]
tx_data=[]
t_data_raw=np.array([0])
t_data=np.array([0])
Tmaxs,Tmins=[],[]
for i in range(len(t_data_file)):
    t_temp=t_data_file[i].squeeze().astype(np.float32)
    if len(t_temp)>len(t_data):
        t_data_raw=t_temp
    
    T_data_raw.append(T_data_file[i].reshape(-1,1).astype(np.float32))
    alpha_data.append(alpha_data_file[i].reshape(-1,1).astype(np.float32))
    Tmaxs.append(np.max(T_data_raw[i]))
    Tmins.append(np.min(T_data_raw[i]))
    if len(alpha_data_file[i])!=len(t_temp):
        raise ValueError("The number of time steps in alpha_data and t_arr_results do not match")   
    if len(T_data_file[i])!=len(t_temp):
        raise ValueError("The number of time steps in T_data and t_arr_results do not match")
scaler_t=np.max(t_data_raw)
t_data=t_data_raw/scaler_t
x_,t_=np.meshgrid(x_grid,t_data)
tx_data=(np.concatenate((t_.reshape(-1,1),x_.reshape(-1,1)),axis=1)) 
Tmax,Tmin=max(Tmaxs),min(Tmins)
scaler_T,shift_T=(Tmax-Tmin),Tmin
T_data = [(T - shift_T) / scaler_T for T in T_data_raw]
Talpha_data=[np.concatenate((T_data[i],alpha_data[i]),axis=1) for i in range(len(T_data))]
Talpha_data_raw=[np.concatenate((T_data_raw[i],alpha_data[i]),axis=1) for i in range(len(T_data_raw))]
T0_min,T0_max=np.min(pcs_data_raw[:,0]),np.max(pcs_data_raw[:,0])
scaler_T0,shift_T0=T0_max-T0_min,T0_min
alpha0_min,alpha0_max=np.min(pcs_data_raw[:,1]),np.max(pcs_data_raw[:,1])
scaler_alpha0,shift_alpha0=alpha0_max-alpha0_min,alpha0_min
pcs_data=(pcs_data_raw-np.array([shift_T0,shift_alpha0]))/np.array([scaler_T0,scaler_alpha0])

# T_data,T_data_raw: list of np with shape(1,nx*nt), shift_T,scaler_T: float
# alpha_data, list of np with shape(1,nx*nt)
# t_data, t_data_raw:  1D np, scaler_t: float
#x_grid, x_grid_raw: 1D np, scaler_x: float
# tx_data: 2D np, meshgrid of x_grid and t_data, [t,x] format
# pcs_data, pcs_data_raw: 2D np, [T0,alpha0] format

# %%
def pad_sequences(arrays, pad_value=0):
    # Determine the maximum number of rows
    max_rows = max(arr.shape[0] for arr in arrays)
    
    # Initialize a list to store the padded arrays
    padded_arrays = []
    
    for arr in arrays:
        # Create a new array with the same number of columns but with max_rows number of rows
        padded_arr = np.full((max_rows, arr.shape[1]), pad_value, dtype=arr.dtype)
        padded_arr[:arr.shape[0], :] = arr
        padded_arrays.append(padded_arr)
    
    return np.stack(padded_arrays)

padding_value=-1000
Talpha_data_padded=pad_sequences(Talpha_data,padding_value)

#Talpha_data_padded = tf.keras.preprocessing.sequence.pad_sequences(Talpha_data, padding='post',value=padding_value)
mask = (Talpha_data_padded != padding_value).astype('float32')
Talpha_data_padded=Talpha_data_padded*mask
# %% 
num_train=5000
pcs_train=pcs_data[:num_train]
Talpha_train=Talpha_data_padded[:num_train]
mask_train=mask[:num_train]
Talpha_raw_train=Talpha_data_raw[:num_train]

pcs_testing=pcs_data[-1000:]
Talpha_testing=Talpha_data_padded[-1000:]
mask_testing=mask[-1000:]
Talpha_raw_testing=Talpha_data_raw[-1000:]
# %%
x_train = (pcs_train, tx_data)
y_train = Talpha_train
x_testing = (pcs_testing, tx_data)
y_testing = Talpha_testing


# %%
data_test = DeepONet.TripleCartesianProd(x_testing, y_testing,aux_data=mask_testing, shuffle=False)
data_train = DeepONet.TripleCartesianProd(x_train, y_train,aux_data=mask_train, batch_size=64)


# %%
model = DeepONet.DeepONetCartesianProd(
    [2, 100, 100,100,100,100,100],
    [2, 100,100,100,100,100,100],
    {"branch": "relu", "trunk": "tanh"},num_outputs=2,
)

laplace_op = DeepONet.EvaluateDeepONetPDEs(model, DeepONet.laplacian)


def ErrorMeasure(X, rhs):
    lhs = -0.01 * (laplace_op(X)) * scaler_solution
    lhs = np.squeeze(lhs)
    return np.linalg.norm(lhs - rhs, axis=1) / np.linalg.norm(rhs, axis=1)


# %%
start_time = timeit.default_timer()

optm = tf.keras.optimizers.Adam(learning_rate=1e-3)
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
# model.load_weights(checkpoint_fname)
# h = model.load_history(filebase)
h = model.fit(
    data_train.dataset,
    validation_data=data_test.dataset,
    epochs=500,
    verbose=2,
    callbacks=[model_checkpoint],
)
model.save_history(filebase)
model.load_weights(checkpoint_fname)


stop_time = timeit.default_timer()
print("training Run time so far: ", round(stop_time - start_time, 2), "(s)")

fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["loss"], label="loss")
ax.plot(h["val_loss"], label="val_loss")
ax.legend()
ax.set_yscale("log")


# Plotting Results
# %%
x_validate=(x_testing[0],x_testing[1],mask_testing)
y_validate=Talpha_raw_testing
mask_validate=x_validate[2].astype(bool)

y_pred_out = model.predict(x_validate)
y_pred_raw=np.ones_like(y_pred_out)
y_pred_raw[:,:,0]=y_pred_out[:,:,0]*scaler_T+shift_T
y_pred_raw[:,:,1]=y_pred_out[:,:,1]
y_pred = [y_pred_raw[i][mask_validate[i,:,0]] for i in range(y_pred_raw.shape[0])]
# %%
error_s = []
for i in range(len(y_validate)):
    error_t = np.linalg.norm(y_validate[i][:,0] - y_pred[i][:,0]) / np.linalg.norm(
        y_validate[i][:,0]
    )
    error_a = np.linalg.norm(y_validate[i][:,1] - y_pred[i][:,1]) / np.linalg.norm(
        y_validate[i][:,1]
    )
    error_s_tmp=0.5*(error_t+error_a)
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

# %%

data_video=[]
tx_data_validate=[x_validate[1][mask_validate[i,:,0]] for i in min_median_max_index]
for i,idx in enumerate(min_median_max_index):
    t= scaler_t* tx_data_validate[i][:,0].reshape(-1,len(x_grid_file))
    Ttrue=y_validate[idx][:,0].reshape(-1,len(x_grid_file))
    Tpred=y_pred[idx][:,0].reshape(-1,len(x_grid_file))
    alpha_true=y_validate[idx][:,1].reshape(-1,len(x_grid_file))
    alpha_pred=y_pred[idx][:,1].reshape(-1,len(x_grid_file))
    data={'x':x_grid_file,'t':t[:,0],"T_true":Ttrue,"T_pred":Tpred,"alpha_true":alpha_true,"alpha_pred":alpha_pred}
    data_video.append(data)
# %%
ani=DeepONet.get_video(data_video[0])
HTML(ani.to_html5_video())
# %%
ani=DeepONet.get_video(data_video[1])
HTML(ani.to_html5_video())
# %%
ani=DeepONet.get_video(data_video[-1])
HTML(ani.to_html5_video())
# %%

# # %%
# laplace_op_val_ = laplace_op((x_validate[0][min_median_max_index], x_validate[1]))
# # %%
# laplace_op_val = -0.01 * laplace_op_val_ * scaler_solution


# nr, nc = 3, 2
# i = 0
# fig = plt.figure(figsize=(8, 10))


# for i, index in enumerate(min_median_max_index):

#     vmin = np.min(u0_validate[index])
#     vmax = np.max(u0_validate[index])

#     ax = plt.subplot(nr, nc, nc * i + 1)
#     # py.figure(figsize = (14,7))
#     c1 = ax.contourf(
#         x_grid,
#         y_grid,
#         u0_validate[index].reshape(Ny, Nx),
#         20,
#         vmax=vmax,
#         vmin=vmin,
#         cmap="jet",
#     )
#     ax.set_title(r"Source Distrubution")
#     cbar = fig.colorbar(c1, ax=ax)
#     plt.tight_layout()
#     ax = plt.subplot(nr, nc, nc * i + 2)
#     # py.figure(figsize = (14,7))
#     c2 = ax.contourf(
#         x_grid,
#         y_grid,
#         laplace_op_val[i].reshape(Ny, Nx),
#         20,
#         vmax=vmax,
#         vmin=vmin,
#         cmap="jet",
#     )
#     ax.set_title(r"$-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$")
#     cbar = fig.colorbar(c2, ax=ax)
#     plt.tight_layout()


# # %%
# ErrorMeasure(
#     (x_validate[0][min_median_max_index], x_validate[1]),
#     u0_validate[min_median_max_index],
# )

# dx = x_grid[0, 1] - x_grid[0, 0]
# dy = y_grid[1, 0] - y_grid[0, 0]
# dx=(np.max(x_grid)-np.min(x_grid))/(Nx-1)/5
# dy=(np.max(y_grid)-np.min(y_grid))/(Ny-1)/5
# dx=dx.astype(np.float32)
# dy=dy.astype(np.float32)
# # %%
# def laplacian_FD(X_data, dx, dy,model):
#     """u shape=(batch_size,Ny,Nx)
#     x shape=(Ny*Nx,2)
#     return shape=(batch_size,Ny-2,Nx-2)"""
#     x=X_data[1] # shape=(Ny*Nx,2) coordinates
#     u0=X_data[0] # shape=(batch_size,Ny*Nx) source terms
#     x_plus_dx=np.concatenate( (x[:,0:1]+dx, x[:,1:2]),axis=1)
#     x_minus_dx=np.concatenate( (x[:,0:1]-dx, x[:,1:2]),axis=1)
#     x_plus_dy=np.concatenate( (x[:,0:1], x[:,1:2]+dy),axis=1)
#     x_minus_dy=np.concatenate( (x[:,0:1], x[:,1:2]-dy),axis=1)
#     u=model((u0,x))
#     u_plus_dx=model((u0,x_plus_dx))
#     u_minus_dx=model((u0,x_minus_dx))
#     u_plus_dy=model((u0,x_plus_dy))
#     u_minus_dy=model((u0,x_minus_dy))
#     du_dxx=(u_plus_dx-2*u+u_minus_dx)/dx**2
#     du_dyy=(u_plus_dy-2*u+u_minus_dy)/dy**2
#     lap = du_dxx + du_dyy
#     return lap

# X_data = (x_validate[0][min_median_max_index], x_validate[1])
# x=X_data[1] # shape=(Ny*Nx,2) coordinates
# u0=X_data[0] # shape=(batch_size,Ny*Nx) source terms
# x_plus_dx=np.concatenate( (x[:,0:1]+dx, x[:,1:2]),axis=1)
# u=model((u0,x)).numpy().reshape(-1,Ny,Nx)*scaler_solution+shift_solution
# u_plus_dx=model((u0,x_plus_dx)).numpy().reshape(-1,Ny,Nx)*scaler_solution+shift_solution
# # %%
# # def laplacian_FD(u,dx,dy):
# #     """u shape=(batch_size,Ny,Nx)
# #     return shape=(batch_size,Ny-2,Nx-2)"""
# #     du_dxx=(u[:,1:-1,2:]-2*u[:,1:-1,1:-1]+u[:,1:-1,:-2])/dx**2
# #     du_dyy=(u[:,2:,1:-1]-2*u[:,1:-1,1:-1]+u[:,:-2,1:-1])/dy**2
# #     return du_dxx+du_dyy

# laplace_FD_val=-0.01*laplacian_FD((x_validate[0][min_median_max_index], x_validate[1]), dx, dy,model)*scaler_solution

# #laplace_FD_val = -0.01 * laplacian_FD(y_pred[min_median_max_index].reshape(-1,Ny,Nx), dx, dy)
# #laplace_FD_true= -0.01 * don.laplacian_FD(y_validate[min_median_max_index].reshape(-1,Ny,Nx), dx, dy)
# laplace_FD_true=laplace_FD_val
# # %%
# nr, nc = 3, 4
# i = 0
# fig = plt.figure(figsize=(16, 10))


# for i, index in enumerate(min_median_max_index):

#     vmin = np.min(u0_validate[index])
#     vmax = np.max(u0_validate[index])

#     ax = plt.subplot(nr, nc, nc * i + 1)
#     # py.figure(figsize = (14,7))
#     c1 = ax.contourf(
#         x_grid,
#         y_grid,
#         u0_validate[index].reshape(Ny, Nx),
#         20,
#         vmax=vmax,
#         vmin=vmin,
#         cmap="jet",
#     )
#     ax.set_title(r"Source Distrubution")
#     cbar = fig.colorbar(c1, ax=ax)
#     plt.tight_layout()

#     # ax = plt.subplot(nr, nc, nc * i + 2)
#     # # py.figure(figsize = (14,7))
#     # c3 = ax.contourf(
#     #     x_grid[1:-1, 1:-1],
#     #     y_grid[1:-1, 1:-1],
#     #     laplace_FD_true[i],
#     #     20,
#     #     vmax=vmax,
#     #     vmin=vmin,
#     #     cmap="jet",
#     # )
#     # ax.set_title(r"Source Distrubution by FD of Reference Solution")
#     # cbar = fig.colorbar(c3, ax=ax)
#     # plt.tight_layout()

#     ax = plt.subplot(nr, nc, nc * i + 3)
#     # py.figure(figsize = (14,7))
#     c2 = ax.contourf(
#         x_grid,
#         y_grid,
#         laplace_op_val[i].reshape(Ny, Nx),
#         20,
#         vmax=vmax,
#         vmin=vmin,
#         cmap="jet",
#     )
#     ax.set_title(r"AD: $-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$")
#     cbar = fig.colorbar(c2, ax=ax)
#     plt.tight_layout()

#     ax = plt.subplot(nr, nc, nc * i + 4)
#     # py.figure(figsize = (14,7))
#     c3 = ax.contourf(
#         x_grid,
#         y_grid,
#         laplace_FD_val[i].numpy().reshape(Ny, Nx),
#         # x_grid[1:-1, 1:-1],
#         # y_grid[1:-1, 1:-1],
#         # laplace_FD_val[i],
#         20,
#         vmax=vmax,
#         vmin=vmin,
#         cmap="jet",
#     )
#     ax.set_title(r"FD: $-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$")
#     cbar = fig.colorbar(c3, ax=ax)
#     plt.tight_layout()


# # # %%
# # %%timeit
# # laplace_op_val_ = laplace_op((source_terms, x_train[1]))
