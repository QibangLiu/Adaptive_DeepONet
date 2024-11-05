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
# from deepxde import utils
# import deepxde as dde
# from deepxde.backend import tf
import tensorflow as tf
import timeit
from IPython.display import HTML
import scipy.integrate as sciint
physical_devices = tf.config.list_physical_devices()
for dev in physical_devices:
    print(dev)
print("tf version:", tf.__version__)
# In[]
tf.keras.backend.set_floatx('float32')
filebase = "./saved_model/adapt_k0c0dN50case0/iter29"
train = False
# In[3]:


fenics_data = scio.loadmat("./TrainingData/FP1Drand_all.mat")

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
    if len(t_temp)>len(t_data_raw):
        t_data_raw=t_temp
    
    T_data_raw.append(T_data_file[i].reshape(-1,1).astype(np.float32))
    alpha_data.append(alpha_data_file[i].reshape(-1,1).astype(np.float32))
    Tmaxs.append(np.max(T_data_raw[i]))
    Tmins.append(np.min(T_data_raw[i]))
    if len(alpha_data_file[i])!=len(t_temp):
        raise ValueError("The number of time steps in alpha_data and t_arr_results do not match")   
    if len(T_data_file[i])!=len(t_temp):
        raise ValueError("The number of time steps in T_data and t_arr_results do not match")
scaler_t=np.max(t_data_raw) # ()
t_data=t_data_raw/scaler_t # (Nt,)
x_,t_=np.meshgrid(x_grid,t_data) # (Nt, Nx)
dx=x_grid_raw[1]-x_grid_raw[0]
dt=t_data_raw[1]-t_data_raw[0]
tx_data=(np.concatenate((t_.reshape(-1,1),x_.reshape(-1,1)),axis=1))  # (Nt*Nx,2)
Tmax,Tmin=max(Tmaxs),min(Tmins)
scaler_T,shift_T=(Tmax-Tmin),Tmin
T_data = [(T - shift_T) / scaler_T for T in T_data_raw] #[(Nt*Nx,1)]
Talpha_data=[np.concatenate((T_data[i],alpha_data[i]),axis=1) for i in range(len(T_data))] #[(Nt*Nx,2)],(T, alpha)
Talpha_data_raw=[np.concatenate((T_data_raw[i],alpha_data[i]),axis=1) for i in range(len(T_data_raw))]
T0_min,T0_max=np.min(pcs_data_raw[:,0]),np.max(pcs_data_raw[:,0])
scaler_T0,shift_T0=T0_max-T0_min,T0_min
alpha0_min,alpha0_max=np.min(pcs_data_raw[:,1]),np.max(pcs_data_raw[:,1])
scaler_alpha0,shift_alpha0=alpha0_max-alpha0_min,alpha0_min
pcs_data=(pcs_data_raw-np.array([shift_T0,shift_alpha0]))/np.array([scaler_T0,scaler_alpha0]) # (Nb,2)

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
num_train=500
num_test=pcs_data.shape[0]-num_train
num_test=1000

pcs_train=pcs_data[:num_train]
Talpha_train=Talpha_data_padded[:num_train]
mask_train=mask[:num_train]
Talpha_raw_train=Talpha_data_raw[:num_train]

pcs_testing=pcs_data[-num_test:]
pcs_testing_raw=pcs_data_raw[-num_test:]
Talpha_testing=Talpha_data_padded[-num_test:]
mask_testing=mask[-num_test:]
Talpha_raw_testing=Talpha_data_raw[-num_test:]
# %%
x_train = (pcs_train, tx_data)
y_train = Talpha_train
x_testing = (pcs_testing, tx_data)
y_testing = Talpha_testing


# %%
data_test = DeepONet.TripleCartesianProd(x_testing, y_testing,aux_data=mask_testing, shuffle=False)
data_train = DeepONet.TripleCartesianProd(x_train, y_train,aux_data=mask_train, batch_size=64)


# %%
numNode=200
model = DeepONet.DeepONetCartesianProd(
    [2, numNode, numNode,numNode,numNode,numNode,2*numNode],
    [2, numNode, numNode,numNode,numNode,numNode,2*numNode],
    {"branch": "relu", "trunk": "tanh"},num_outputs=2,
)



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

if h is not None:
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.plot(h["loss"], label="loss")
    ax.plot(h["val_loss"], label="val_loss")
    ax.legend()
    ax.set_yscale("log")


# Plotting Results
# %%
#pcs_validate_raw=pcs_testing_raw
x_validate=(x_testing[0],x_testing[1],mask_testing)
y_validate=Talpha_raw_testing
mask_validate=x_validate[2].astype(bool)
def L2RelativeError(x_validate, y_validate):
    mask_validate=x_validate[2].astype(bool)

    y_pred_out = model.predict(x_validate)
    y_pred_raw=np.ones_like(y_pred_out)
    y_pred_raw[:,:,0]=y_pred_out[:,:,0]*scaler_T+shift_T
    y_pred_raw[:,:,1]=y_pred_out[:,:,1]
    y_pred = [y_pred_raw[i][mask_validate[i,:,0]] for i in range(y_pred_raw.shape[0])]
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
    return np.stack(error_s), y_pred # error_s:(N,)
    

error_s,y_pred = L2RelativeError(x_validate, y_validate)

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


nr, nc = 3, 2
fig = plt.figure(figsize=(18, 5))
tx_data_validate=[x_validate[1][mask_validate[i,:,0]] for i in min_median_max_index]
for i, idx in enumerate(min_median_max_index):
    t= scaler_t* tx_data_validate[i][:,0].reshape(-1,len(x_grid_file))
    Ttrue=y_validate[idx][:,0].reshape(-1,len(x_grid_file))
    Tpred=y_pred[idx][:,0].reshape(-1,len(x_grid_file))
    alpha_true=y_validate[idx][:,1].reshape(-1,len(x_grid_file))
    alpha_pred=y_pred[idx][:,1].reshape(-1,len(x_grid_file))
    Nt=len(t)
    ax = plt.subplot(nr, nc, 2*i + 1)
    # py.figure(figsize = (14,7))
    num_curv = 8
    step = (Nt - 16) / (num_curv + 1)
    curv = [int(16 + (i + 1) * step) for i in range(num_curv)]
    curv[-1] = Nt - 1
    for j, c in enumerate(curv):
        if j == 0:
            ax.plot(x_grid_file, Ttrue[c, :], "b", label="True")
            ax.plot(x_grid_file, Tpred[c, :], "r--", label="Predicted")
        else:
            ax.plot(x_grid_file, Ttrue[c, :], "b")
            ax.plot(x_grid_file, Tpred[c, :], "r--")
    ax.legend()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("T [K]")
    
    ax = plt.subplot(nr, nc, 2*i + 2)
    for j, c in enumerate(curv):
        if j == 0:
            ax.plot(x_grid_file, alpha_true[c, :], "b", label="True")
            ax.plot(x_grid_file, alpha_pred[c, :], "r--", label="Predicted")
        else:
            ax.plot(x_grid_file, alpha_true[c, :], "b")
            ax.plot(x_grid_file, alpha_true[c, :], "r--")
    ax.legend()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("alpha")
plt.tight_layout()


# %%
def animations_data():
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
    return data_video
# ani=DeepONet.get_video(data_video[0])
# HTML(ani.to_html5_video())
# ani.save('FP1D_min.mp4')
# # # %%
# ani=DeepONet.get_video(data_video[1])
# HTML(ani.to_html5_video())
# ani.save('FP1D_median.mp4')
# # # %%
# ani=DeepONet.get_video(data_video[-1])
# HTML(ani.to_html5_video())
# ani.save('FP1D_max.mp4')

# %%

# def derivative_op_wr(y,x):
#     # y shape=(np,2)
#     # x shape=(np,2) [t,x ]
#     T=y[:,0:1] # (np,1)
#     alpha=y[:,1:2] #(np,1)
#     T_X=DeepONet.jacobian(T,x) # (np,2)
#     T_x=T_X[:,1:2]*scaler_T/scaler_x # (np,1)
#     T_raw=(T*scaler_T+shift_T)
#     return T_raw,alpha,T_x

# op = DeepONet.EvaluateDeepONetPDEs(model, derivative_op_wr)

# def ErrorMeasure_wr(inputs,op,dx=dx,dt=dt):
#     A, Er, m, n, Ca, alpha_c = 8.55e15, 110750, 0.77, 1.72, 14.48, 0.405
#     rho, kappa, Cp, H=980.0, 0.152, 1600.5, 350000.0
#     R = 8.314

#     data_all=op(inputs,stack=False)
#     masks_all=inputs[2]
#     res_all=[]
#     idt1,idt2=30,-5
#     idx1,idx2=10,-10
#     for data,masks in zip(data_all,masks_all):
#         T_raw,alpha,T_x=data
#         indices=np.where(masks[:,0]==1.0)[0]
#         T_x=T_x.numpy()[indices].reshape(-1,Nx)[idt1:idt2,idx1:idx2]
        
#         alpha=alpha.numpy()[indices].reshape(-1,Nx)[idt1:idt2,idx1:idx2]
#         T_raw=T_raw.numpy()[indices].reshape(-1,Nx)[idt1:idt2,idx1:idx2]

#         alpha=np.maximum(alpha,0.0)
#         alpha=np.minimum(alpha,1.0)
        
#         exp_term = A*np.exp(-Er / (R * T_raw))
#         f_alpha = (1-alpha)**m*alpha**n/(1 + np.exp(Ca * (alpha - alpha_c)))
#         rea_rhs =  exp_term * f_alpha

        
#         item1=(sciint.simpson(T_raw[-1]-T_raw[0],dx=dx))*Cp/H
#         item2=(sciint.simpson(T_x[:,-1]-T_x[:,0],dx=dt))*kappa/(rho*H)
#         item3=sciint.simpson(alpha[-1]-alpha[0],dx=dx)
#         item4_1=sciint.simpson(rea_rhs,dx=dx,axis=-1)
#         item4=sciint.simpson(item4_1,dx=dt)
#         res=abs(item1-item2-item3)/abs(item3)+abs(item4-item3)/abs(item3)
#         res_all.append(res*0.5)
#     return np.array(res_all)
    

# res_op_val_ = ErrorMeasure_wr((x_validate[0], x_validate[1],x_validate[2]),op)
# %%


def derivative_op(y,x):
    # y shape=(np,2)
    # x shape=(np,2) [t,x ]
    T=y[:,0:1] # (np,1)
    alpha=y[:,1:2] #(np,1)
    T_X=DeepONet.jacobian(T,x) # (np,2)
    T_x=T_X[:,1:2] # (np,1)
    T_t=T_X[:,0:1]*scaler_T/scaler_t # (np,1)
    T_xX=DeepONet.jacobian(T_x,x) # (np,2)
    T_xx=T_xX[:,1:2]*scaler_T/scaler_x**2 # (np,1)
    T_raw=(T*scaler_T+shift_T)
    alpha_t=DeepONet.jacobian(alpha,x)[:,0:1]/scaler_t
    return T_raw,alpha,T_t,T_xx,alpha_t
op = DeepONet.EvaluateDeepONetPDEs(model, derivative_op)

def ErrorMeasure(inputs,op):
    A, Er, m, n, Ca, alpha_c = 8.55e15, 110750, 0.77, 1.72, 14.48, 0.405
    rho, kappa, Cp, H=980.0, 0.152, 1600.5, 350000.0
    R = 8.314

    data_all=op(inputs,stack=False)
    masks_all=inputs[2]
    res_all=[]
    idt1,idt2=30,-5
    idx1,idx2=10,-10
    for data,masks in zip(data_all,masks_all):
        T,alpha,T_t,T_xx,alpha_t=data
        indices=np.where(masks[:,0]==1.0)[0]
        T=T.numpy()[indices].reshape(-1,Nx)[idt1:idt2,idx1:idx2]
        alpha=alpha.numpy()[indices].reshape(-1,Nx)[idt1:idt2,idx1:idx2]
        T_t=T_t.numpy()[indices].reshape(-1,Nx)[idt1:idt2,idx1:idx2]
        T_xx=T_xx.numpy()[indices].reshape(-1,Nx)[idt1:idt2,idx1:idx2]
        alpha_t=alpha_t.numpy()[indices].reshape(-1,Nx)[idt1:idt2,idx1:idx2]
        res1=Cp/H*T_t-kappa/(rho*H)*T_xx-alpha_t
        r1=np.linalg.norm(res1)/np.linalg.norm(alpha_t)
        exp_term = A*np.exp(-Er / (R * T))
        alpha=np.maximum(alpha,0.0)
        alpha=np.minimum(alpha,1.0)
        f_alpha = (1-alpha)**m*alpha**n/(1 + np.exp(Ca * (alpha - alpha_c)))
        rea_rhs =  exp_term * f_alpha
        res2=rea_rhs-alpha_t
        r2=np.linalg.norm(res2)/np.linalg.norm(alpha_t)
        res_all.append(r1*0.5+r2*0.5)
    return np.array(res_all)
 
res_op_val_ = ErrorMeasure((x_validate[0], x_validate[1],x_validate[2]),op)   
# %%

gap=1
plt.plot(error_s[sort_idx][::gap],(res_op_val_)[sort_idx][::gap],'o')
# Fit a straight line
coefficients = np.polyfit(error_s[sort_idx], (res_op_val_)[sort_idx], 1)
line = np.poly1d(coefficients)

# Plot the line
plt.plot(error_s[sort_idx], (res_op_val_)[sort_idx], 'o')
plt.plot(error_s[sort_idx], line(error_s[sort_idx]), color='red')

# Add labels and title
plt.xlabel('L2 Relative Error')
plt.ylabel('Residual Error')
plt.title('L2 Relative Error vs Residual Error')
# plt.xlim(0.005, 0.02)
# plt.ylim(0.00, 0.015)
#%%
correlation = np.corrcoef(error_s, res_op_val_)[0, 1]
print("Pearson correlation coefficient:", correlation)

from scipy.stats import spearmanr
r_spearman, _ = spearmanr(error_s, res_op_val_)
print(f"Spearman's rank correlation coefficient: {r_spearman}")
# %%

# %%
# check the derivatives

def check_derivatives(inputs,op):
    A, Er, m, n, Ca, alpha_c = 8.55e15, 110750, 0.77, 1.72, 14.48, 0.405
    rho, kappa, Cp, H=980.0, 0.152, 1600.5, 350000.0
    R = 8.314

    data_all=op(inputs,stack=False)
    masks_all=inputs[2]
    lhs1=[]
    lhs2=[]
    rhs=[]
    for data,masks in zip(data_all,masks_all):
        T,alpha,T_t,T_xx,alpha_t=data
        indices=np.where(masks[:,0]==1.0)[0]
        T=T.numpy()[indices].reshape(-1,Nx)
        alpha=alpha.numpy()[indices].reshape(-1,Nx)
        alpha=np.maximum(alpha,0.0)
        alpha=np.minimum(alpha,1.0)
        T_t=T_t.numpy()[indices].reshape(-1,Nx)
        T_xx=T_xx.numpy()[indices].reshape(-1,Nx)
        alpha_t=alpha_t.numpy()[indices].reshape(-1,Nx)
        l1=Cp/H*T_t-kappa/(rho*H)*T_xx
        
        exp_term = A*np.exp(-Er / (R * T))
        f_alpha = (1-alpha)**m*alpha**n/(1 + np.exp(Ca * (alpha - alpha_c)))
        rea_rhs =  exp_term * f_alpha
        lhs1.append(l1)
        lhs2.append(alpha_t)
        rhs.append(rea_rhs)
        
    return (lhs1),(lhs2),(rhs)
# %%
lhs1,lhs2,rhs=check_derivatives(x_validate,op)
# %%
nr, nc = 3, 3
fig = plt.figure(figsize=(18, 5))
for i, idx in enumerate(min_median_max_index):
    l_eq1=lhs1[idx]
    l_eq2=lhs2[idx]
    r_eq=rhs[idx]
    ax = plt.subplot(nr, nc, 3*i + 1)
    c1=ax.contourf(l_eq1)
    cbar = fig.colorbar(c1, ax=ax)
    ax = plt.subplot(nr, nc, 3*i + 2)
    c2=ax.contourf(l_eq2)
    cbar = fig.colorbar(c2, ax=ax)
    ax = plt.subplot(nr, nc, 3*i + 3)
    c3=ax.contourf(r_eq)
    cbar = fig.colorbar(c3, ax=ax)
    
# %%
# k,c=8.0,0.0
# LR=res_op_val_
# probility = np.power(LR, k) / np.power(LR, k).mean() + c
# probility_normalized = probility / np.sum(probility)
# s_idx = np.random.choice(
#     a=np.arange(len(pcs_validate_raw)),
#     size=600,
#     replace=False,
#     p=probility_normalized,
# )

# fig=plt.figure(figsize=(12,5))
# ax=plt.subplot(1,2,1)
# c1=ax.scatter(pcs_validate_raw[s_idx,0],pcs_validate_raw[s_idx,1],c=res_op_val_[s_idx],cmap='jet')
# cbar = fig.colorbar(c1, ax=ax)
# ax=plt.subplot(1,2,2)
# c2=ax.scatter(pcs_validate_raw[s_idx,0],pcs_validate_raw[s_idx,1],c=error_s[s_idx],cmap='jet')
# cbar = fig.colorbar(c2, ax=ax)
# %%
