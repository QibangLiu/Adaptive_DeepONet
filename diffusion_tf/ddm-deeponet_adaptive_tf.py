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
import tensorflow as tf
import timeit
import h5py
from scipy.stats import spearmanr
import json
import pandas as pd
# %run ddm-deeponet_adaptive_tf.py 200 0 2 0 0 0
# In[]

prefix_filebase = "./saved_model"
str_dN, str_start, str_end = sys.argv[1:4]
str_k, str_c = sys.argv[4:-1]
str_caseID = sys.argv[-1]

# In[3]:

# fenics_data = scio.loadmat("./TrainingData/diffusion_gauss_cov20k.mat")
# x_grid = fenics_data["x_grid"].squeeze().astype(np.float32)  # shape (Nx,)
# t_grid = fenics_data["t_grid"].squeeze().astype(np.float32)
# ICs_raw = fenics_data["ICs"].astype(np.float32) 
# solutions_raw = fenics_data["solutions"].astype(np.float32)  # shape (N, Nt, Nx)
hf=h5py.File("./TrainingData/diffusion_gauss_cov40k.h5", 'r')
x_grid=hf['x_grid'][:].squeeze().astype(np.float32)  # shape (Nx,)\
t_grid = hf["t_grid"][:].squeeze().astype(np.float32)
ICs_raw = hf["ICs"][:].astype(np.float32) 
solutions_raw = hf["solutions"][:].astype(np.float32)  # shape (N, Nt, Nx)
hf.close()

dx=x_grid[1]-x_grid[0]
dt=t_grid[1]-t_grid[0]
Nx, Nt = len(x_grid), len(t_grid)
m = Nx * Nt
x_grid, t_grid = np.meshgrid(x_grid, t_grid)
solutions_raw=solutions_raw.reshape(-1,m)


shift_solution, scaler_solution = np.mean(solutions_raw), np.std(solutions_raw) 
shift_ICs, scaler_ICs= np.mean(ICs_raw), np.std(ICs_raw)
solutions = (solutions_raw - shift_solution) / scaler_solution
ICs = (ICs_raw - shift_ICs) / scaler_ICs
# %%

num_test = 1000
num_train = -num_test

u0_train = ICs[:num_train]
u0_train_raw = ICs_raw[:num_train]  
u0_testing = ICs[-num_test:]
u0_testing_raw = ICs_raw[-num_test:]
s_train = solutions[:num_train]
s_testing = solutions[-num_test:]
s_train_raw = solutions_raw[:num_train]
s_testing_raw = solutions_raw[-num_test:]


xy_train_testing = np.concatenate(
    [x_grid.reshape(-1, 1), t_grid.reshape(-1, 1)], axis=1
)

# %%

x_test = (u0_testing, xy_train_testing)
y_test = s_testing
data_test = DeepONet.TripleCartesianProd(x_test, y_test, shuffle=False)
# %%

numNode=200
model = DeepONet.DeepONetCartesianProd(
    [Nx, numNode, numNode,numNode,numNode,numNode,numNode],
    [2, numNode, numNode,numNode,numNode,numNode,numNode],
    {"branch": "relu", "trunk": "tanh"}
)
# %%
alpha=0.01
def ResidualError(y, x, aux=None):
    # nb: batch size, =Nt*Nx
    # nx: dimension of trunck input, =2
    dYdX = tf.gradients(y, x)[0] #(nb,nx)
    dYdx = dYdX[:, 0:1] # (nb,1)
    dYdt = dYdX[:, 1:2] # (nb,1)
    dYdxdX = DeepONet.jacobian(dYdx, x) #(nb,nx)
    dYdx2=dYdxdX[:,0:1] # (nb,1)
    #
    y_inv=scaler_solution*y+shift_solution
    lhs=(dYdt*scaler_solution+y_inv-alpha*dYdx2*scaler_solution)
    lhs = tf.reshape(lhs, (Nt, Nx))
    aux_norm = tf.norm(aux)
    l2s = [tf.norm(lhs[i, :] - tf.squeeze(aux)) / aux_norm for i in range(lhs.shape[0])]
    
    res = tf.reduce_mean(tf.stack(l2s))
    return res

res_op = DeepONet.EvaluateDeepONetPDEs(model,ResidualError)


def ErrorMeasure(X, aux):
    res = res_op(X, aux=aux)
    return res


def sampling(iter, dN, pre_filebase):
    all_data_idx = np.arange(len(u0_train))
    currTrainDataIDX__ = None
    if iter == 0:
        currTrainDataIDX__ = np.random.choice(a=all_data_idx, size=dN, replace=False)
    else:
        pre_train_data_idx = np.genfromtxt(
            os.path.join(pre_filebase, "trainDataIDX.csv"), dtype=int, delimiter=","
        )
        # potential training data
        potential_train_data_idx = np.delete(all_data_idx, pre_train_data_idx)

        LR = ErrorMeasure(
            (u0_train[potential_train_data_idx], xy_train_testing),
            u0_train_raw[potential_train_data_idx]
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

def L2RelativeError(x_validate,y_validate):
    y_pred = model.predict(x_validate)
    y_pred = y_pred * scaler_solution + shift_solution
    error_s = []
    for i in range(len(y_validate)):
        error_s_tmp = np.linalg.norm(y_validate[i] - y_pred[i]) / np.linalg.norm(
            y_validate[i]
        )
        error_s.append(error_s_tmp)
    error_s = np.stack(error_s)
    return error_s, y_pred  

# %%
numCase_add = int(str_dN)
k = float(str_k)
c = float(str_c)

iter_start, iter_end = int(str_start), int(str_end)

project_name = (
    "adapt_k" + str_k + "c" + str_c + "dN" + str_dN + "case" + str_caseID
)
filebase = os.path.join(prefix_filebase, project_name)
start_time = timeit.default_timer()

if iter_start != 0:
    pre_filebase = os.path.join(filebase, "iter" + str(iter_start - 1))
    model.load_history(pre_filebase)
    model.load_weights(os.path.join(pre_filebase, "model.ckpt"))

correla_file=os.path.join(filebase, "correlation.csv")
if os.path.exists(correla_file):
    correla_hist = pd.read_csv(correla_file).to_dict(orient="list")
else:
    correla_hist = {"num_sample":[], "spearman": [], "pearson": []}

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
    data_train = DeepONet.TripleCartesianProd(x_train, curr_y_train, batch_size=64)
    optm = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optm, loss="mse")
    checkpoint_fname = os.path.join(current_filebase, "model.ckpt")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_fname,
        save_weights_only=True,
        monitor="loss",
        verbose=0,
        save_freq="epoch",
        save_best_only=True,
        mode="min",
    )
    # model.set_weights(initial_weights)
    h = model.fit(
        data_train.dataset,
        validation_data=data_test.dataset,
        epochs=1000,
        verbose=2,
        callbacks=[model_checkpoint],
    )
    model.save_history(filebase)
    model.load_weights(checkpoint_fname)
    
    error_test,_=L2RelativeError(data_test.X_data,s_testing_raw)
    np.savetxt(
        os.path.join(current_filebase, "TestL2Error.csv"),
        error_test,
        fmt="%.4e",
        delimiter=",",
    )
    
    res_test = ErrorMeasure(
            data_test.X_data,
            u0_testing_raw
        )
    
    r_spearman, _ = spearmanr(error_test, res_test)
    print(f"Spearman's rank correlation coefficient: {r_spearman}")
    pearson_coe = np.corrcoef(error_test, res_test)[0, 1]
    print("Pearson correlation coefficient:", pearson_coe)
    
    correla_hist["num_sample"].append(len(currTrainDataIDX))
    correla_hist["spearman"].append(r_spearman)
    correla_hist["pearson"].append(pearson_coe)

    
    error_train,_=L2RelativeError(data_train.X_data,s_train_raw[currTrainDataIDX])
    np.savetxt(
        os.path.join(current_filebase, "TrainL2Error.csv"),
        error_train,
        fmt="%.4e",
        delimiter=",",
    )
    
    df = pd.DataFrame(correla_hist)
    df.to_csv(os.path.join(filebase, "correlation.csv"), index=False)
    
    stop_time = timeit.default_timer()
    print("training Run time so far: ", round(stop_time - start_time, 2), "(s)")
 


fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["loss"], label="loss")
ax.plot(h["val_loss"], label="val_loss")
ax.legend()
ax.set_yscale("log")
