# %%

import sys
import os
import DeepONet_tf as DeepONet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.io as scio
import h5py
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

# In[3]:
train=True
filebase = "./saved_model/test1D"
# %%
hf=h5py.File('./TrainingData/RD_gauss_cov40k.h5', 'r')
x_grid=hf['x_grid'][:].squeeze().astype(np.float32)# shape (Nx,)
t_grid = hf["t_grid"][:].squeeze().astype(np.float32)
dx=x_grid[1]-x_grid[0]
dt=t_grid[1]-t_grid[0]
Nx, Nt = len(x_grid), len(t_grid)
m = Nx * Nt
x_grid, t_grid = np.meshgrid(x_grid, t_grid)
ICs_raw = hf["u0s"][:].astype(np.float32) 
Ks_raw = hf["Ks"][:].astype(np.float32)
solutions_raw = hf["solutions"][:].astype(np.float32)  # shape (N, Nt, Nx)
solutions_raw=solutions_raw.reshape(-1,m)
hf.close()


shift_solution, scaler_solution = np.mean(solutions_raw), np.std(solutions_raw) 
shift_ICs, scaler_ICs= np.mean(ICs_raw), np.std(ICs_raw)
shift_Ks, scaler_Ks = np.mean(Ks_raw), np.std(Ks_raw)
solutions = (solutions_raw - shift_solution) / scaler_solution
ICs = (ICs_raw - shift_ICs) / scaler_ICs
Ks = (Ks_raw - shift_Ks) / scaler_Ks
u0s=np.concatenate([ICs,Ks],axis=1)
u0s_raw=np.concatenate([ICs_raw,Ks_raw],axis=1)
# %%


num_train = 10000
num_test = 1000
u0_train = u0s[:num_train]
u0_train_raw = u0s_raw[:num_train]  
u0_testing = u0s[-num_test:]
u0_testing_raw = u0s_raw[-num_test:]
s_train = solutions[:num_train]
s_testing = solutions[-num_test:]
s_train_raw = solutions_raw[:num_train]
s_testing_raw = solutions_raw[-num_test:]


xy_train_testing = np.concatenate(
    [x_grid.reshape(-1, 1), t_grid.reshape(-1, 1)], axis=1
)

# %%
x_train = (u0_train, xy_train_testing)
y_train = s_train

x_testing = (u0_testing, xy_train_testing)
y_testing = s_testing

# %%
data_test = DeepONet.TripleCartesianProd(x_testing, y_testing, shuffle=False)
data_train = DeepONet.TripleCartesianProd(x_train, y_train, batch_size=128)


# %%
numNode=200
model = DeepONet.DeepONetCartesianProd(
    [2*Nx, numNode/2, numNode,numNode,numNode,numNode,numNode],
    [2, numNode, numNode,numNode,numNode,numNode,numNode],
    {"branch": "relu", "trunk": "tanh"}
)

# %%

start_time = timeit.default_timer()

optm = tf.keras.optimizers.Adam(learning_rate=1e-4)
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

# %%

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

fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["loss"], label="loss")
ax.plot(h["val_loss"], label="val_loss")
ax.legend()
ax.set_yscale("log")



# %%
x_validate = x_testing
y_validate = s_testing_raw
u0_validate = u0_testing_raw

# %%
def L2RelativeError(x_validate,y_validate,scaler_solution,shift_solution):
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


error_s,y_pred=L2RelativeError(x_validate,y_validate,scaler_solution,shift_solution)
fig = plt.figure()
_ = plt.hist(error_s)


# %%
sort_idx = np.argsort(error_s)
min_index = sort_idx[0]
max_index = sort_idx[-1]
median_index = sort_idx[len(sort_idx) // 2]

# Print the indexes
print("Index for minimum element:", min_index, "with error", error_s[min_index])
print("Index for maximum element:", max_index, "with error", error_s[max_index])
print("Index for median element:", median_index, "with error", error_s[median_index])


min_median_max_index = np.array([min_index, median_index, max_index])

nr, nc = 1, 3
fig = plt.figure(figsize=(18, 5))
for i, index in enumerate(min_median_max_index):
    y_validate_nx_ny = y_validate[index].reshape(Nt, Nx)
    s_pred_nx_ny = y_pred[index].reshape(Nt, Nx)

    ax = plt.subplot(nr, nc, i + 1)
    # py.figure(figsize = (14,7))
    num_curv = 8
    step = (Nt - 16) / (num_curv + 1)
    curv = [int(16 + (i + 1) * step) for i in range(num_curv)]
    curv[-1] = Nt - 1
    for j, c in enumerate(curv):
        if j == 0:
            ax.plot(x_grid[c, :], y_validate_nx_ny[c, :], "b", label="True")
            ax.plot(x_grid[c, :], s_pred_nx_ny[c, :], "r--", label="Predicted")
        else:
            ax.plot(x_grid[c, :], y_validate_nx_ny[c, :], "b")
            ax.plot(x_grid[c, :], s_pred_nx_ny[c, :], "r--")
    ax.legend()
plt.tight_layout()



# %%
alpha=0.01
def lhs_eqn(y, x, aux=None):
    # nb: batch size, =Nt*Nx
    # nx: dimension of trunck input, =2
    dYdX = tf.gradients(y, x)[0] #(nb,nx)
    dYdx = dYdX[:, 0:1] # (nb,1)
    dYdt = dYdX[:, 1:2] # (nb,1)
    dYdxdX = DeepONet.jacobian(dYdx, x) #(nb,nx)
    dYdx2=dYdxdX[:,0:1] # (nb,1)
    #
    k=tf.squeeze(aux[Nx:])[None,:] # (1,Nx)
    y_inv=scaler_solution*y+shift_solution # (nb,1)
    y_inv=k*(tf.reshape(y_inv,(Nt,Nx)))# (Nt,Nx)
    lhs=(dYdt*scaler_solution-alpha*dYdx2*scaler_solution)
    lhs = tf.reshape(lhs, (Nt, Nx))
    lhs=lhs-y_inv
    lhs = tf.reshape(lhs, [-1, 1])
    return lhs


lhs_op = DeepONet.EvaluateDeepONetPDEs(model,lhs_eqn)
lhs = lhs_op( x_validate,u0_validate)

# %%
nr, nc = 1, 3
fig = plt.figure(figsize=(18, 5))
for i, index in enumerate(min_median_max_index):
    source_validate = u0_validate[index][:Nx]
    source_ad = lhs[index].reshape(Nt, Nx)

    ax = plt.subplot(nr, nc, i + 1)
    # py.figure(figsize = (14,7))
    num_curv = 5
    step = (Nt - 16) / (num_curv + 1)
    curv = [int(16 + (i + 1) * step) for i in range(num_curv)]
    curv[-1] = Nt - 1
    for j, c in enumerate(curv):
        if j == 0:
            ax.plot(x_grid[c, :], source_validate, "b", label="True")
            ax.plot(x_grid[c, :], source_ad[c, :], "r--", label="AD")
        else:
            ax.plot(x_grid[c, :], source_validate, "b")
            ax.plot(x_grid[c, :], source_ad[c, :], "r--")
    ax.legend()
    ax.set_title(f"b = %.2f" % u0_validate[index, 0])
plt.tight_layout()

# %%
def ResidualError(y, x, aux=None):
    # nb: batch size, =Nt*Nx
    # nx: dimension of trunck input, =2
    dYdX = tf.gradients(y, x)[0] #(nb,nx)
    dYdx = dYdX[:, 0:1] # (nb,1)
    dYdt = dYdX[:, 1:2] # (nb,1)
    dYdxdX = DeepONet.jacobian(dYdx, x) #(nb,nx)
    dYdx2=dYdxdX[:,0:1] # (nb,1)
    #
    
    u0=tf.squeeze(aux[:Nx]) # (Nx,)
    k=tf.squeeze(aux[Nx:]) # (Nx,)
    
    
    y_inv=scaler_solution*y+shift_solution
    y_inv=k*tf.reshape(y_inv,(Nt,Nx)) #(Nt,Nx)
    rhs=u0[None,:]+y_inv # (Nt,Nx)
    rhs=tf.reshape(rhs,[-1,1])
    
    lhs=(dYdt*scaler_solution-alpha*dYdx2*scaler_solution)
    
    res=tf.norm(lhs-rhs)/tf.norm(rhs)
    
    return res

res_op = DeepONet.EvaluateDeepONetPDEs(model,ResidualError)
res = res_op(
    x_validate, aux=u0_validate
)
# %%
correlation = np.corrcoef(error_s, res)[0, 1]
print("Pearson correlation coefficient:", correlation)

from scipy.stats import spearmanr
r_spearman, _ = spearmanr(error_s, res)
print(f"Spearman's rank correlation coefficient: {r_spearman}")

gap=2
fig = plt.figure()
ax=plt.subplot(1,1,1)
ax.plot(error_s[sort_idx][::gap],(res)[sort_idx][::gap],'o')
# Fit a straight line
coefficients = np.polyfit(error_s[sort_idx], (res)[sort_idx], 1)
line = np.poly1d(coefficients)

# Plot the line
ax.plot(error_s[sort_idx], (res)[sort_idx], 'o')
ax.plot(error_s[sort_idx], line(error_s[sort_idx]), color='red')

# Add labels and title
ax.set_xlabel('L2 Relative Error')
ax.set_ylabel('Residual Error')
ax.set_title('L2 Relative Error vs Residual Error')

# %%
