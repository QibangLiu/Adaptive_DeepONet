# %%
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.io as scio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import DeepONet
import scipy.integrate as sciint

# In[]
filebase = "./saved_model/test1D"
os.makedirs(filebase, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
# In[3]:

Nx = 128
Nt = 128
m = Nx * Nt

x_grid = np.linspace(0, 1, Nx, dtype=np.float32)
t_grid = np.linspace(0, 1, Nt, dtype=np.float32)
dx=x_grid[1]-x_grid[0]
dt=t_grid[1]-t_grid[0]
x_grid, t_grid = np.meshgrid(x_grid, t_grid)
bmin, bmax = 1.0, 40.0
b_raw = np.random.uniform(bmin, bmax, 20000).reshape(-1, 1).astype(np.float32)
scaler_b, shift_b = bmax - bmin, bmin
b = (b_raw - shift_b) / scaler_b
# %%
source_terms = []
solutions = []
for param in b_raw:
    source_terms.append(
        (param - 2 * param**2 * np.tanh(param * (x_grid - t_grid)))
        / (2 * np.cosh(param * (x_grid - t_grid)) ** 2)
    )
    solutions.append(0.5 * (1 - np.tanh(param * (x_grid - t_grid))))
source_terms = np.array(source_terms)
source_terms = source_terms.reshape(-1, Nx * Nt)
solutions = np.array(solutions).reshape(-1, Nx * Nt)

num_train = 6000
u0_train = b[:num_train]
u0_testing = b[-1000:]
s_train = solutions[:num_train]
s_testing = solutions[-1000:]

u0_testing_raw = b_raw[-1000:]
u0_train_raw = b_raw[:num_train]

xy_train_testing = np.concatenate(
    [x_grid.reshape(-1, 1), t_grid.reshape(-1, 1)], axis=1
)

source_terms_train = source_terms[:num_train]
source_terms_testing = source_terms[-1000:]

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train = (torch.tensor(u0_train).to(device), torch.tensor(xy_train_testing).to(device))
y_train = torch.tensor(s_train).to(device)

x_test = (
    torch.tensor(u0_testing).to(device),
    torch.tensor(xy_train_testing).to(device),
)
y_test = torch.tensor(s_testing).to(device)

dataset_train = DeepONet.TripleCartesianProd(x_train, y_train)
dataset_test = DeepONet.TripleCartesianProd(x_test, y_test)
# %%
train_loader = DataLoader(
    dataset_train,
    batch_size=128,
    shuffle=True,
    collate_fn=dataset_train.custom_collate_fn,
)

test_loader = DataLoader(
    dataset_test,
    batch_size=dataset_test.__len__(),
    collate_fn=dataset_test.custom_collate_fn,
)

# %%
mse = torch.nn.MSELoss()
model = DeepONet.DeepONetCartesianProd(
    [1, 100, 100, 100, 100, 100, 100],
    [2, 100, 100, 100, 100, 100, 100],
    activation={"branch": nn.ReLU(), "trunk": nn.Tanh()},
)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
model.compile(optimizer=optimizer, loss=mse)

model.to(device)
checkpoint_fname = os.path.join(filebase, "model.ckpt")
checkpoint_callback = DeepONet.ModelCheckpoint(
    checkpoint_fname, monitor="val_loss", save_best_only=True
)

# %%

h = model.fit(train_loader, test_loader, epochs=1000, callbacks=checkpoint_callback)
model.save_logs(filebase)
# %%

model.load_weights(checkpoint_fname, device)
h = model.load_logs(filebase)
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["loss"], label="loss")
ax.plot(h["val_loss"], label="val_loss")
ax.legend()
ax.set_yscale("log")

# %%
x_validate = x_train
y_validate = s_train
u0_validate = u0_train_raw
source_terms_validate = source_terms_train

input_branch, input_trunk = x_validate[0], x_validate[1]
input_trunk.requires_grad_(True)

y_pred_out = model((input_branch, input_trunk))
y_pred = y_pred_out.detach().cpu().numpy()

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
    num_curv = 5
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
    ax.set_title(f"b = %.2f" % u0_validate[index, 0])
plt.tight_layout()


# %%


# %%
def ResidualError(y, x, aux=None):
    dYdX = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0] #(nb,nx)
    dYdx = dYdX[:, 0:1] # (nb,1)
    dYdt = dYdX[:, 1:2] # (nb,1)
    dYdxdX = DeepONet.jacobian(dYdx, x, create_graph=True) #(nb,nx)
    dYdx2=dYdxdX[:,0:1] # (nb,1)
    
    return dYdt - dYdx2


res_op = DeepONet.EvaluateDeepONetPDEs(ResidualError, model=model)
res_op_val = res_op(
    (input_branch, input_trunk), aux=None
)
laplace_op_val = res_op_val.detach().cpu().numpy()
# %%
nr, nc = 1, 3
fig = plt.figure(figsize=(18, 5))
for i, index in enumerate(min_median_max_index):
    source_validate = source_terms_validate[index].reshape(Nt, Nx)
    source_ad = laplace_op_val[index].reshape(Nt, Nx)

    ax = plt.subplot(nr, nc, i + 1)
    # py.figure(figsize = (14,7))
    num_curv = 5
    step = (Nt - 16) / (num_curv + 1)
    curv = [int(16 + (i + 1) * step) for i in range(num_curv)]
    curv[-1] = Nt - 1
    for j, c in enumerate(curv):
        if j == 0:
            ax.plot(x_grid[c, :], source_validate[c, :], "b", label="True")
            ax.plot(x_grid[c, :], source_ad[c, :], "r--", label="AD")
        else:
            ax.plot(x_grid[c, :], source_validate[c, :], "b")
            ax.plot(x_grid[c, :], source_ad[c, :], "r--")
    ax.legend()
    ax.set_title(f"b = %.2f" % u0_validate[index, 0])
plt.tight_layout()

# %%
res=np.linalg.norm(laplace_op_val.squeeze() - source_terms_validate, axis=1) / np.linalg.norm(source_terms_validate, axis=1)

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



# %%

# %%
def derivative_op(y,x):
    dYdX = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0] #(nb,nx)
    dYdx = dYdX[:, 0:1] # (nb,1)
    return dYdx
op = DeepONet.EvaluateDeepONetPDEs(derivative_op, model=model)

def ErrorMeasure(inputs,op,dx=dx,dt=dt,aux=None):
    dYdx=op(inputs,aux=None)
    dYdx=dYdx.detach().cpu().numpy().squeeze().reshape(-1,Nt,Nx)
    Y = model(inputs)
    Y = y_pred_out.detach().cpu().numpy().squeeze().reshape(-1,Nt,Nx)
    int_dYdt_dt=Y[:,-1,:]-Y[:,0,:]
    int_dYdt_dxdt=sciint.simpson(int_dYdt_dt,dx=dx,axis=1)
    int_dYdx_ds=dYdx[:,:,-1]-dYdx[:,:,0]
    int_dYdx_dsdt=sciint.simpson(int_dYdx_ds,dx=dt,axis=1)
    source_validate = aux.reshape(-1,Nt, Nx)
    int_f_dx=sciint.simpson(source_validate,dx=dx,axis=2)
    int_f_dxdt=sciint.simpson(int_f_dx,dx=dt,axis=1)
    error=int_dYdt_dxdt-int_dYdx_dsdt-int_f_dxdt
    return error

res_error = ErrorMeasure(x_validate,op,aux=source_terms_validate)    
# %%
correlation = np.corrcoef(error_s, res_error)[0, 1]
print("Pearson correlation coefficient:", correlation)


r_spearman, _ = spearmanr(error_s, res_error)
print(f"Spearman's rank correlation coefficient: {r_spearman}")

gap=2
fig = plt.figure()
ax=plt.subplot(1,1,1)
ax.plot(error_s[sort_idx][::gap],(res_error)[sort_idx][::gap],'o')
# Fit a straight line
coefficients = np.polyfit(error_s[sort_idx], (res_error)[sort_idx], 1)
line = np.poly1d(coefficients)

# Plot the line
ax.plot(error_s[sort_idx], (res_error)[sort_idx], 'o')
ax.plot(error_s[sort_idx], line(error_s[sort_idx]), color='red')

# Add labels and title
ax.set_xlabel('L2 Relative Error')
ax.set_ylabel('Residual Error')
ax.set_title('L2 Relative Error vs Residual Error')
# %%