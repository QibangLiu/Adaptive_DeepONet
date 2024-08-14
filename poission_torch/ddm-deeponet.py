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

# In[]
filebase = "./saved_model/pytorch_ddm2D"
os.makedirs(filebase, exist_ok=True)

# In[3]:

Nx = 128
Ny = 128
m = Nx * Ny
###  N number of samples 1000
###  m number of points 40
###  P number of output sensors (coordinates) 1600
### x_train is a tuple of u0(N,m) and output sensores, all coordinates xy_train_testing(P,2)
### y_train is target solutions (our s) u(N,P)


# tf.keras.utils.set_random_seed(seed)
fenics_data = scio.loadmat("../Adap_possion/TrainingData/poisson_gauss_cov20k.mat")

x_grid = fenics_data["x_grid"].astype(np.float32)  # shape (Ny, Nx)
y_grid = fenics_data["y_grid"].astype(np.float32)
source_terms_raw = fenics_data["source_terms"].astype(np.float32)  # shape (N, Ny, Nx)
source_terms_raw = source_terms_raw.reshape(-1, Nx * Ny)
solutions_raw = fenics_data["solutions"].astype(np.float32)  # shape (N, Ny, Nx)
solutions_raw = solutions_raw.reshape(-1, Nx * Ny)
# shift_solution,scaler_solution = np.min(solutions_raw),np.max(solutions_raw)-np.min(solutions_raw)
# shift_source,scaler_source = np.min(source_terms_raw),np.max(source_terms_raw)-np.min(source_terms_raw)
# shift_solution, scaler_solution = 0, 1
# shift_source, scaler_source = 0, 1
shift_solution, scaler_solution = np.mean(solutions_raw), np.std(solutions_raw) 
shift_source, scaler_source = np.mean(source_terms_raw), np.std(source_terms_raw)
solutions = (solutions_raw - shift_solution) / scaler_solution
source_terms = (source_terms_raw - shift_source) / scaler_source
num_train = 6000
u0_train = source_terms[:num_train]
u0_testing = source_terms[-1000:]
s_train = solutions[:num_train]
s_testing = solutions[-1000:]

u0_testing_raw = source_terms_raw[-1000]
u0_train_raw = source_terms_raw[:num_train]
s_testing_raw = solutions_raw[-1000:1000]
s_train_raw = solutions_raw[:num_train]

xy_train_testing = np.concatenate(
    [x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)], axis=1
)


print("u0_train.shape = ", u0_train.shape)
print("type of u0_train = ", type(u0_train))
print("u0_testing.shape = ", u0_testing.shape)
print("s_train.shape = ", s_train.shape)
print("s_testing.shape = ", s_testing.shape)
print("xy_train_testing.shape", xy_train_testing.shape)


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train = (torch.tensor(u0_train).to(device), torch.tensor(xy_train_testing).to(device))
y_train = torch.tensor(s_train).to(device)
aux=x_train[0]
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
    [m, 100, 100, 100, 100, 100, 100],
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

# h = model.fit(train_loader,test_loader, epochs=1000, callbacks=checkpoint_callback)
# model.save_logs(filebase)
# %%

model.load_weights(checkpoint_fname, device)
h= model.load_logs(filebase)
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["loss"], label="loss")
ax.plot(h["val_loss"], label="val_loss")
ax.legend()
ax.set_yscale("log")

# %%
x_validate = x_train
y_validate = s_train_raw
u0_validate = u0_train_raw

input_branch, input_trunk = x_validate[0], x_validate[1]
input_trunk.requires_grad_(True)

y_pred_out = model((input_branch, input_trunk))
y_pred = scaler_solution * y_pred_out.detach().cpu().numpy() + shift_solution

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
def ResidualError(y, x, aux=None):
    dydx2 = DeepONet.laplacian(y, x)
    return -0.01 * (dydx2) * scaler_solution - aux


res_op = DeepONet.EvaluateDeepONetPDEs(ResidualError, model=model)
res_op_val = res_op(
    (input_branch[min_median_max_index], input_trunk),aux=aux[min_median_max_index]
)
laplace_op_val = res_op_val.detach().cpu().numpy()
# %%
#laplace_op_val=res_op_val
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
def LaplaceOperator2D(y, x, aux=None):
    dydx2 = DeepONet.laplacian(y, x)
    return -0.01 * (dydx2) * scaler_solution
# %%
laplace_op = DeepONet.EvaluateDeepONetPDEs(LaplaceOperator2D,model)
laplace_op_val = laplace_op(
    (input_branch[min_median_max_index], input_trunk)
)
laplace_op_val = laplace_op_val.detach().cpu().numpy()

# %%

# %%

dx = x_grid[0, 1] - x_grid[0, 0]
dy = y_grid[1, 0] - y_grid[0, 0]
out=model((input_branch[min_median_max_index], input_trunk))
laplace_FD_val=-0.01*DeepONet.laplacian_FD(out.reshape(-1,Ny,Nx),dx,dy)*scaler_solution
laplace_FD_val=laplace_FD_val.detach().cpu().numpy()
out_ref=y_validate[min_median_max_index]
laplace_FD_true=-0.01*DeepONet.laplacian_FD(out_ref.reshape(-1,Ny,Nx),dx,dy)

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

    ax = plt.subplot(nr, nc, nc * i + 2)
    # py.figure(figsize = (14,7))
    c3 = ax.contourf(
        x_grid[1:-1, 1:-1],
        y_grid[1:-1, 1:-1],
        laplace_FD_true[i],
        20,
        vmax=vmax,
        vmin=vmin,
        cmap="jet",
    )
    ax.set_title(r"Source Distrubution by FD of Reference Solution")
    cbar = fig.colorbar(c3, ax=ax)
    plt.tight_layout()

    ax = plt.subplot(nr, nc, nc * i + 3)
    # py.figure(figsize = (14,7))
    c2 = ax.contourf(
        x_grid,
        y_grid,
        np.abs(laplace_op_val[i].reshape(Ny, Nx)-u0_validate[index].reshape(Ny,Nx)),
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
        x_grid[1:-1, 1:-1],
        y_grid[1:-1, 1:-1],
        laplace_FD_val[i],
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



# %%
