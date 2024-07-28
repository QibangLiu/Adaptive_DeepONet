

# %%
import DeepONet
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.io as scio
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as  nn

# dde.config.set_default_float("float64")
# In[]
filebase = (
    "./saved_model/pytorch_test_PI_2D"
)
os.makedirs(filebase, exist_ok=True)

# In[3]:

# tf.keras.utils.set_random_seed(seed)
fenics_data = scio.loadmat(
    "../Adap_possion/TrainingData/poisson.mat"
)
gap=4
x_grid_full = fenics_data["x_grid"].astype(np.float32)  
y_grid_full = fenics_data["y_grid"].astype(np.float32)
x_idx=np.arange(0,x_grid_full.shape[1],gap)
x_idx=np.append(x_idx,x_grid_full.shape[1]-1)
y_idx=x_idx[:,None]
x_idx=x_idx[None,:]

x_grid=x_grid_full[y_idx,x_idx]# shape (Ny, Nx)
y_grid=y_grid_full[y_idx,x_idx]# shape (Ny, Nx)
Ny,Nx=x_grid.shape
source_terms_raw_full = fenics_data["source_terms"].astype(np.float32)  # shape (N, Ny, Nx)
source_terms_raw = source_terms_raw_full[:,y_idx,x_idx].reshape(-1, Nx * Ny)
solutions_raw_full = fenics_data["solutions"].astype(np.float32)  # shape (N, Ny, Nx)
solutions_raw = solutions_raw_full[:,y_idx,x_idx].reshape(-1, Nx * Ny)
# scaler_source = 0.5 * (np.max(source_terms_raw) - np.min(source_terms_raw))
# scaler_solution = 0.5 * (np.max(solutions_raw) - np.min(solutions_raw))
scaler_source = 1
scaler_solution = 1
solutions = solutions_raw / scaler_solution
source_terms = source_terms_raw / scaler_source
num_train=5000
u0_train = source_terms[:num_train]
u0_testing = source_terms[5000:6000]
s_train = solutions[:num_train]
s_testing = solutions[5000:6000]

u0_testing_raw = source_terms_raw[5000:6000]
u0_train_raw = source_terms_raw[:num_train]
s_testing_raw = solutions_raw[5000:6000]
s_train_raw = solutions_raw[:num_train]

xy_train_testing = np.concatenate(
    [x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)], axis=1
)

boundary_mask = (xy_train_testing[:,0] == x_grid.min()) | (xy_train_testing[:,0] == x_grid.max()) \
    | (xy_train_testing[:,1] == y_grid.min())  | (xy_train_testing[:,1] == y_grid.max())
interior_mask = ~boundary_mask
boundary_indices = np.where(boundary_mask)[0]
interior_indices = np.where(interior_mask)[0]

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
aux_train = torch.tensor(u0_train_raw).to(device)
x_test = (
    torch.tensor(u0_testing).to(device),
    torch.tensor(xy_train_testing).to(device),
)
y_test = torch.tensor(s_testing).to(device)
aux_test = torch.tensor(u0_testing_raw).to(device)

dataset_train = DeepONet.TripleCartesianProd(x_train, y_train,aux_train)
#dataset_test = DeepONet.TripleCartesianProd(x_test, y_test,u0_testing_raw)

train_loader = DataLoader(
    dataset_train,
    batch_size=128,
    shuffle=True,
    #generator=torch.Generator(device=device),
    collate_fn=dataset_train.custom_collate_fn,
)

# test_loader = DataLoader(
#     dataset_test,
#     batch_size=dataset_test.__len__(),
#     generator=torch.Generator(device=device),
#     collate_fn=dataset_train.custom_collate_fn,
# )

# %%


class BCLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self, indices, bc_v, size_average=None, reduce=None, reduction: str = "mean"
    ):
        super().__init__(size_average, reduce, reduction)
        self.indices = indices
        self.bc_v = torch.tensor(bc_v).to(device)

    def forward(self, y_pred=None):
        return torch.mean(torch.square(y_pred[:, self.indices] - self.bc_v))


bc_loss = BCLoss(boundary_indices, 0)


def equation(y,x, f):
    #dy_xx = dde.grad.hessian(y, x)
    dy_xx=DeepONet.laplacian(y, x)
    return -0.01*dy_xx - f

# import deepxde as dde
# def equation(y,x,f):
#     dydxx=dde.grad.hessian(y, x, i=0, j=0)
#     dydyy=dde.grad.hessian(y, x, i=1, j=1)
#     #print('shapes', dydxx.shape, dydyy.shape, f.shape)
#     return -0.01*(dydxx+dydyy)-f
        
class PI_DeepONet(DeepONet.DeepONetCartesianProd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_losses(self, data):
        inputs, _,aux, = data
        input_branch, input_trunk = inputs
        input_trunk.requires_grad_(True)
        out = self((input_branch, input_trunk))
        pde_losses = []
        for aux_, y in zip(aux, out):
            res = equation(y[:, None], input_trunk,  aux_[:, None])
            pde_losses.append(torch.mean(torch.square(res)))
        pde_loss_ms = torch.mean((torch.stack(pde_losses)))
        bc_l = bc_loss(out)
        tot_loss = pde_loss_ms + bc_l
        loss_dic = {
            "loss": tot_loss.item(),
            "pde_loss": pde_loss_ms.item(),
            "bc_loss": bc_l.item(),
        }
        return tot_loss, loss_dic



# %%
model = PI_DeepONet(
    [Nx*Ny, 100, 100, 100, 100,100,100],
    [2, 100, 100, 100, 100,100,100],activation={"branch": nn.ReLU(), "trunk": nn.Tanh()},
)

checkpoint_fname = os.path.join(filebase, "model.ckpt")
checkpoint_callback = DeepONet.ModelCheckpoint(
    checkpoint_fname, monitor="loss", save_best_only=True
)

model.load_weights(checkpoint_fname,device)
h=model.load_logs(filebase)
# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.compile(optimizer=optimizer, loss=nn.MSELoss())  # torch.optim.Adam
model.to(device)


torch.cuda.empty_cache()  # Clear cached memory
#h = model.fit(train_loader, device=device, epochs=1000,callbacks=checkpoint_callback)
model.save_logs(filebase)
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["pde_loss"], label="pde_loss")
ax.plot(h["bc_loss"], label="bc_loss")
ax.legend()
ax.set_yscale("log")



# %%
x_validate = x_test
y_validate = s_testing_raw
u0_validate = u0_testing_raw

input_branch, input_trunk = x_validate[0], x_validate[1]
input_trunk.requires_grad_(True)
out=model((input_branch,input_trunk))
y_pred=out.detach().cpu().numpy()
error_s = []
for i in range(len(y_validate)):
    error_s_tmp = np.linalg.norm(y_validate[i] - y_pred[i]) / np.linalg.norm(y_validate[i])
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
def LaplaceOperator2D(y, x,aux=None):
    dydx2=DeepONet.laplacian(y, x)

    return -0.01 * (dydx2) 


# %%
laplace_op = DeepONet.EvaluateDeepONetPDEs(LaplaceOperator2D)
laplace_op_val = laplace_op((input_branch[min_median_max_index], input_trunk),model=model)
laplace_op_val=laplace_op_val.detach().cpu().numpy()
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

