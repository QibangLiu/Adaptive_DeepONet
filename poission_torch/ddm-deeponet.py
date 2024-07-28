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
filebase = (
    "./saved_model/pytorch_test_ddm2D"
)
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


fenics_data = scio.loadmat(
    "../Adap_possion/TrainingData/poisson.mat"
)
gap=1
x_grid_full = fenics_data["x_grid"].astype(np.float32)  
y_grid_full = fenics_data["y_grid"].astype(np.float32)
x_idx=np.arange(0,x_grid_full.shape[1],gap)
#x_idx=np.append(x_idx,x_grid_full.shape[1]-1)
y_idx=x_idx[:,None]
x_idx=x_idx[None,:]

x_grid=x_grid_full#[y_idx,x_idx]# shape (Ny, Nx)
y_grid=y_grid_full#[y_idx,x_idx]# shape (Ny, Nx)
Ny,Nx=x_grid.shape#
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
Nx = 128
Ny = 128
m = Nx * Ny
###  N number of samples 1000
###  m number of points 40
###  P number of output sensors (coordinates) 1600
### x_train is a tuple of u0(N,m) and output sensores, all coordinates xy_train_testing(P,2)
### y_train is target solutions (our s) u(N,P)


# tf.keras.utils.set_random_seed(seed)
fenics_data = scio.loadmat(
    "../Adap_possion/TrainingData/poisson.mat"
)

x_grid_new = fenics_data["x_grid"].astype(np.float32)  # shape (Ny, Nx)
y_grid_new = fenics_data["y_grid"].astype(np.float32)
source_terms_raw_new = fenics_data["source_terms"].astype(np.float32)  # shape (N, Ny, Nx)
source_terms_raw_new = source_terms_raw_new.reshape(-1, Nx * Ny)
solutions_raw_new = fenics_data["solutions"].astype(np.float32)  # shape (N, Ny, Nx)
solutions_raw_new = solutions_raw_new.reshape(-1, Nx * Ny)
#shift_solution,scaler_solution = np.min(solutions_raw),np.max(solutions_raw)-np.min(solutions_raw)
#shift_source,scaler_source = np.min(source_terms_raw),np.max(source_terms_raw)-np.min(source_terms_raw)
shift_solution,scaler_solution=0,1
shift_source,scaler_source=0,1
solutions_new = (solutions_raw_new-shift_solution) / scaler_solution
source_terms_new = (source_terms_raw_new-shift_source) / scaler_source
num_train=5000
u0_train_new = source_terms_new[:num_train]
u0_testing_new = source_terms_new[num_train:6000]
s_train_new = solutions_new[:num_train]
s_testing_new = solutions_new[num_train:6000]

u0_testing_raw_new = source_terms_raw_new[num_train:6000]
u0_train_raw_new = source_terms_raw_new[:num_train]
s_testing_raw_new = solutions_raw_new[num_train:6000]
s_train_raw_new = solutions_raw_new[:num_train]

xy_train_testing_new = np.concatenate(
    [x_grid_new.reshape(-1, 1), y_grid_new.reshape(-1, 1)], axis=1
)


print("u0_train.shape = ", u0_train_new.shape)
print("type of u0_train = ", type(u0_train_new))
print("u0_testing.shape = ", u0_testing_new.shape)
print("s_train.shape = ", s_train_new.shape)
print("s_testing.shape = ", s_testing_new.shape)
print("xy_train_testing.shape", u0_train_new.shape)
# %%

class TripleCartesianProd(Dataset):
    """Dataset with each data point as a triple. The ordered pair of the first two
    elements are created from a Cartesian product of the first two lists. If we compute
    the Cartesian product of the first two arrays, then we have a ``Triple`` dataset.

    This dataset can be used with the network ``DeepONetCartesianProd`` for operator
    learning.
    """

    def __init__(self, X_train, y_train, aux=None, transform=None):
        # Initialize dataset parameters, load data, etc.
        # TODO: add transform
        self.transform = transform
        self.X_branch, self.X_trunk = X_train
        self.y = y_train
        self.aux = aux
        if (
            len(self.X_branch) != self.y.shape[0]
            or len(self.X_trunk) != self.y.shape[1]
        ):
            raise ValueError(
                "The training dataset does not have the format of Cartesian product."
            )
        if aux is not None and (
            len(aux) != self.y.shape[0] or aux.shape[1] != self.y.shape[1]
        ):
            raise ValueError("The auxiliary data does not have the correct shape.")

    def __len__(self):
        # Return the size of the dataset
        return len(self.X_branch)

    def __getitem__(self, idx):
        # Retrieve and preprocess a sample from the dataset
        # Example: Assuming your data is a tuple (input_data, output_data)
        x_branch = self.X_branch[idx]
        output_data = self.y[idx]
        x_trunk = self.X_trunk
        # TODO: add transform
        # if self.transform:
        #     input_data = self.transform(input_data)
        #     output_data = self.transform(output_data)
        # TODO: make x_trunk lenght be 1
        if self.aux is not None:
            aux = self.aux[idx]
            return (x_branch, x_trunk), output_data, aux

        else:
            return (x_branch, x_trunk), output_data, self.aux

    @staticmethod
    def custom_collate_fn(batch):
        # Assuming data is a list of tuples (sample, label)
        # batch=[((inp,out)),...], len=batch_size
        # inp=(x_branch, x_trunk)
        input, out, aux = zip(*batch)
        #out = torch.tensor(np.array(out))
        input_branch, input_trunk = zip(*input)
        #input_branch = torch.tensor(np.array(input_branch))
        input_trunk = input_trunk[0]
        if aux[0] is None:
            data = ((input_branch, input_trunk), out)

        else:
            data = ((input_branch, input_trunk), out, aux)

        return data


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train = (torch.tensor(u0_train).to(device), torch.tensor(xy_train_testing).to(device))
y_train = torch.tensor(s_train).to(device)
# x_test = (u0_testing, xy_train_testing)
# y_test = s_testing
dataset_train = TripleCartesianProd(x_train, y_train)
#dataset_test = DeepONet.TripleCartesianProd(x_test, y_test)
# %%
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
#     #generator=torch.Generator(device=device),
#     collate_fn=dataset_train.custom_collate_fn,
# )

# %%
mse = torch.nn.MSELoss()
model = DeepONet.DeepONetCartesianProd(
    [m, 100, 100, 100, 100, 100, 100],
    [2, 100, 100, 100, 100, 100, 100],
    activation={"branch": nn.ReLU(), "trunk": nn.Tanh()},
)
model.compile(optimizer=torch.optim.Adam, lr=1e-3, loss=mse)

model.to(device)
checkpoint_fname = os.path.join(filebase, "model.ckpt")
checkpoint_callback = DeepONet.ModelCheckpoint(
    checkpoint_fname, monitor="loss", save_best_only=True
)

# %%

h = model.fit(
    train_loader, device=device, epochs=20, callbacks=checkpoint_callback
)
# %%
model.save_logs(filebase)
model.load_weights(checkpoint_fname, device)
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["loss"], label="loss")
#ax.plot(h["val_loss"], label="val_loss")
ax.legend()
ax.set_yscale("log")

# %%
x_plot=x_train
y_plot = s_train_raw
u0_plot_raw=u0_train_raw

input_branch, input_trunk = x_plot[0], x_plot[1]
input_branch = torch.tensor(input_branch).to(device)
input_trunk = torch.tensor(input_trunk,requires_grad=True).to(device)

y_pred_out=model((input_branch,input_trunk))
y_pred=scaler_solution *y_pred_out.detach().cpu().numpy()+shift_solution
y_test = s_testing_raw
# %%

error_s = []
for i in range(len(y_test)):
    error_s_tmp = np.linalg.norm(y_test[i] - y_pred[i]) / np.linalg.norm(y_test[i])
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

    u0_testing_nx_ny = u0_testing_raw[index].reshape(Ny, Nx)
    s_testing_nx_ny = y_test[index].reshape(Ny, Nx)
    s_pred_nx_ny = y_pred[index].reshape(Ny, Nx)
    vmin = min(s_testing_nx_ny.min(), s_pred_nx_ny.min())
    vmax = max(s_testing_nx_ny.max(), s_pred_nx_ny.max())

    ax = plt.subplot(nr, nc, nc * i + 1)
    # py.figure(figsize = (14,7))
    c = ax.contourf(x_grid, y_grid, u0_testing_nx_ny, 20, cmap="jet")
    ax.set_title(r"Source Distrubution")
    plt.colorbar(c)
    plt.tight_layout()
    ax = plt.subplot(nr, nc, nc * i + 2)
    # py.figure(figsize = (14,7))
    c1 = ax.contourf(x_grid, y_grid, s_testing_nx_ny, 20, cmap="jet")
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
def LaplaceOperator2D(y, x,aux=None):
    dydx2=DeepONet.laplacian(y, x)
    return -0.01 * (dydx2) * scaler_solution

# import dde
# def LaplaceOperator2D(x, y,aux=None):
#     dydx2 = dde.grad.hessian(y, x, i=0, j=0)
#     dydy2 = dde.grad.hessian(y, x, i=1, j=1)
#     return -0.01 *(dydx2 + dydy2)*scaler_solution
# %%
laplace_op = DeepONet.EvaluateDeepONetPDEs(LaplaceOperator2D)
laplace_op_val = laplace_op((input_branch[min_median_max_index], input_trunk),model=model)
laplace_op_val=laplace_op_val.detach().cpu().numpy()

# %%
nr, nc = 3, 2
i = 0
fig = plt.figure(figsize=(8, 10))

for i, index in enumerate(min_median_max_index):

    vmin = np.min(u0_plot_raw[index])
    vmax = np.max(u0_plot_raw[index])

    ax = plt.subplot(nr, nc, nc * i + 1)
    # py.figure(figsize = (14,7))
    c1 = ax.contourf(
        x_grid, y_grid, u0_plot_raw[index].reshape(Ny, Nx), 20, cmap="jet"
    )
    ax.set_title(r"Source Distrubution")
    cbar = fig.colorbar(c1, ax=ax)
    plt.tight_layout()
    ax = plt.subplot(nr, nc, nc * i + 2)
    # py.figure(figsize = (14,7))
    c2 = ax.contourf(x_grid, y_grid, laplace_op_val[i].reshape(Ny, Nx), 20, cmap="jet")
    ax.set_title(r"$-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$")
    cbar = fig.colorbar(c2, ax=ax)
    plt.tight_layout()


# %%
