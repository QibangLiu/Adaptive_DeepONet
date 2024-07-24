

# %%
import Adaptive_DeepONet.Adap_possion.DeepONet_torch as DeepONet_torch 
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.io as scio
import torch
from torch.utils.data import Dataset, DataLoader


# dde.config.set_default_float("float64")
# In[]
filebase = (
    "./saved_model/pytorch_test_PI"
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


# tf.keras.utils.set_random_seed(seed)
fenics_data = scio.loadmat(
    "./TrainingData/poisson_gauss_cov.mat"
)

x_grid = fenics_data["x_grid"].astype(np.float32)  # shape (Ny, Nx)
y_grid = fenics_data["y_grid"].astype(np.float32)
source_terms_raw = fenics_data["source_terms"].astype(np.float32)  # shape (N, Ny, Nx)
source_terms_raw = source_terms_raw.reshape(-1, Nx * Ny)
solutions_raw = fenics_data["solutions"].astype(np.float32)  # shape (N, Ny, Nx)
solutions_raw = solutions_raw.reshape(-1, Nx * Ny)
scaler_source = 0.5 * (np.max(source_terms_raw) - np.min(source_terms_raw))
scaler_solution = 0.5 * (np.max(solutions_raw) - np.min(solutions_raw))
solutions = solutions_raw / scaler_solution
source_terms = source_terms_raw / scaler_source
u0_train = source_terms[:5000]
u0_testing = source_terms[5000:6000]
s_train = solutions[:5000]
s_testing = solutions[5000:6000]

u0_testing_raw = source_terms_raw[5000:6000]
u0_train_raw = source_terms_raw[:5000]
s_testing_raw = solutions_raw[5000:6000]
s_train_raw = solutions_raw[:5000]

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
x_train = (u0_train, xy_train_testing)
y_train = s_train
x_test = (u0_testing, xy_train_testing)
y_test = s_testing


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_train = DeepONet_torch.TripleCartesianProd(x_train, y_train,u0_train_raw)
dataset_test = DeepONet_torch.TripleCartesianProd(x_test, y_test,u0_testing_raw)
# %%
# dataLolder_train = DataLoader(dataset_train, batch_size=64)


train_loader = DataLoader(
    dataset_train,
    batch_size=32,
    shuffle=True,
    #generator=torch.Generator(device=device),
    collate_fn=dataset_train.custom_collate_fn,
)

test_loader = DataLoader(
    dataset_test,
    batch_size=dataset_test.__len__(),
    #generator=torch.Generator(device=device),
    collate_fn=dataset_train.custom_collate_fn,
)
# %%

# dataloader = DataLoader(
#     dataset_train, batch_size=2000, collate_fn=custom_collate_fn)

# %%
mse = torch.nn.MSELoss()

class PDELoss(DeepONet_torch.Losses):
    '''indices: list of indices of the output that we want to apply the loss to
        generally is the points inside the doamin
    '''
    def __init__(self,indices,pde_evaluator, size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__(size_average, reduce, reduction)
        self.indices = indices
        self.pde_evaluator=pde_evaluator
    def forward(self,y_true=None, y_pred=None, inputs=None,aux=None, model=None):
        '''kwargs: dictionary of outputs of the model'''
        inputs[1].requires_grad_(True)
        input_=(inputs[0],inputs[1][self.indices,:])
        aux=aux[:,self.indices]
        losses= self.pde_evaluator(input_,model,aux)
        return torch.mean(torch.square(losses))
        
class BCLoss(DeepONet_torch.Losses):
    def __init__(self,indices,bc_v, size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__(size_average, reduce, reduction)
        self.indices = indices     
        self.bc_v=torch.tensor(bc_v).to(device)  
    def forward(self,y_true=None, y_pred=None, inputs=None,aux=None, model=None):
        input_=(inputs[0],inputs[1][self.indices,:])
        out=model(input_)
        return torch.mean(torch.square(out-self.bc_v))

bc_loss=BCLoss(boundary_indices,0)

def equation(x, y,aux=None):
    dy_dx = torch.autograd.grad(
        outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]
    dydx, dydy = dy_dx[:, 0:1], dy_dx[:, 1:2]
    dydx2 = torch.autograd.grad(
        dydx,
        x,
        grad_outputs=torch.ones_like(dydx),
        create_graph=True,
    )[0][:, 0:1]
    dydy2 = torch.autograd.grad(
        dydy, x, grad_outputs=torch.ones_like(dydy), create_graph=True
    )[0][:, 1:2]

    return  -0.01*scaler_solution*(dydx2 + dydy2)-aux.view(-1,1)

equa_evaluator=DeepONet_torch.EvaluateDeepONetPDEs(equation)

pde_loss=PDELoss(interior_indices,equa_evaluator)
# %%
model = DeepONet_torch.DeepONetCartesianProd(
    [m, 100, 100, 100, 100,100,100],
    [2, 100, 100, 100, 100,100,100],
)
model.compile(optimizer=torch.optim.Adam, lr=0.002, loss=[bc_loss,pde_loss],loss_names=['bc_loss','pde_loss'])

model.to(device)
# keras.backend.set_value(model.optimizer.lr, 5e-4)
checkpoint_fname = os.path.join(filebase, "model.ckpt")
checkpoint_callback = DeepONet_torch.ModelCheckpoint(
    checkpoint_fname, monitor="loss", save_best_only=True
)


# %%
torch.cuda.empty_cache()  # Clear cached memory
h = model.fit(
     train_loader, device=device,get_output=False, epochs=400, callbacks=checkpoint_callback
 )
model.save_logs(filebase)
model.summary()
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["pde_loss"], label="pde_loss")
ax.plot(h["bc_loss"], label="bc_loss")
ax.legend()
ax.set_yscale("log")
# %%
# %%
def LaplaceOperator2D(x, y,aux=None):
    dy_dx = torch.autograd.grad(
        outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]
    dydx, dydy = dy_dx[:, 0:1], dy_dx[:, 1:2]
    dydx2 = torch.autograd.grad(
        dydx,
        x,
        grad_outputs=torch.ones_like(dydx),
        create_graph=True,
    )[0][:, 0:1]
    dydy2 = torch.autograd.grad(
        dydy, x, grad_outputs=torch.ones_like(dydy), create_graph=True
    )[0][:, 1:2]

    return -0.01 * (dydx2 + dydy2) * scaler_solution

# import dde
# def LaplaceOperator2D(x, y,aux=None):
#     dydx2 = dde.grad.hessian(y, x, i=0, j=0)
#     dydy2 = dde.grad.hessian(y, x, i=1, j=1)
#     return -0.01 *(dydx2 + dydy2)*scaler_solution
# %%
laplace_op = DeepONet_torch.EvaluateDeepONetPDEs(LaplaceOperator2D)

# %%
x_plot=x_test
u0_plot_raw=u0_testing_raw

input_branch, input_trunk = x_plot[0][:2], x_plot[1]
input_branch = torch.tensor(input_branch).to(device)
input_trunk = torch.tensor(input_trunk,requires_grad=True).to(device)
output = model((input_branch, input_trunk))

dy_dx = torch.autograd.grad(
        outputs=output, inputs=input_trunk, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]
dy_dx.shape
# %%
laplace_op_val = laplace_op((input_branch[[[1,2,3]]], input_trunk),model=model)
laplace_op_val=laplace_op_val.detach().cpu().numpy()
# %%

y_pred_p = model.predict(x_test, device)
y_pred = scaler_solution * y_pred_p
y_test = s_testing_raw


# %%


# %%

error_s = []
for i in range(len(y_test)):
    error_s_tmp = np.linalg.norm(y_test[i] - y_pred[i]) / np.linalg.norm(y_test[i])
    error_s.append(error_s_tmp)
error_s = np.stack(error_s)
fig = plt.figure()
_ = plt.hist(error_s)


# %%

#### Plotting Results
# %%
import matplotlib.pyplot as plt
import pylab as py
import matplotlib.colors as mcolors


# Defining custom plotting functions
def my_contourf(x, y, F, ttl, vmin=None, vmax=None):
    cnt = py.contourf(
        x, y, F, 20, cmap="jet", norm=mcolors.Normalize(vmin=vmin, vmax=vmax)
    )
    # py.colorbar()
    py.xlabel("x")
    py.ylabel("y")
    py.title(ttl)
    return 0


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

    # if index == min_index:
    #     plt.savefig("temperature_min_error_5000_3.jpg", dpi=300)
    # if index == median_index:
    #     plt.savefig("temperature_median_error_5000_3.jpg", dpi=300)
    # if index == max_index:
    #     plt.savefig("temperature_max_error_5000_3.jpg", dpi=300)
    # plt.savefig("temperature_sample{}_5000_3.jpg".format(index), dpi=300)
    # plt.show()


# %%
class EvaluateDeepONetPDEs:
    """Generates the derivative of the outputs with respect to the trunck inputs.
    Args:
        model: DeepOnet.
        operator: Operator to apply to the outputs for derivative.
    """

    def __init__(self, model, operator):
        self.op = operator
        self.model = model

        def op(inputs,**kwargs):
            y = self.model(inputs)
            # QB: inputs[1] is the input of the trunck
            # QB: y[0] is the output corresponding
            # to the first input sample of the branch input,
            # each time we only consider one sample
            return self.op(inputs[1], y[0][:, None], **kwargs)

        self.tf_op = op

    def __call__(self, inputs,stack=True, **kwargs):
        self.value = []
        input_branch, input_trunck = inputs
        for inp in input_branch:
            x = (inp[None, :], input_trunck)
            self.value.append(self.tf_op(x), **kwargs)
        if stack:
            self.value = torch.stack(self.value)
        return self.value

    def get_values(self):
        return self.value


def LaplaceOperator2D(x, y):
    dydx2 = dde.grad.hessian(y, x, i=0, j=0)
    dydy2 = dde.grad.hessian(y, x, i=1, j=1)
    return -0.01 * (dydx2 + dydy2) * scaler_solution


# %%
def LaplaceOperator2D(x, y):
    dy_dx = torch.autograd.grad(
        outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]
    dydx, dydy = dy_dx[:, 0:1], dy_dx[:, 1:2]
    dydx2 = torch.autograd.grad(
        dydx,
        x,
        grad_outputs=torch.ones_like(dydx),
        create_graph=True,
    )[0][:, 0:1]
    dydy2 = torch.autograd.grad(
        dydy, x, grad_outputs=torch.ones_like(dydy), create_graph=True
    )[0][:, 1:2]

    return -0.01 * (dydx2 + dydy2) * scaler_solution


# %%
laplace_op = EvaluateDeepONetPDEs(model, LaplaceOperator2D)

# %%
input_branch, input_trunk = x_test[0], x_test[1]
input_branch = torch.tensor(input_branch).to(device)
x = torch.tensor(input_trunk[:, 0:1], requires_grad=True).to(device)
y = torch.tensor(input_trunk[:, 1:2], requires_grad=True).to(device)
input_trunk = torch.cat((x, y), 1)
# input_trunk = torch.tensor(input_trunk,requires_grad=True).to(device)
output = model((input_branch, input_trunk))

laplace_op_val = laplace_op.eval((input_branch[[min_median_max_index]], input_trunk))

# %%
nr, nc = 3, 2
i = 0
fig = plt.figure(figsize=(8, 10))

for i, index in enumerate(min_median_max_index):

    vmin = np.min(u0_testing_raw[index])
    vmax = np.max(u0_testing_raw[index])

    ax = plt.subplot(nr, nc, nc * i + 1)
    # py.figure(figsize = (14,7))
    c1 = ax.contourf(
        x_grid, y_grid, u0_testing_raw[index].reshape(Ny, Nx), 20, cmap="jet"
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
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


def custom_collate_fn(batch):
    # Assuming data is a list of tuples (sample, label)

    data, labels = zip(*batch)
    print("==", len(data), "==", len(labels))
    # Custom collation logic
    data = torch.stack(data, dim=0)
    labels = torch.tensor(labels)
    return data, labels


# Example data
data = torch.randn(100, 3, 224, 224)  # 100 samples of 3x224x224 images
labels = torch.randint(0, 2, (100,))  # Binary labels for the 100 samples

dataset = CustomDataset(data, labels)
dataloader = DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=custom_collate_fn
)

for batch_idx, (inputs, targets) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}")
    print(f"Inputs: {inputs.shape}")
    print(f"Targets: {targets.shape}")
