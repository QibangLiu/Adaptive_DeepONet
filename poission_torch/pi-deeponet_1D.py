# %%
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.io as scio
import torch
from torch.utils.data import Dataset, DataLoader
import DeepONet
import deepxde as dde

# dde.config.set_default_float("float64")
# In[]
filebase = "./saved_model/pytorch_test_PI_1D"
os.makedirs(filebase, exist_ok=True)

# In[3]:
# Choose evaluation points
num_eval_points = 10
evaluation_points = np.linspace(0, 1, num_eval_points, dtype="float32").reshape(-1, 1)


degree = 2
space = dde.data.PowerSeries(N=degree + 1)
num_function = 100
features = space.random(num_function)
fx = space.eval_batch(features, evaluation_points)


boundary_mask = (evaluation_points[:, 0] == evaluation_points.min()) | (
    evaluation_points[:, 0] == evaluation_points.max()
)
interior_mask = ~boundary_mask
boundary_indices = np.where(boundary_mask)[0]
interior_indices = np.where(interior_mask)[0]


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train = (torch.tensor(fx).to(device), torch.tensor(evaluation_points).to(device))
y_train = np.zeros((num_function, num_eval_points))
y_train = torch.tensor(y_train).to(device)
aux = x_train[0]

# %%


dataset_train = DeepONet.TripleCartesianProd(x_train, y_train, aux)

train_loader = DataLoader(
    dataset_train,
    batch_size=100,
    shuffle=True,
    generator=torch.Generator(device=device),
    collate_fn=dataset_train.custom_collate_fn,
)
# %%
for data in train_loader:
    print(type(data))
    inputs, targets, aux = data
    break
# %%


def equation(y, x, f=None):
    # dy_xx = dde.grad.hessian(y, x)
    dy_xx = DeepONet.laplacian(y, x)
    return -dy_xx - f


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


class PI_DeepONet(DeepONet.DeepONetCartesianProd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_losses(self, data):
        inputs, _, aux = data
        input_branch, input_trunk = inputs
        input_trunk.requires_grad_(True)
        out = self((input_branch, input_trunk))
        pde_losses = []
        for aux_, y in zip(aux, out):
            res = equation(y[:, None], input_trunk, aux_[:, None])
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
p = 32
model = PI_DeepONet(
    [num_eval_points, 32, 32, 32, p],
    [1, 32, 32, 32, p]
)

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.996)
model.compile(optimizer=optimizer)  # torch.optim.Adam

model.to(device)
# keras.backend.set_value(model.optimizer.lr, 5e-4)
checkpoint_fname = os.path.join(filebase, "model.ckpt")
checkpoint_callback = DeepONet.ModelCheckpoint(
    checkpoint_fname, monitor="loss", save_best_only=True
)


# %%
torch.cuda.empty_cache()  # Clear cached memory
h = model.fit(train_loader, epochs=1000, callbacks=checkpoint_callback)
model.save_logs(filebase)
model.load_weights(checkpoint_fname,device=device)
# %%
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["pde_loss"], label="pde_loss")
ax.plot(h["bc_loss"], label="bc_loss")
ax.legend()
ax.set_yscale("log")


# %%
laplace_op = DeepONet.EvaluateDeepONetPDEs(DeepONet.laplacian)

# %%
laplace_op_val = laplace_op(x_train, model=model)
dydxx_v = laplace_op_val.detach().cpu().numpy()
dydxx_v.shape
# %%
x = evaluation_points
fx_ = -fx

# %%
fig = plt.figure(figsize=(7, 8))
i = 0
id = 26
ax = plt.subplot(2, 2, i + 1)
ax.plot(x, dydxx_v[i + id], "r", label="dydx2")
ax.plot(x, fx_[i + id], "--b", label="f")
ax.legend()
i = 1
ax = plt.subplot(2, 2, i + 1)
ax.plot(x, dydxx_v[i + id], "r", label="dydx2")
ax.plot(x, fx_[i + id], "--b", label="f")
ax.legend()
i = 2
ax = plt.subplot(2, 2, i + 1)
ax.plot(x, dydxx_v[i + id], "r", label="dydx2")
ax.plot(x, fx_[i + id], "--b", label="f")
ax.legend()
i = 3
ax = plt.subplot(2, 2, i + 1)
ax.plot(x, dydxx_v[i + id], "r", label="dydx2")
ax.plot(x, fx_[i + id], "--b", label="f")
ax.legend()
# %%
