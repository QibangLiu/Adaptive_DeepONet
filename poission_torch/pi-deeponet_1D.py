

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
filebase = (
    "./saved_model/pytorch_test_PI_1D"
)
os.makedirs(filebase, exist_ok=True)

# In[3]:
# Choose evaluation points
num_eval_points = 10
evaluation_points = np.linspace(0, 1, num_eval_points,dtype='float32').reshape(-1, 1)


degree = 3
space = dde.data.PowerSeries(N=degree + 1)
num_function = 100
features = space.random(num_function)
fx = space.eval_batch(features, evaluation_points)


boundary_mask = (evaluation_points[:,0] == evaluation_points.min()) | (evaluation_points[:,0] == evaluation_points.max()) 
interior_mask = ~boundary_mask
boundary_indices = np.where(boundary_mask)[0]
interior_indices = np.where(interior_mask)[0]



# %%
x_train = (fx, evaluation_points)
y_train = np.zeros((num_function, num_eval_points))
aux=fx

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_train = DeepONet.TripleCartesianProd(x_train, y_train,aux)
# %%
# dataLolder_train = DataLoader(dataset_train, batch_size=64)


train_loader = DataLoader(
    dataset_train,
    batch_size=16,
    shuffle=True,
    generator=torch.Generator(device=device),
    collate_fn=dataset_train.custom_collate_fn,
)

# %%
def equation(x, y,aux=None):
    dydx = torch.autograd.grad(
        outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]
    dydx2 = torch.autograd.grad(
        dydx,
        x,
        grad_outputs=torch.ones_like(dydx),
        create_graph=True,
    )[0]
    return  -dydx2-aux

class BCLoss(torch.nn.modules.loss._Loss):
    def __init__(self,indices,bc_v, size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__(size_average, reduce, reduction)
        self.indices = indices     
        self.bc_v=torch.tensor(bc_v).to(device)  
    def forward(self,y_pred=None):
        return torch.mean(torch.square(y_pred[self.indices]-self.bc_v))

bc_loss=BCLoss(boundary_indices,0)

class PI_DeepONet(DeepONet.DeepONetCartesianProd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    # def train_step(self, data, device="cpu"):
    #     self.optimizer.zero_grad()
    #     inputs,aux,_=data
    #     input_branch,input_trunk=inputs[0].to(device),inputs[1].to(device)
    #     aux=aux.to(device)
    #     input_trunk.requires_grad_(True)
    #     pde_losses,bc_losses=[],[]
    #     for inp_b,aux_ in zip(input_branch,aux):
    #         y=self((inp_b[None,:],input_trunk))
    #         pde_losses.append((equation(input_trunk,y[0][:,None],aux_)))  
    #         bc_losses.append(bc_loss(y[0][:,None]))
    #     pde_loss_ms=torch.mean(torch.square(torch.stack(pde_losses)))
    #     bc_loss_mean=torch.mean((torch.stack(bc_losses)))
    #     loss=pde_loss_ms+bc_loss_mean
    #     loss.backward()
    #     self.optimizer.step()
    #     return {"loss": loss.item(), "pde_loss": pde_loss_ms.item(), "bc_loss": bc_loss_mean.item()}
    def train_step(self, data, device="cpu"):
        loss_dic={}
        def closure():
            self.optimizer.zero_grad()
            inputs,aux,_=data
            input_branch,input_trunk=inputs[0].to(device),inputs[1].to(device)
            aux=aux.to(device)
            input_trunk.requires_grad_(True)
            pde_losses,bc_losses=[],[]
            for inp_b,aux_ in zip(input_branch,aux):
                y=self((inp_b[None,:],input_trunk))
                pde_losses.append((equation(input_trunk,y[0][:,None],aux_)))  
                bc_losses.append(bc_loss(y[0][:,None]))
            pde_loss_ms=torch.mean(torch.square(torch.stack(pde_losses)))
            bc_loss_mean=torch.mean((torch.stack(bc_losses)))
            loss=pde_loss_ms+bc_loss_mean
            loss.backward()
            loss_dic["loss"]=loss.item()
            loss_dic["pde_loss"]=pde_loss_ms.item()
            loss_dic["bc_loss"]=bc_loss_mean.item()
            return loss
        self.optimizer.step(closure)
        return loss_dic
    
    def compile(self, optimizer, lr=0.001):
        self.optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=lr,
            max_iter=1000,
            # max_eval=LBFGS_options["fun_per_step"],
            # tolerance_grad=LBFGS_options["gtol"],
            # tolerance_change=LBFGS_options["ftol"],
            # history_size=LBFGS_options["maxcor"],
            # line_search_fn=("strong_wolfe" if LBFGS_options["maxls"] > 0 else None),
        )

# %%
p=32
model = PI_DeepONet(
    [num_eval_points, 32, 32,32,32,p, ],
    [1, 32,32,32, p],
)



model.compile(optimizer=torch.optim.Adam, lr=0.0001)# torch.optim.Adam

model.to(device)
# keras.backend.set_value(model.optimizer.lr, 5e-4)
checkpoint_fname = os.path.join(filebase, "model.ckpt")
checkpoint_callback = DeepONet.ModelCheckpoint(
    checkpoint_fname, monitor="loss", save_best_only=True
)


# %%
torch.cuda.empty_cache()  # Clear cached memory
h = model.fit(
     train_loader, device=device, epochs=400
 )
model.save_logs(filebase)
# %%
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["pde_loss"], label="pde_loss")
ax.plot(h["bc_loss"], label="bc_loss")
ax.legend()
ax.set_yscale("log")
# %%
# %%
def LaplaceOperator2D(x, y,aux=None):
    dydx = torch.autograd.grad(
        outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]
    dydx2 = torch.autograd.grad(
        dydx,
        x,
        grad_outputs=torch.ones_like(dydx),
        create_graph=True,
    )[0]

    return  dydx2


# import dde
# def LaplaceOperator2D(x, y,aux=None):
#     dydx2 = dde.grad.hessian(y, x, i=0, j=0)
#     dydy2 = dde.grad.hessian(y, x, i=1, j=1)
#     return -0.01 *(dydx2 + dydy2)*scaler_solution
# %%
laplace_op = DeepONet.EvaluateDeepONetPDEs(LaplaceOperator2D)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x=evaluation_points
fx_t = torch.tensor(fx).to(device)
x_t = torch.tensor(x,requires_grad=True).to(device)

laplace_op = DeepONet.EvaluateDeepONetPDEs(LaplaceOperator2D)
laplace_op_val = laplace_op((fx_t,x_t),model=model)
dydxx_v=laplace_op_val.detach().cpu().numpy()
dydxx_v.shape
# %%
# %%
fx_ = -space.eval_batch(features, x)

# %%
fig = plt.figure(figsize=(7, 8))
i=0
ax=plt.subplot(2,2,i+1)
ax.plot(x,dydxx_v[i],'r',label='dydx2')
ax.plot(x,fx_[i],'--b',label='f')
ax.legend()
i=1
ax=plt.subplot(2,2,i+1)
ax.plot(x,dydxx_v[i],'r',label='dydx2')
ax.plot(x,fx_[i],'--b',label='f')
ax.legend()
i=2
ax=plt.subplot(2,2,i+1)
ax.plot(x,dydxx_v[i],'r',label='dydx2')
ax.plot(x,fx_[i],'--b',label='f')
ax.legend()
i=3
ax=plt.subplot(2,2,i+1)
ax.plot(x,dydxx_v[i],'r',label='dydx2')
ax.plot(x,fx_[i],'--b',label='f')
ax.legend()
# %%