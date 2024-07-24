

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
dataset_train = DeepONet_torch.TripleCartesianProd(x_train, y_train,aux)
# %%
# dataLolder_train = DataLoader(dataset_train, batch_size=64)


train_loader = DataLoader(
    dataset_train,
    batch_size=32,
    shuffle=True,
    generator=torch.Generator(device=device),
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

equa_evaluator=DeepONet_torch.EvaluateDeepONetPDEs(equation)

pde_loss=PDELoss(interior_indices,equa_evaluator)
# %%
p=32
model = DeepONet_torch.DeepONetCartesianProd(
    [num_eval_points, 32, p, ],
    [1, 32, p],
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
laplace_op = DeepONet_torch.EvaluateDeepONetPDEs(LaplaceOperator2D)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x=evaluation_points
fx_t = torch.tensor(fx).to(device)
x_t = torch.tensor(x,requires_grad=True).to(device)

laplace_op = DeepONet_torch.EvaluateDeepONetPDEs(LaplaceOperator2D)
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