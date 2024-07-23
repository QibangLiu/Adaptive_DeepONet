#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import os
from deepxde import utils
from deepxde.backend import tf
import scipy.io as scio
from myutils import find_checkpoint_2restore,EvaluateDerivatives,LaplaceOperator2D,UNET

# dde.config.disable_xla_jit()
# %%
filebase='/scratch/bbpq/qibang/repository/Adap_data_driven_possion/saved_model/PIFenicsData'
os.makedirs(filebase,exist_ok=True)
restore_path=find_checkpoint_2restore(filebase)

# In[ ]:
'''
Load external data
'''
Nx = 128
Ny = 128
m = Nx * Ny
fenics_data=scio.loadmat('/scratch/bbpq/qibang/repository/Adap_data_driven_possion/TrainingData/poisson.mat')

x_grid=fenics_data['x_grid'].astype(np.float32) # shape (Ny, Nx)
y_grid=fenics_data['y_grid'].astype(np.float32)
source_terms=fenics_data['source_terms'].astype(np.float32) # shape (N, Ny, Nx)
source_terms=source_terms.reshape(-1,Nx*Ny)
solutions=fenics_data['solutions'].astype(np.float32) # shape (N, Ny, Nx)
solutions=solutions.reshape(-1,Nx*Ny)
u0_train=source_terms[:5000]
u0_testing=source_terms[5000:]
s_train=solutions[:5000]
s_testing=solutions[5000:]

xy_coor=np.concatenate([x_grid.reshape(-1,1),y_grid.reshape(-1,1)],axis=1)

boundary_mask = (xy_coor[:,0] == x_grid.min()) | (xy_coor[:,0] == x_grid.max()) \
    | (xy_coor[:,1] == y_grid.min())  | (xy_coor[:,1] == y_grid.max())
interior_mask = ~boundary_mask
boundary_indices = np.where(boundary_mask)[0]
interior_indices = np.where(interior_mask)[0]
boundary_xy = xy_coor[boundary_indices]
interior_xy = xy_coor[interior_indices]

# In[ ]:

def equation(x, y, v):
    # Most backends
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    # Backend jax
    # dy_xx, _ = dde.grad.hessian(y, x, i=0, j=0, component=0)
    # dy_yy, _ = dde.grad.hessian(y, x, i=1, j=1, component=0)
    ##return -dy_xx - dy_yy - v
    return 0.01*(dy_xx + dy_yy) + v


def boundary(_, on_boundary):
    return on_boundary


##geom = dde.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])
geom = dde.geometry.Rectangle([0, 0], [1, 1])
bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

#### num_domain are # of points sampled inside domain, num_boundary # of points sampled on BC, 
#### num_test=None all points inside domain are used for testing PDE residual loss
pde = dde.data.PDE(geom, equation, bc, num_domain=len(interior_indices), num_boundary=len(boundary_indices))
#test=None, self.train_x, self.train_y, self.train_aux_vars,self.train_x_bc self.train_x_all

train_x_all_indices=np.concatenate([boundary_indices,interior_indices])
train_x_indices=np.concatenate([boundary_indices,train_x_all_indices])
pde.train_x_bc=xy_coor[boundary_indices]
pde.train_x_all=xy_coor[train_x_all_indices]
pde.train_x=xy_coor[train_x_indices]
pde.train_y=None
pde.train_aux_vars=None    
# Function space
##func_space = dde.data.GRF(kernel="ExpSineSquared", length_scale=1)
# %%
# will be replaced by external data
func_space = dde.data.GRF2D(length_scale=0.1,interp='linear', N=100)

### random source funcion is evaluated on 64 x 64 grid, i.e m = 50x50=2500
x = np.linspace(0, 1, num=128)
y = np.linspace(0, 1, num=128)
xv, yv = np.meshgrid(x, y)
eval_pts = np.vstack((np.ravel(xv), np.ravel(yv))).T
# %%
### 5000 random distributions, or 5000 traning samples, 100 testing samples
data = dde.data.PDEOperatorCartesianProd(
    pde, func_space, eval_pts, len(u0_train), function_variables=None, num_test=len(u0_testing), batch_size=128
)

# In[ ]:
'''
Replace the data with external data
To be replaced: self.train_x, self.train_y, self.train_aux_vars, eval_pts,num_test=100
Following the source code of dde.data.PDEOperatorCartesianProd:

    let's say f(x) be the input of branch
    train_x=(f(evaluation_points),pde.train_x))
    num_eval_points is the shape of branch's input layer
    evaluation_points and pde.train_x are different
    train_aux_vars=f(pde.train_x)
    text_x=train_x or test data
    train_y=None
'''

data.train_x=(u0_train, pde.train_x)
data.train_aux_vars=u0_train[:,train_x_indices]
data.test_x=(u0_testing, pde.train_x)
data.test_aux_vars=u0_testing[:,train_x_indices]
data.train_y=None
data.test_y=None
# In[ ]:

branch_model = UNET(
    img_size=(128, 128),
    output_size=100,
    img_channels=1,
    widths=[8, 16, 32, 64],
    has_attention=[False, False, True, True],
    num_res_blocks=2,
    norm_groups=8,
)

# Net branch first, trunk second 
net = dde.nn.DeepONetCartesianProd(
    [None,branch_model,100],
    [2, 100, 100, 100, 100, 100, 100],
    "tanh",
    "Glorot normal",
)
print("num_trainable_parameters", net.num_trainable_parameters())
# %%
"""
def periodic(x):
    x, t = x[:, :1], x[:, 1:]
    x = x * 2 * np.pi
    return concat([cos(x), sin(x), cos(2 * x), sin(2 * x), t], 1)


net.apply_feature_transform(periodic)
"""

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
check_point_filename=os.path.join(filebase, 'model.ckpt')
checkpointer = dde.callbacks.ModelCheckpoint(check_point_filename, verbose=1, save_better_only=True)

if restore_path is not None:
    model.restore(restore_path)
    
losshistory, train_state = model.train(iterations=60000,batch_size=128, callbacks=[checkpointer])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
loss_data=losshistory.__dict__
fig=plt.figure()
ax=plt.subplot(1,1,1)
ax.plot(loss_data['steps'], loss_data['loss_train'],label='loss_train')
ax.plot(loss_data['steps'], loss_data['loss_test'],label='loss_test')
ax.plot(loss_data['steps'], loss_data['metrics_test'],label='metrics_test')
ax.legend()
ax.set_yscale('log')
# In[ ]:


import time as TT
st = TT.time()
y_pred = model.predict((u0_testing,xy_coor))
duration = TT.time() - st
print('y_pred.shape =', y_pred.shape)

print('Inference took ' , duration , ' s' )
print('Prediction speed = ' , duration / float(len(y_pred)) , ' s/case' )

error_s = []
y_test = s_testing
for i in range(len(y_test)):
    error_s_tmp = np.linalg.norm(y_test[i] - y_pred[i]) / np.linalg.norm(y_test[i])
    error_s.append(error_s_tmp)
error_s = np.stack(error_s)
_=plt.hist(error_s)
plt.xlabel("Relative error")
plt.ylabel("Frequency")
# %%
import matplotlib.pyplot as plt
import pylab as py


# Defining custom plotting functions
def my_contourf(x, y, F, ttl, vmin=None, vmax=None):
    cnt = py.contourf(x, y, F, 12, cmap="jet", vmin=vmin, vmax=vmax)
    py.colorbar()
    py.xlabel("x")
    py.ylabel("y")
    py.title(ttl)
    return 0

min_index = np.argmin(error_s)
max_index = np.argmax(error_s)
median_index = np.median(error_s).astype(int)

# Print the indexes
print("Index for minimum element:", min_index)
print("Index for maximum element:", max_index)
print("Index for median element:", median_index)


min_median_max_index = np.array([min_index, median_index, max_index])


for index in min_median_max_index:

    u0_testing_nx_ny = u0_testing[index].reshape(Ny, Nx)
    s_testing_nx_ny = y_test[index].reshape(Ny, Nx)
    s_pred_nx_ny = y_pred[index].reshape(Ny, Nx)
    vmin,vmax=np.min(s_testing_nx_ny),np.max(s_testing_nx_ny)
    fig = plt.figure(figsize=(18, 5))
    ax=plt.subplot(1, 3, 1)
    # py.figure(figsize = (14,7))
    my_contourf(x_grid, y_grid, u0_testing_nx_ny, r"Source Distrubution",vmin=vmin, vmax=vmax)
    plt.tight_layout()
    ax=plt.subplot(1, 3, 2)
    # py.figure(figsize = (14,7))
    my_contourf(x_grid, y_grid, s_testing_nx_ny, r"Reference Solution",vmin=vmin, vmax=vmax)
    plt.tight_layout()
    ax=plt.subplot(1, 3, 3)
    # py.figure(figsize = (14,7))
    my_contourf(x_grid, y_grid, s_pred_nx_ny, r"Predicted Solution",vmin=vmin, vmax=vmax)
    plt.tight_layout()
# In[ ]:

# %%
laplace_op = EvaluateDerivatives(model, LaplaceOperator2D)
# %%
laplace_op_val = laplace_op.eval((u0_testing,xy_coor))
laplace_op_val = -0.01 * laplace_op_val

# %%
# %%
nr, nc = 3, 2
i = 0
fig = plt.figure(figsize=(8, 10))
ax = plt.subplot(nr, nc, 2 * i + 1)
# ax.contourf(x,y,data.train_x[0][min_median_max_index][i].reshape(Nx,Ny),12,cmap = 'jet')
vmin = np.min(u0_testing[min_median_max_index][i])
vmax=np.max( u0_testing[min_median_max_index][i])
my_contourf(
    x_grid,
    y_grid,
    u0_testing[min_median_max_index][i].reshape(Ny, Nx),
    r"Source Distrubution",vmin=vmin,vmax=vmax,
)
plt.tight_layout()
ax = plt.subplot(nr, nc, 2 * i + 2)
# ax.contourf(x,y,laplace_op_val[i].reshape(Nx,Ny),12,cmap = 'jet')
my_contourf(
    x_grid,
    y_grid,
    laplace_op_val[i].reshape(Ny, Nx),
    r"$-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$",vmin=vmin,vmax=vmax,
)
plt.tight_layout()
i = 1
vmin = np.min(u0_testing[min_median_max_index][i])
vmax=np.max( u0_testing[min_median_max_index][i])
ax = plt.subplot(nr, nc, 2 * i + 1)
# ax.contourf(x,y,data.train_x[0][min_median_max_index][i].reshape(Nx,Ny),12,cmap = 'jet')
my_contourf(
    x_grid,
    y_grid,
    u0_testing[min_median_max_index][i].reshape(Ny, Nx),
    r"Source Distrubution",vmin=vmin,vmax=vmax
)
plt.tight_layout()
ax = plt.subplot(nr, nc, 2 * i + 2)
my_contourf(
    x_grid,
    y_grid,
    laplace_op_val[i].reshape(Ny, Nx),
    r"$-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$",vmin=vmin,vmax=vmax,
)
plt.tight_layout()
i = 2
vmin = np.min(u0_testing[min_median_max_index][i])
vmax=np.max( u0_testing[min_median_max_index][i])
ax = plt.subplot(nr, nc, 2 * i + 1)
# ax.contourf(x,y,data.train_x[0][min_median_max_index][i].reshape(Nx,Ny),12,cmap = 'jet')
my_contourf(
    x_grid,
    y_grid,
    u0_testing[min_median_max_index][i].reshape(Ny, Nx),
    r"Source Distrubution",vmin=vmin,vmax=vmax,
)
plt.tight_layout()
ax = plt.subplot(nr, nc, 2 * i + 2)
# ax.contourf(x,y,laplace_op_val[i].reshape(Nx,Ny),12,cmap = 'jet')
my_contourf(
    x_grid,
    y_grid,
    laplace_op_val[i].reshape(Ny, Nx),
    r"$-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$",vmin=vmin,vmax=vmax,
)
plt.tight_layout()