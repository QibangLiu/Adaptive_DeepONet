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
from myutils import find_checkpoint_2restore,EvaluateDerivatives,LaplaceOperator2D
# dde.config.disable_xla_jit()
#dde.config.set_default_float("float64")

filebase='/scratch/bbpq/qibang/repository/Adap_data_driven_possion/saved_model/PI_dde'




restore_path=find_checkpoint_2restore(filebase)
# %%
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
#pde = dde.data.PDE(geom, equation, bc, num_domain=10000, num_boundary=396, num_test=None)
pde = dde.data.PDE(geom, equation, bc, num_domain=15876, num_boundary=508, num_test=None)

# %%
# Function space
##func_space = dde.data.GRF(kernel="ExpSineSquared", length_scale=1)
func_space = dde.data.GRF2D(length_scale=0.1,interp='linear', N=100)

### random source funcion is evaluated on 64 x 64 grid, i.e m = 50x50=2500
num_grid=128
x = np.linspace(0, 1, num=num_grid)
y = np.linspace(0, 1, num=num_grid)
xv, yv = np.meshgrid(x, y)
eval_pts = np.vstack((np.ravel(xv), np.ravel(yv))).T

### 5000 random distributions, or 5000 traning samples, 100 testing samples
#data = dde.data.PDEOperatorCartesianProd(
#     pde, func_space, eval_pts, 5000, function_variables=None, num_test=100, batch_size=32
# )
data = dde.data.PDEOperatorCartesianProd(
    pde, func_space, eval_pts, 5000, function_variables=None, num_test=1000, batch_size=32
)
# In[ ]:

# Net branch first, trunk second 
net = dde.nn.DeepONetCartesianProd(
    [num_grid*num_grid, 100, 100, 100, 100, 100, 100],
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
losshistory, train_state = model.train(iterations=20,batch_size=32, callbacks=[checkpointer])
#dde.saveplot(losshistory, train_state, issave=True, isplot=True)


# In[ ]:


import time as TT
st = TT.time()
y_pred = model.predict(data.test_x)
duration = TT.time() - st
print('y_pred.shape =', y_pred.shape)

print('Inference took ' , duration , ' s' )
print('Prediction speed = ' , duration / float(len(y_pred)) , ' s/case' )



##xy_test = geom.uniform_points(10000, boundary=True)
# %%
x_s = np.linspace(0, 1, 64)
y_s = np.linspace(0, 1, 64)
XX_s, YY_s = np.meshgrid(x_s, y_s)
xy_test = np.vstack((np.ravel(XX_s), np.ravel(YY_s))).T

n = 3
features = func_space.random(n)
fx_test = func_space.eval_batch(features, xy_test)
#fx_test=data.train_x[0][0:3]
y_test = model.predict((fx_test, xy_test))





# In[ ]:


# %%
laplace_op=EvaluateDerivatives(model,LaplaceOperator2D)

laplace_op.eval((fx_test, xy_test))
laplace_op_val=laplace_op.get_values()

# %%
import pylab as py
def my_contourf(x,y,F,ttl):
    cnt = py.contourf(x,y,F,12,cmap = 'jet')
    py.colorbar()
    py.xlabel('x'); py.ylabel('y'); py.title(ttl)
    return 0
# %%
Nx,Ny=64,64
nr,nc=3,2
i=0
fig = plt.figure(figsize=(8,10))
ax=plt.subplot(nr,nc,2*i+1)
#ax.contourf(x,y,data.train_x[0][min_median_max_index][i].reshape(Nx,Ny),12,cmap = 'jet')
my_contourf(XX_s, YY_s,fx_test[i].reshape(Nx,Ny).T,r'Source Distrubution')
plt.tight_layout()
ax=plt.subplot(nr,nc,2*i+2)
#ax.contourf(XX_s, YY_s,laplace_op_val[i].reshape(Nx,Ny),12,cmap = 'jet')
my_contourf(XX_s, YY_s,-0.01*laplace_op_val[i].reshape(Nx,Ny).T,r'$\kappa(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$')
plt.tight_layout()
i=1
ax=plt.subplot(nr,nc,2*i+1)
#ax.contourf(XX_s, YY_s,data.train_x[0][min_median_max_index][i].reshape(Nx,Ny),12,cmap = 'jet')
my_contourf(XX_s, YY_s,fx_test[i].reshape(Nx,Ny).T,r'Source Distrubution')
plt.tight_layout()
ax=plt.subplot(nr,nc,2*i+2)
my_contourf(XX_s, YY_s,-0.01*laplace_op_val[i].reshape(Nx,Ny).T,r'$\kappa(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$')
plt.tight_layout()
i=2
ax=plt.subplot(nr,nc,2*i+1)
#ax.contourf(XX_s, YY_s,data.train_x[0][min_median_max_index][i].reshape(Nx,Ny),12,cmap = 'jet')
my_contourf(XX_s, YY_s,fx_test[i].reshape(Nx,Ny).T,r'Source Distrubution')
plt.tight_layout()
ax=plt.subplot(nr,nc,2*i+2)
#ax.contourf(XX_s, YY_s,laplace_op_val[i].reshape(Nx,Ny),12,cmap = 'jet')
my_contourf(XX_s, YY_s,-0.01*laplace_op_val[i].reshape(Nx,Ny).T,r'$\kappa(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$')
plt.tight_layout()
plt.savefig(os.path.join(filebase,'source_laplace.png'))
# %%
