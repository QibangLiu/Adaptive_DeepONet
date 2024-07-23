# %%
"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from deepxde.backend import tf
from deepxde import utils
# %%
# Poisson equation: -u_xx = f
def equation(x, y, f):
    dy_xx = dde.grad.hessian(y, x)
    return -dy_xx - f


# Domain is interval [0, 1]
geom = dde.geometry.Interval(0, 1)


# Zero Dirichlet BC
def u_boundary(_):
    return 0


def boundary(_, on_boundary):
    return on_boundary


bc = dde.icbc.DirichletBC(geom, u_boundary, boundary)

# Define PDE
pde = dde.data.PDE(geom, equation, bc, num_domain=99, num_boundary=2)
# change the training data, following the src in the deepxde
pde.train_x_bc=np.array([0.0,1.0],dtype='float32')[:,None]
train_x_domain=np.linspace(0.01,0.99,99,dtype='float32')[:,None]
pde.train_x_all=np.concatenate((pde.train_x_bc,train_x_domain))
pde.train_x=np.concatenate((pde.train_x_bc,pde.train_x_all))
pde.train_y=None
pde.train_aux_vars=None

# In[]
# Function space for f(x) are polynomials
degree = 3
space = dde.data.PowerSeries(N=degree + 1)
# Choose evaluation points
num_eval_points = 10
evaluation_points = geom.uniform_points(num_eval_points, boundary=True)

space_p1 = dde.data.PowerSeries(N=degree + 2)
num_function = 100
features = space_p1.random(num_function)
fx = space_p1.eval_batch(features, evaluation_points)


# Define PDE operator
pde_op = dde.data.PDEOperatorCartesianProd(
    pde,
    space,
    evaluation_points,
    num_function=num_function,
)
# train_y=None
# let's say f(x) be in input of branch
# train_x=(f(evaluation_points),pde.train_x))
# num_eval_points is the shape of branch's input layer
# evaluation_points and pde.train_x are different
# train_aux_vars=f(pde.train_x)
# text_x=train_x
pde_op.train_x=(fx,pde.train_x)
pde_op.train_aux_vars=space_p1.eval_batch(features, pde.train_x)
pde_op.test_x=pde_op.train_x
pde_op.test_aux_vars=pde_op.train_aux_vars

# In[]
# Setup DeepONet
dim_x = 1
p = 32
net = dde.nn.DeepONetCartesianProd(
    [num_eval_points, 32, p],
    [dim_x, 32, p],
    activation="tanh",
    kernel_initializer="Glorot normal",
)
# %%
# Define and train model
model = dde.Model(pde_op, net)
dde.optimizers.set_LBFGS_options(maxiter=1000)
model.compile("L-BFGS")
model.train()
# %%
# Plot realisations of f(x)
n = 4
features = space_p1.random(n)
fx = space_p1.eval_batch(features, evaluation_points)

x = geom.uniform_points(100, boundary=True)
y = model.predict((fx, x))

# Setup figure
fig = plt.figure(figsize=(7, 8))
plt.subplot(2, 1, 1)
plt.title("Poisson equation: Source term f(x) and solution u(x)")
plt.ylabel("f(x)")
z = np.zeros_like(x)
plt.plot(x, z, "k-", alpha=0.1)

# Plot source term f(x)
for i in range(n):
    plt.plot(evaluation_points, fx[i], "--")

# Plot solution u(x)
plt.subplot(2, 1, 2)
plt.ylabel("u(x)")
plt.plot(x, z, "k-", alpha=0.1)
for i in range(n):
    plt.plot(x, y[i], "-")
plt.xlabel("x")

plt.show()
# %%
def second_derivative(x, y):
    return dde.grad.hessian(y, x)

class EvaluateDerivatives():
    """Generates the derivative of the outputs with respect to the trunck inputs.

    Args:
        model: DeepOnet.
        operator: Operator to apply to the outputs for derivative.
    """

    def __init__(self,model,operator):
        self.op=operator
        self.model=model
        @tf.function
        def op(inputs):
            y = self.model.net(inputs)
             # QB: inputs[1] is the input of the trunck
             # QB: y[0] is the output corresponding
             # to the first input sample of the branch input,
             # each time we only consider one sample
            return self.op(inputs[1], y[0][:,None]) 
        self.tf_op = op
    def eval(self,inputs):
        self.value=[]
        input_branch,input_trunck=inputs
        for inp in input_branch:
            x=(inp[None,:],input_trunck)
            self.value.append(utils.to_numpy(self.tf_op(x)))
        self.value=np.array(self.value)
    def get_values(self):
        return self.value

# %%
dydxx=EvaluateDerivatives(model,second_derivative)
# %%
# dydxx_v=[]
# for f in fx:
#     dydxx.eval((f[None,:],x))
#     dydxx_v.append(dydxx.get_values())
dydxx_v=dydxx.eval((fx,x))
dydxx_v=dydxx.get_values()
# %%
fx_ = -space_p1.eval_batch(features, x)

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
