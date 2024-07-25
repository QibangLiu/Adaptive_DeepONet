# %%
"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from deepxde import utils
# %%
def myfun(x,y):
    return dde.grad.hessian(y, x)
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
pde = dde.data.PDE(geom, equation, bc, num_domain=100, num_boundary=2)

# Function space for f(x) are polynomials
degree = 3
space = dde.data.PowerSeries(N=degree + 1)

# Choose evaluation points
num_eval_points = 10
evaluation_points = geom.uniform_points(num_eval_points, boundary=True)

# Define PDE operator
pde_op = dde.data.PDEOperatorCartesianProd(
    pde,
    space,
    evaluation_points,
    num_function=100,
)

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
# dde.optimizers.set_LBFGS_options(maxiter=1000)
# model.compile("L-BFGS")
# model.train()
model.compile("adam", lr=0.001)
model.train(iterations=1000)
# %%
# Plot realisations of f(x)
n = 4
features = space.random(n)
fx = space.eval_batch(features, evaluation_points)

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
import torch
def second_derivative(x, y,aux=None):
    return dde.grad.hessian(y, x)


def second_derivative(x, y,aux=None):
    dy_dx = torch.autograd.grad(
        outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]
    dydx2 = torch.autograd.grad(
        dy_dx,
        x,
        grad_outputs=torch.ones_like(dy_dx),
        create_graph=True,
    )[0]

    return  dydx2

# %%
class EvaluateDeepONetPDEs:
    """Generates the derivative of the outputs with respect to the trunck inputs.
    Args:
        operator: Operator to apply to the outputs for derivative.
    """

    def __init__(self, operator):
        self.op = operator

        def op(inputs,model=None,aux=None):
            ''' model: The modDeepOnetel to evaluate the derivative.'''
            y = model(inputs)
            # QB: inputs[1] is the input of the trunck
            # QB: y[0] is the output corresponding
            # to the first input sample of the branch input,
            # each time we only consider one sample
            return self.op(inputs[1], y[0][:, None], aux)

        self.operator = op

    def __call__(self, inputs,model=None,aux=None,stack=True):
        self.value = []
        input_branch, input_trunck = inputs
        if aux is not None:
            for inp,aux_ in zip(input_branch,aux):
                x = (inp[None, :], input_trunck)
                self.value.append(self.operator(x,model,aux_[None, :]))
        else:
            for inp in input_branch:
                x = (inp[None, :], input_trunck)
                self.value.append(self.operator(x,model,aux))
        if stack:
            self.value = torch.stack(self.value)
        return self.value

    def get_values(self):
        return self.value
# %%
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fx_t = torch.tensor(fx).to(device)
x_t = torch.tensor(x,requires_grad=True).to(device)

laplace_op=EvaluateDeepONetPDEs(second_derivative)
laplace_op_val = laplace_op((fx_t,x_t),model=model.net)
dydxx_v=laplace_op_val.detach().cpu().numpy()
dydxx_v.shape
# %%
out=model.net((fx_t,x_t))
dydxx_v=[]
for i in range(4):
    dydxx_v.append(second_derivative(x_t,out[i:i+1]))
dydxx_v = torch.stack(dydxx_v).detach().cpu().numpy()
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
