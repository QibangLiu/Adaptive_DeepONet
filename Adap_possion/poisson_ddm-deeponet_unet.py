#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.io as scio
from deepxde import utils
import deepxde as dde
from deepxde.backend import tf
from myutils import (
    find_checkpoint_2restore,
    EvaluateDerivatives,
    LaplaceOperator2D,
    UNET,
)
import myutils
# Set logging level to ERROR
# tf.get_logger().setLevel('FATAL')
# dde.backend.set_default_backend("tensorflow")
# dde.config.set_default_float("float64")
# In[]
filebase = "/scratch/bbpq/qibang/repository/Adap_data_driven_possion/saved_model/FenicsDataUnet"
os.makedirs(filebase, exist_ok=True)

restore_path = find_checkpoint_2restore(filebase)
# In[3]:

Nx = 128
Ny = 128
m = Nx * Ny
###  N number of samples 1000
###  m number of points 40
###  P number of output sensors (coordinates) 1600
### x_train is a tuple of u0(N,m) and output sensores, all coordinates xy_train_testing(P,2)
### y_train is target solutions (our s) u(N,P)

tf.keras.backend.clear_session()
# tf.keras.utils.set_random_seed(seed)
fenics_data = scio.loadmat(
    "/scratch/bbpq/qibang/repository/Adap_data_driven_possion/TrainingData/poisson.mat"
)

x_grid = fenics_data["x_grid"].astype(np.float32)  # shape (Ny, Nx)
y_grid = fenics_data["y_grid"].astype(np.float32)
source_terms = fenics_data["source_terms"].astype(np.float32)  # shape (N, Ny, Nx)
source_terms = source_terms.reshape(-1, Nx * Ny)
solutions = fenics_data["solutions"].astype(np.float32)  # shape (N, Ny, Nx)
solutions = solutions.reshape(-1, Nx * Ny)
u0_train = source_terms[:500]
u0_testing = source_terms[5000:]
s_train = solutions[:500]
s_testing = solutions[5000:]

xy_train_testing = np.concatenate(
    [x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)], axis=1
)


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
# x_train, y_train = get_data("train_IC2.npz")
# x_test, y_test = get_data("test_IC2.npz")

# %%
class DeepONetCartesianProd(dde.maps.NN):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super().__init__()
        if isinstance(activation, dict):
            self.activation_branch = activation["branch"]
            self.activation_trunk = dde.maps.activations.get(activation["trunk"])
        else:
            self.activation_branch = self.activation_trunk = dde.maps.activations.get(activation)
        self.trunk = layer_sizes_trunk[0] 
        self.branch=layer_sizes_branch[0]
        self.b = tf.Variable(tf.zeros(1))
    def call(self, inputs, training=False):
        x_func = inputs[0] # branch input
        x_loc = inputs[1] # trunk input
        x_func=self.branch(x_func)
        x_loc=self.trunk(x_loc)
        tf.print("x_func.shape", x_func.shape)
        tf.print("x_loc.shape", x_loc.shape)
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        y = self.merge_branch_trunk(x_func, x_loc)
        return y
    def merge_branch_trunk(self, x_func, x_loc):
        y = tf.einsum("bi,ni->bn", x_func, x_loc)
        y += self.b[index]
        return y
# %%


data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)
# %%

branch_model = UNET(
    img_size=(128, 128),
    output_size=100,
    img_channels=1,
    widths=[4],
    has_attention=[False, False, False, False],
    num_res_blocks=1,
    norm_groups=4,
)
trunk_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100,input_shape=(2, ),activation='tanh'),  # Flatten the input
    tf.keras.layers.Dense(100, activation='tanh'),  # First dense layer with ReLU activation
    tf.keras.layers.Dense(100, activation='tanh'),                   # Dropout layer to prevent overfitting
    tf.keras.layers.Dense(100, activation='tanh'),
    tf.keras.layers.Dense(100, activation='tanh'),
    tf.keras.layers.Dense(100)
])

net = dde.maps.DeepONetCartesianProd(
    (128*128,branch_model,100),
    [2, 100, 100, 100, 100, 100, 100],
    "tanh",
    "Glorot normal",
)


model = dde.Model(data, net)
model.compile(
    "adam",
    lr=1e-3,
    decay=("inverse time", 1, 1e-4),
    metrics=["mean l2 relative error"],
)
# %%

# IC1
# losshistory, train_state = model.train(epochs=100000, batch_size=None)
# IC2

check_point_filename = os.path.join(filebase, "model.ckpt")
checkpointer = dde.callbacks.ModelCheckpoint(
    check_point_filename, verbose=1, save_better_only=True
)
if restore_path is not None:
    model.restore(restore_path)
losshistory, train_state = model.train(
    iterations=20, batch_size=32, callbacks=[checkpointer]
)

y_pred = model.predict(data.test_x)
print("y_pred.shape =", y_pred.shape)
##np.savetxt("y_pred_deeponet.dat", y_pred[0].reshape(nt, nx))
##np.savetxt("y_true_deeponet.dat", data.test_y[0].reshape(nt, nx))
##np.savetxt("y_error_deeponet.dat", (y_pred[0] - data.test_y[0]).reshape(nt, nx))

# %%

error_s = []
for i in range(len(y_test)):
    error_s_tmp = np.linalg.norm(y_test[i] - y_pred[i]) / np.linalg.norm(y_test[i])
    error_s.append(error_s_tmp)
error_s = np.stack(error_s)
_ = plt.hist(error_s)
plt.xlabel("Relative error")
plt.ylabel("Frequency")

# %%

#### Plotting Results
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
    vmin, vmax = np.min(s_testing_nx_ny), np.max(s_testing_nx_ny)
    fig = plt.figure(figsize=(18, 5))
    ax = plt.subplot(1, 3, 1)
    # py.figure(figsize = (14,7))
    my_contourf(
        x_grid, y_grid, u0_testing_nx_ny, r"Source Distrubution", vmin=vmin, vmax=vmax
    )
    plt.tight_layout()
    ax = plt.subplot(1, 3, 2)
    # py.figure(figsize = (14,7))
    my_contourf(
        x_grid, y_grid, s_testing_nx_ny, r"Reference Solution", vmin=vmin, vmax=vmax
    )
    plt.tight_layout()
    ax = plt.subplot(1, 3, 3)
    # py.figure(figsize = (14,7))
    my_contourf(
        x_grid, y_grid, s_pred_nx_ny, r"Predicted Solution", vmin=vmin, vmax=vmax
    )
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
laplace_op = EvaluateDerivatives(model, LaplaceOperator2D)
# %%
laplace_op_val = laplace_op.eval((data.test_x[0][min_median_max_index], data.test_x[1]))
laplace_op_val = -0.01 * laplace_op_val

# %%
nr, nc = 3, 2
i = 0
fig = plt.figure(figsize=(8, 10))
ax = plt.subplot(nr, nc, 2 * i + 1)
# ax.contourf(x,y,data.train_x[0][min_median_max_index][i].reshape(Nx,Ny),12,cmap = 'jet')
vmin = np.min(data.test_x[0][min_median_max_index][i])
vmax = np.max(data.test_x[0][min_median_max_index][i])
my_contourf(
    x_grid,
    y_grid,
    data.test_x[0][min_median_max_index][i].reshape(Ny, Nx),
    r"Source Distrubution",
    vmin=vmin,
    vmax=vmax,
)
plt.tight_layout()
ax = plt.subplot(nr, nc, 2 * i + 2)
# ax.contourf(x,y,laplace_op_val[i].reshape(Nx,Ny),12,cmap = 'jet')
my_contourf(
    x_grid,
    y_grid,
    laplace_op_val[i].reshape(Ny, Nx),
    r"$-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$",
    vmin=vmin,
    vmax=vmax,
)
plt.tight_layout()
i = 1
vmin = np.min(data.test_x[0][min_median_max_index][i])
vmax = np.max(data.test_x[0][min_median_max_index][i])
ax = plt.subplot(nr, nc, 2 * i + 1)
# ax.contourf(x,y,data.train_x[0][min_median_max_index][i].reshape(Nx,Ny),12,cmap = 'jet')
my_contourf(
    x_grid,
    y_grid,
    data.test_x[0][min_median_max_index][i].reshape(Ny, Nx),
    r"Source Distrubution",
    vmin=vmin,
    vmax=vmax,
)
plt.tight_layout()
ax = plt.subplot(nr, nc, 2 * i + 2)
my_contourf(
    x_grid,
    y_grid,
    laplace_op_val[i].reshape(Ny, Nx),
    r"$-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$",
    vmin=vmin,
    vmax=vmax,
)
plt.tight_layout()
i = 2
vmin = np.min(data.test_x[0][min_median_max_index][i])
vmax = np.max(data.test_x[0][min_median_max_index][i])
ax = plt.subplot(nr, nc, 2 * i + 1)
# ax.contourf(x,y,data.train_x[0][min_median_max_index][i].reshape(Nx,Ny),12,cmap = 'jet')
my_contourf(
    x_grid,
    y_grid,
    data.test_x[0][min_median_max_index][i].reshape(Ny, Nx),
    r"Source Distrubution",
    vmin=vmin,
    vmax=vmax,
)
plt.tight_layout()
ax = plt.subplot(nr, nc, 2 * i + 2)
# ax.contourf(x,y,laplace_op_val[i].reshape(Nx,Ny),12,cmap = 'jet')
my_contourf(
    x_grid,
    y_grid,
    laplace_op_val[i].reshape(Ny, Nx),
    r"$-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$",
    vmin=vmin,
    vmax=vmax,
)
plt.tight_layout()
# %%
