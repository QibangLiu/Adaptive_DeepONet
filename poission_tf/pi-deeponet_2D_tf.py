# In[1]:
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.io as scio
import DeepONet_tf as DeepONet
import deepxde as dde
import tensorflow as tf

# dde.config.set_default_float("float64")
# In[]
filebase = "./saved_model/tf_test_PI_2D"
os.makedirs(filebase, exist_ok=True)

# In[3]:
fenics_data = scio.loadmat("../Adap_possion/TrainingData/poisson_gauss_cov20k.mat")
gap = 4
x_grid_full = fenics_data["x_grid"].astype(np.float32)
y_grid_full = fenics_data["y_grid"].astype(np.float32)
x_idx = np.arange(0, x_grid_full.shape[1], gap)
x_idx = np.append(x_idx, x_grid_full.shape[1] - 1)
y_idx = x_idx[:, None]
x_idx = x_idx[None, :]

x_grid = x_grid_full[y_idx, x_idx]  # shape (Ny, Nx)
y_grid = y_grid_full[y_idx, x_idx]  # shape (Ny, Nx)
Ny, Nx = x_grid.shape
source_terms_raw_full = fenics_data["source_terms"].astype(
    np.float32
)  # shape (N, Ny, Nx)
source_terms_raw = source_terms_raw_full[:, y_idx, x_idx].reshape(-1, Nx * Ny)
solutions_raw_full = fenics_data["solutions"].astype(np.float32)  # shape (N, Ny, Nx)
solutions_raw = solutions_raw_full[:, y_idx, x_idx].reshape(-1, Nx * Ny)

shift_solution, scaler_solution = np.mean(solutions_raw_full), np.std(solutions_raw_full)
shift_source, scaler_source = np.mean(source_terms_raw_full), np.std(source_terms_raw_full)
# shift_solution, scaler_solution = np.min(solutions_raw_full), np.max(solutions_raw_full)-np.min(solutions_raw_full)
# shift_source, scaler_source = np.min(source_terms_raw_full), np.max(source_terms_raw_full)-np.min(source_terms_raw_full)
# shift_solution, scaler_solution = (
#     0.0,
#     (np.max(solutions_raw_full) - np.min(solutions_raw_full)) * 0.5,
# )
# shift_source, scaler_source = (
#     0.0,
#     (np.max(source_terms_raw_full) - np.min(source_terms_raw_full)) * 0.5,
# )
solutions = (solutions_raw - shift_solution) / scaler_solution
source_terms = (source_terms_raw - shift_source) / scaler_source

num_train = 5000
num_test = 1000
u0_train = source_terms[:num_train]
u0_testing = source_terms[-num_test:]
s_train = solutions[:num_train]
s_testing = solutions[-num_test:]

u0_testing_raw = source_terms_raw[-num_test:]
u0_train_raw = source_terms_raw[:num_train]
s_testing_raw = solutions_raw[-num_test:]
s_train_raw = solutions_raw[:num_train]

xy_train_testing = np.concatenate(
    [x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)], axis=1
)

boundary_mask = (
    (xy_train_testing[:, 0] == x_grid.min())
    | (xy_train_testing[:, 0] == x_grid.max())
    | (xy_train_testing[:, 1] == y_grid.min())
    | (xy_train_testing[:, 1] == y_grid.max())
)
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
aux_train = u0_train_raw
y_train = s_train

x_test = (u0_testing, xy_train_testing)
aux_test = u0_testing_raw
y_test = s_testing


batch_size = 50
batch_size = DeepONet.closest_divisor(x_train[0].shape[0], batch_size)

dataset_train = DeepONet.TripleCartesianProd(
    x_train, y_train, aux_data=aux_train, batch_size=batch_size
)

# %%


def equation(y, x, f=None):
    dy_xx = DeepONet.laplacian(y, x)  # shape (num_points,)
    return -dy_xx * 0.01 * scaler_solution - f


class BCLoss(tf.keras.losses.Loss):
    def __init__(self, indices, bc_v, name="bcloss"):
        super().__init__(name=name)
        self.indices = tf.constant(indices)
        self.bc_v = tf.constant(bc_v)

    def __call__(self, y_pred=None):
        return tf.reduce_mean(
            tf.square(tf.gather(y_pred, self.indices, axis=1) - self.bc_v)
        )


bc_loss = BCLoss(boundary_indices, 0.0)


class PI_DeepONet(DeepONet.DeepONetCartesianProd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.pde_loss_tracker = tf.keras.metrics.Mean(name="pde_loss")
        self.bc_loss_tracker = tf.keras.metrics.Mean(name="bc_loss")

    def train_step(self, data):
        input, _ = data
        x_branch, x_trunk = input[0], input[1]
        aux = input[2]
        with tf.GradientTape() as tape:
            out = self((x_branch, x_trunk))
            pde_losses = []
            for i in range(batch_size):
                res = equation(out[i][:, None], x_trunk, aux[i])
                pde_losses.append(tf.reduce_mean(tf.square(res)))
            pde_loss_ms = tf.reduce_mean(tf.stack(pde_losses))
            bc_l = bc_loss(y_pred=out)
            loss = pde_loss_ms + bc_l
        # with tf.GradientTape() as tape:
        #     pde_losses = []
        #     for i in range(batch_size):
        #         out = self((x_branch[i:i+1], x_trunk))
        #         res = equation(out[0][:, None], x_trunk, aux[i])
        #         pde_losses.append(tf.reduce_mean(tf.square(res)))
        #     pde_loss_ms = tf.reduce_mean(tf.stack(pde_losses))
        #     bc_l = bc_loss(y_pred=out)
            loss = pde_loss_ms + bc_l
        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.pde_loss_tracker.update_state(pde_loss_ms)
        self.bc_loss_tracker.update_state(bc_l)
        return {
            "loss": self.loss_tracker.result(),
            "pde_loss": self.pde_loss_tracker.result(),
            "bc_loss": self.bc_loss_tracker.result(),
        }


# %%
model = PI_DeepONet(
    [Ny * Nx, 100, 100, 100, 100, 100, 100,],
    [2, 100, 100, 100, 100, 100, 100,100],
    {"branch": tf.keras.activations.swish, "trunk": "tanh"},
)

# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer)  # torch.optim.Adam
# tf.keras.backend.set_value(model.optimizer.lr, 1e-3)
checkpoint_fname = os.path.join(filebase, "model.ckpt")
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_fname,
    save_weights_only=True,
    monitor="loss",
    verbose=0,
    save_freq="epoch",
    save_best_only=True,
    mode="min",
)


# %%
# model.load_weights(checkpoint_fname)
# h=model.load_history(filebase)
h = model.fit(dataset_train.dataset, epochs=10, verbose=2, callbacks=model_checkpoint)
model.save_history(filebase)
# %%
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["pde_loss"], label="pde_loss")
ax.plot(h["bc_loss"], label="bc_loss")
ax.legend()
ax.set_yscale("log")


# %%
u0_validate = u0_testing_raw
y_validate = s_testing_raw
x_validate = (u0_testing, xy_train_testing)
y_pred_out = model.predict(x_validate)
y_pred = y_pred_out * scaler_solution + shift_solution

# %%
error_s = []
for i in range(len(y_validate)):
    error_s_tmp = np.linalg.norm(y_validate[i] - y_pred[i]) / np.linalg.norm(
        y_validate[i]
    )
    error_s.append(error_s_tmp)
error_s = np.stack(error_s)
fig = plt.figure()
_ = plt.hist(error_s)

# %%
laplace_op = DeepONet.EvaluateDeepONetPDEs(model, DeepONet.laplacian)
def ErrorMeasure(X, rhs):
    lhs = -0.01 * (laplace_op(X)) * scaler_solution
    lhs = np.squeeze(lhs)
    return np.linalg.norm(lhs - rhs, axis=1) / np.linalg.norm(rhs, axis=1)
# %%
# %%time
# error_s = ErrorMeasure(x_validate, u0_validate)
# fig = plt.figure()
# _ = plt.hist(error_s)

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

    u0_validate_nx_ny = u0_validate[index].reshape(Ny, Nx)
    y_validate_nx_ny = y_validate[index].reshape(Ny, Nx)
    s_pred_nx_ny = y_pred[index].reshape(Ny, Nx)
    vmin = min(y_validate_nx_ny.min(), s_pred_nx_ny.min())
    vmax = max(y_validate_nx_ny.max(), s_pred_nx_ny.max())

    ax = plt.subplot(nr, nc, nc * i + 1)
    # py.figure(figsize = (14,7))
    c = ax.contourf(x_grid, y_grid, u0_validate_nx_ny, 20, cmap="jet")
    ax.set_title(r"Source Distrubution")
    plt.colorbar(c)
    plt.tight_layout()
    ax = plt.subplot(nr, nc, nc * i + 2)
    # py.figure(figsize = (14,7))
    c1 = ax.contourf(x_grid, y_grid, y_validate_nx_ny, 20, cmap="jet")
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
laplace_op_val_ = laplace_op((x_validate[0][min_median_max_index], x_validate[1]))
laplace_op_val = -0.01 * laplace_op_val_ * scaler_solution
# %%

nr, nc = 3, 2
i = 0
fig = plt.figure(figsize=(8, 10))


for i, index in enumerate(min_median_max_index):

    vmin = np.min(u0_validate[index])
    vmax = np.max(u0_validate[index])

    ax = plt.subplot(nr, nc, nc * i + 1)
    # py.figure(figsize = (14,7))
    c1 = ax.contourf(
        x_grid,
        y_grid,
        u0_validate[index].reshape(Ny, Nx),
        20,
        vmax=vmax,
        vmin=vmin,
        cmap="jet",
    )
    ax.set_title(r"Source Distrubution")
    cbar = fig.colorbar(c1, ax=ax)
    plt.tight_layout()
    ax = plt.subplot(nr, nc, nc * i + 2)
    # py.figure(figsize = (14,7))
    c2 = ax.contourf(
        x_grid,
        y_grid,
        laplace_op_val[i].reshape(Ny, Nx),
        20,
        vmax=vmax,
        vmin=vmin,
        cmap="jet",
    )
    ax.set_title(r"$-0.01*(\frac{d^2u}{dx^2}+\frac{d^2u}{dy^2})$")
    cbar = fig.colorbar(c2, ax=ax)
    plt.tight_layout()

# %%
