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
filebase = "./saved_model/tf_test_PI_1D"
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
x_train=(fx,evaluation_points)
y_train = np.zeros((num_function, num_eval_points))
aux = x_train[0]
batch_size = x_train[0].shape[0]

batch_size=DeepONet.closest_divisor(x_train[0].shape[0],batch_size)

dataset_train = DeepONet.TripleCartesianProd(x_train, y_train, aux_data=aux,batch_size=batch_size)



def equation(y, x, f=None):
    dy_xx = DeepONet.laplacian(y, x)# shape (num_points,)
    return -dy_xx - f


class BCLoss(tf.keras.losses.Loss):
    def __init__(
        self, indices,bc_v,name="bcloss" ):
        super().__init__(name=name)
        self.indices =tf.constant(indices)
        self.bc_v = tf.constant(bc_v)

    def __call__(self, y_pred=None):
        return tf.reduce_mean(tf.square(tf.gather(y_pred,self.indices,axis=1) - self.bc_v))


bc_loss = BCLoss(boundary_indices, 0.0)


class PI_DeepONet(DeepONet.DeepONetCartesianProd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.pde_loss_tracker = tf.keras.metrics.Mean(name="pde_loss")
        self.bc_loss_tracker = tf.keras.metrics.Mean(name="bc_loss")
        
    # @property
    # def metrics(self):
    #     return [
    #         self.loss_tracker,
    #         self.pde_loss_tracker,
    #         self.bc_loss_tracker,
    #     ]

    def train_step(self, data):
        input, _ = data
        x_branch,x_trunk=input[0],input[1]
        aux=input[2]
        with tf.GradientTape() as tape:
            out=self((x_branch,x_trunk))
            pde_losses = []
            for i in range(batch_size):
                res=equation(out[i][:,None], x_trunk, aux[i])
                pde_losses.append(tf.reduce_mean(tf.square(res)))
            pde_loss_ms = tf.reduce_mean(tf.stack(pde_losses))
            bc_l = bc_loss(y_pred=out)
            loss=pde_loss_ms+bc_l
        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.pde_loss_tracker.update_state(pde_loss_ms)
        self.bc_loss_tracker.update_state(bc_l)
        return {"loss": self.loss_tracker.result(), "pde_loss": self.pde_loss_tracker.result(), "bc_loss": self.bc_loss_tracker.result()}
    


# %%
p = 32
model = PI_DeepONet(
    [num_eval_points, 32, 32, p],
    [1, 32, 32, p]
)

# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)
model.compile(optimizer=optimizer)  # torch.optim.Adam
tf.keras.backend.set_value(model.optimizer.lr, 1e-3)
checkpoint_fname = os.path.join(filebase, "model.ckpt")
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_fname,
    save_weights_only=True,
    monitor="pde_loss",
    verbose=0,
    save_freq="epoch",
    save_best_only=True,
    mode="min",
)


# %%
h = model.fit(dataset_train.dataset, epochs=2,verbose=2, callbacks=model_checkpoint)
model.save_history(filebase)
# %%
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["pde_loss"], label="pde_loss")
ax.plot(h["bc_loss"], label="bc_loss")
ax.legend()
ax.set_yscale("log")


# %%
laplace_op = DeepONet.EvaluateDeepONetPDEs(model,DeepONet.laplacian)

# %%
dydxx_v = laplace_op(x_train)
# %%
x = evaluation_points
fx_ = -fx

# %%
fig = plt.figure(figsize=(7, 8))
i = 0
id = 45
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
