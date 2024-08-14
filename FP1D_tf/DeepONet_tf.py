# %%
import tensorflow as tf
from tensorflow import keras
import os
import json
import timeit
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# %%
class DeepONetCartesianProd(keras.Model):
    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation="tanh",
        num_outputs=1,
        apply_activation_outlayer=True,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        if isinstance(activation, dict):
            self.activation_branch = activation["branch"]
            self.activation_trunk = activation["trunk"]
        else:
            self.activation_branch = self.activation_trunk = activation
        self.trunk = self.build_net(
            layer_sizes_trunk, self.activation_trunk,False, apply_activation_outlayer
        )
        self.branch = self.build_net(
            layer_sizes_branch, self.activation_branch,False, apply_activation_outlayer
        )
        self.b = tf.Variable(tf.zeros(1))
        self.b = [
            tf.Variable(tf.zeros(1))
            for _ in range(self.num_outputs)
        ]
        if self.trunk.output_shape != self.branch.output_shape:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        if self.trunk.output_shape[-1] % self.num_outputs != 0:
            raise AssertionError(
                f"Output size of the branch net is not evenly divisible by {self.num_outputs}."
            )
        self.shift_size=int(self.trunk.output_shape[-1]/self.num_outputs)
        self.fit_history = None

    def build_net(self, layer_sizes, activation,mask_layer=False, apply_activation_outlayer=True):
        # User-defined network
        if callable(layer_sizes[0]):
            return layer_sizes[0]
        # Fully connected network
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(layer_sizes[0],)))
        if mask_layer:
            model.add(tf.keras.layers.Masking(mask_value=self.padding_value))
        for size in layer_sizes[1:]:
            if size == layer_sizes[-1] and not apply_activation_outlayer:
                model.add(tf.keras.layers.Dense(size, activation="linear"))
            else:
                model.add(tf.keras.layers.Dense(size, activation=activation))
        return model

    def call(self, inputs,tranining=False,mask=None):
        x_func = inputs[0]  # branch input dense tensor
        x_loc = inputs[1]  # trunk input raggged tensor
        mask=inputs[2]
        x_func=self.branch(x_func)
        x_loc=self.trunk(x_loc)
        ys=[]
        shift = 0
        for i in range(self.num_outputs):
            x_func_ = x_func[:, shift : shift + self.shift_size]
            x_loc_ = x_loc[:, shift : shift + self.shift_size]
            y = self.merge_branch_trunk(x_func_, x_loc_, i)
            ys.append(y)
            shift += self.shift_size
        ys=tf.stack(ys, axis=-1)
        ys=tf.squeeze(ys)
        if mask is None:
            return ys
        else:
            return ys*mask
    

    

    def merge_branch_trunk(self, x_func, x_loc, index=0):
        y = tf.einsum("bi,ni->bn", x_func, x_loc)
        y += self.b[index]
        return y
    

    def fit(self, *args, **kwargs):
        start_time = timeit.default_timer()
        his = super().fit(*args, **kwargs)
        if self.fit_history is None:
            self.fit_history = his.history
        else:
            self.fit_history = {
                key: self.fit_history[key] + (his.history)[key] for key in his.history
            }
        stop_time = timeit.default_timer()
        print("Training time: %.2f s " % (stop_time - start_time))
        return self.fit_history

    def predict(self, x, mask=None, *args, **kwargs):
        if isinstance(x, tf.data.Dataset):
            pred = super().predict(x, *args, **kwargs)
        else:
            pred = self.call(x)
            pred = pred.numpy()
        if mask is not None:
            pred = [pred[i][mask[i]] for i in range(pred.shape[0])]
        return pred

    def save_history(self, filebase):
        if self.fit_history is not None:
            if not os.path.exists(filebase):
                os.makedirs(filebase, exist_ok=True)
            his_file = os.path.join(filebase, "history.json")
            with open(his_file, "w") as f:
                json.dump(self.fit_history, f)

    def load_history(self, filebase):
        his_file = os.path.join(filebase, "history.json")
        if os.path.exists(his_file):
            with open(his_file, "r") as f:
                self.fit_history = json.load(f)
        return self.fit_history


# %%


class TripleCartesianProd:
    """Dataset with each data point as a triple. The ordered pair of the first two
    elements are created from a Cartesian product of the first two lists. If we compute
    the Cartesian product of the first two arrays, then we have a ``Triple`` dataset.

    This dataset can be used with the network ``DeepONetCartesianProd`` for operator
    learning.

    Args:
        X_data: A tuple of two NumPy arrays. The first element has the shape (`N1`,
            `dim1`), and the second element has the shape (`N2`, `dim2`).
        y_data: A NumPy array of shape (`N1`, `N2`).
        X_data[0] shape: (n_samples, n_features) n_features is or is not n_points
        X_data[1] shape: (n_points, n_dim)
        y_data shape: (n_samples, n_points)
        Aux_train shape: (n_samples, n_points)
    """

    def __init__(
        self, X_data, y_data=None, aux_data=None, batch_size=None, shuffle=True
    ):
        if len(X_data[0]) != y_data.shape[0] or len(X_data[1]) != y_data.shape[1]:
            print((X_data[0].shape),X_data[1].shape,y_data.shape)
            raise ValueError(
                "The dataset does not have the format of Cartesian product."
            )
        if aux_data is not None:
            if (
                len(X_data[0]) != aux_data.shape[0]
                #or len(X_data[1]) != aux_data.shape[1]
            ):
                print((X_data[0].shape), aux_data.shape)
                raise ValueError(
                    "The dataset does not have the format of Cartesian product."
                )
        self.X_data, self.y_data = X_data, y_data
        self.aux_data = aux_data
        self.shuffle = shuffle
        self.dataset = None
        self.batch_training_data(batch_size=batch_size, shuffle=shuffle)

    def batch_training_data(self, batch_size=None, shuffle=True):
        self.dataset = self.get_dataset_batches(
            self.X_data,
            self.y_data,
            self.aux_data,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def get_dataset_batches(
        self, x_data, y_data=None, aux=None, batch_size=None, shuffle=True
    ):
        if batch_size is None:
            bs = x_data[0].shape[0]
        else:
            bs = batch_size
        x_fun, x_loc = x_data
        if aux is None:
            aux = tf.zeros((x_fun.shape[0], x_loc.shape[0]))
        if y_data is None:
            y_data = tf.zeros((x_fun.shape[0], x_loc.shape[0]))
        dataset = tf.data.Dataset.from_tensor_slices((x_fun, y_data, aux))
        x_loc = tf.convert_to_tensor(x_loc)
        dataset = dataset.map(lambda x, y, aux: ((x, x_loc, aux), y))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=x_fun.shape[0])
        dataset = dataset.batch(bs)
        dataset = dataset.map(
            lambda x, y: ((x[0], tf.reshape(x[1][0], x_loc.shape), x[2]), y)
        )
        return dataset



class EvaluateDeepONetPDEs:
    """Generates the derivative of the outputs with respect to the trunck inputs.
    Args:
        model: DeepOnet.
        operator: Operator to apply to the outputs for derivative.
    """

    def __init__(self, model, operator):
        self.op = operator
        self.model = model

        @tf.function
        def op(inputs):
            y = self.model(inputs)
            # QB: inputs[1] is the input of the trunck
            # QB: y[0] is the output corresponding
            # to the first input sample of the branch input,
            # each time we only consider one sample
            return self.op(y[0][:, None], inputs[1])

        self.tf_op = op

    def __call__(self, inputs):
        self.value = []
        input_branch, input_trunck = inputs
        for inp in input_branch:
            x = (inp[None, :], input_trunck)
            self.value.append(self.tf_op(x))
        self.value = tf.stack(self.value).numpy()
        return self.value

    def get_values(self):
        return self.value


def jacobian(y, x):
    dydx = tf.gradients(y, x)[0]
    return dydx


def hessian(y, x):
    dydx = jacobian(y, x)  # (nb,dim_x)
    dydx_dx = []
    for i in range(dydx.shape[1]):
        dydxi = dydx[:, i : i + 1]
        dydxidx = jacobian(dydxi, x)  # (nb,nx)
        dydx_dx.append(dydxidx)
    dydx2 = tf.stack(dydx_dx, axis=1)  # (nb,nx,nx)
    return dydx2


def laplacian(y, x):
    dydx2 = hessian(y, x)
    laplacian_v = tf.reduce_sum(tf.linalg.diag_part(dydx2), axis=1)
    return laplacian_v

def laplacian_FD(u,dx,dy):
    """u shape=(batch_size,Ny,Nx)
    return shape=(batch_size,Ny-2,Nx-2)"""
    du_dxx=(u[:,1:-1,2:]-2*u[:,1:-1,1:-1]+u[:,1:-1,:-2])/dx**2
    du_dyy=(u[:,2:,1:-1]-2*u[:,1:-1,1:-1]+u[:,:-2,1:-1])/dy**2
    return du_dxx+du_dyy

# %%
def can_divide(a, b):
    return b != 0 and a % b == 0


def closest_divisor(num_samples, batch_size):
    if can_divide(num_samples, batch_size):
        return batch_size
    # Searching for the closest divisor

    distance = 1
    bs = batch_size
    while True:
        # Check both sides of batch_size
        if batch_size - distance > 1 and can_divide(num_samples, batch_size - distance):
            bs = batch_size - distance
            break
        if can_divide(num_samples, batch_size + distance):
            bs = batch_size + distance
            break
        distance += 1
    print("Batch size is changed to ", bs)
    return bs

def get_video(data_all):
    fig=plt.figure()
    ax1=plt.subplot(111)
    
    x_grid=1000*data_all["x"].squeeze()
    t_frames=data_all["t"].squeeze()
    
    T_true=data_all["T_true"].squeeze()
    alpha_true=data_all["alpha_true"].squeeze()
    
    T_pred=data_all["T_pred"].squeeze()
    alpha_pred=data_all["alpha_pred"].squeeze()
    
    y_data=[T_true,T_pred,alpha_true,alpha_pred]
    labels=["True Temperature","Predicted Temperature",r"True $\alpha$",r"Predicted $\alpha$"]
    # Initialize the first plot
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("Temperature [K]", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_ylim(260, 600)  # Adjust the range as needed
    
    lines=[]
    (line,) = ax1.plot(x_grid, T_true[0, :], color="blue",linewidth=2, label="True Temperature")
    lines.append(line)
    (line,) = ax1.plot(x_grid, T_pred[0, :],'--',  color="crimson", linewidth=2,label="Predicted Temperature")
    lines.append(line)
    
    
    # Create the second y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$\alpha$", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 1)
    (line,) = ax2.plot(x_grid, alpha_true[0, :], color="red",linewidth=2, label=r"True $\alpha$")
    lines.append(line)
    (line,) = ax2.plot(x_grid, alpha_pred[0, :],'--', color="navy",linewidth=2, label=r"Predicted $\alpha$")
    lines.append(line)
    
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")


    # Function to update the plot
    def update(i):
        for line, y in zip(lines, y_data):
            line.set_ydata(y[i, :])
        plt.title("Time = %.2f s" % t_frames[i])
        return lines


    # Create the animation

    ani = animation.FuncAnimation(
        fig, update, frames=len(t_frames), interval=100, blit=True
    )
    return ani

