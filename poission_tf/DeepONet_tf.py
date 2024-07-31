# %%
import tensorflow as tf
from tensorflow import keras
import os
import json
import timeit
# import deepxde as dde
# from deepxde import utils
import numpy as np


# %%
class DeepONetCartesianProd(keras.Model):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation="tanh",apply_activation_outlayer=True):
        super().__init__()
        if isinstance(activation, dict):
            self.activation_branch = activation["branch"]
            self.activation_trunk = activation["trunk"]
        else:
            self.activation_branch = self.activation_trunk = activation
        self.trunk = self.build_net(layer_sizes_trunk, self.activation_trunk, apply_activation_outlayer)
        self.branch = self.build_net(layer_sizes_branch, self.activation_branch,apply_activation_outlayer)
        self.b = tf.Variable(tf.zeros(1))
        if self.trunk.output_shape != self.branch.output_shape:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        self.fit_history = None

    def build_net(self, layer_sizes, activation,apply_activation_outlayer=True):
        # User-defined network
        if callable(layer_sizes[0]):
            return layer_sizes[0]
        # Fully connected network
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(layer_sizes[0],)))
        for size in layer_sizes[1:]:
            if size == layer_sizes[-1] and not apply_activation_outlayer:
                model.add(tf.keras.layers.Dense(size,activation='linear'))
            else:
                model.add(tf.keras.layers.Dense(size, activation=activation))
        return model

    def call(self, inputs):
        x_func = inputs[0]  # branch input
        x_loc = inputs[1]  # trunk input
        x_func = self.branch(x_func)
        x_loc = self.trunk(x_loc)

        y = self.merge_branch_trunk(x_func, x_loc)
        return y

    def merge_branch_trunk(self, x_func, x_loc):
        y = tf.einsum("bi,ni->bn", x_func, x_loc)
        y += self.b
        return y

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):
        start_time = timeit.default_timer()
        his = super().fit(
            x,
            y,
            batch_size,
            epochs,
            verbose,
            callbacks,
            validation_split,
            validation_data,
            shuffle,
            class_weight,
            sample_weight,
            initial_epoch,
            steps_per_epoch,
            validation_steps,
            validation_batch_size,
            validation_freq,
            max_queue_size,
            workers,
        )
        if self.fit_history is None:
            self.fit_history = his.history
        else:
            self.fit_history = {
                key: self.fit_history[key] + (his.history)[key] for key in his.history
            }
        stop_time = timeit.default_timer()
        print("Training time: %.2f s "%(stop_time - start_time))
        return self.fit_history

    def predict(
        self,
        x,
        batch_size=None,
        verbose="auto",
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):
        if isinstance(x, tf.data.Dataset):
            pred = super().predict(
                x,
                batch_size,
                verbose,
                steps,
                callbacks,
                max_queue_size,
                workers,
                use_multiprocessing,
            )
        else:
            pred = self.call(x)
            pred = pred.numpy()
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
            raise ValueError(
                "The training dataset does not have the format of Cartesian product."
            )
        if aux_data is not None:
            if (
                len(X_data[0]) != aux_data.shape[0]
                or len(X_data[1]) != aux_data.shape[1]
            ):
                raise ValueError(
                    "The training dataset does not have the format of Cartesian product."
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
