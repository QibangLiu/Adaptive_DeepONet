

# %%
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import json
# %%
class DeepONetCartesianProd(keras.Model):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation="tanh"):
        super().__init__()
        if isinstance(activation, dict):
            self.activation_branch = activation["branch"]
            self.activation_trunk = activation["trunk"]
        else:
            self.activation_branch = self.activation_trunk = activation
        self.trunk = self.build_net(layer_sizes_trunk, self.activation_trunk)
        self.branch = self.build_net(layer_sizes_branch, self.activation_branch)
        self.b = tf.Variable(tf.zeros(1))
        if self.trunk.output_shape != self.branch.output_shape:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        self.fit_history = None

    def build_net(self, layer_sizes, activation):
        # User-defined network
        if callable(layer_sizes[0]):
            return layer_sizes[0]
        # Fully connected network
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(layer_sizes[0],)))
        for size in layer_sizes[1:]:
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
            use_multiprocessing,
        )
        if self.fit_history is None:
            self.fit_history = his.history
        else:
            self.fit_history = {
                key: self.fit_history[key] + (his.history)[key] for key in his.history
            }
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
        X_train: A tuple of two NumPy arrays. The first element has the shape (`N1`,
            `dim1`), and the second element has the shape (`N2`, `dim2`).
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(self, X_train, y_train, X_test, y_test,aux_train=None,aux_test=None, batch_size=None):
        if len(X_train[0]) != y_train.shape[0] or len(X_train[1]) != y_train.shape[1]:
            raise ValueError(
                "The training dataset does not have the format of Cartesian product."
            )
        if len(X_test[0]) != y_test.shape[0] or len(X_test[1]) != y_test.shape[1]:
            raise ValueError(
                "The testing dataset does not have the format of Cartesian product."
            )
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test
        self.aux_train = aux_train
        self.aux_test = aux_test
        self.batch_training_data(batch_size=batch_size)
        self.batch_testing_data()

    def batch_training_data(self, batch_size=None, shuffle=True):
        self.train_dataset = self.get_dataset_batches(
            self.train_x, self.train_y, batch_size=batch_size, shuffle=shuffle
        )

    def batch_testing_data(self, batch_size=None, shuffle=False):
        self.test_dataset = self.get_dataset_batches(
            self.test_x, self.test_y, batch_size=batch_size, shuffle=shuffle
        )

    def get_dataset_batches(self, x_data, y_train=None,aux=None, batch_size=None, shuffle=True):
        if batch_size is None:
            bs = x_data[0].shape[0]
        else:
            bs = batch_size
        x_fun, x_loc = x_data
        if aux is None:
            aux = tf.zeros((x_fun.shape[0],1))
        if y_train is None:
            y_train = tf.zeros((x_fun.shape[0], x_loc.shape[0]))
        dataset = tf.data.Dataset.from_tensor_slices((x_fun, y_train,aux))
        x_loc = tf.convert_to_tensor(x_loc)
        dataset = dataset.map(lambda x, y,aux: ((x, x_loc,aux), y))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=x_fun.shape[0])
        dataset = dataset.batch(bs)
        dataset = dataset.map(
            lambda x, y: ((x[0], tf.reshape(x[1][0], x_loc.shape),x[2]), y)
        )
        return dataset

