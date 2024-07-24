import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import json
from abc import ABC, abstractmethod
import timeit
import numpy as np
from torch.nn.modules.loss import _Loss
import inspect

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%


class NameGenerator:
    def __init__(self):
        self.name_count = {}

    def get_name(self, item):
        try:
            return item.__name__
        except AttributeError:
            try:
                return type(item).__base__.__name__
            except AttributeError:
                return type(item).__name__

    def __call__(self, item):
        name_type = self.get_name(item)
        if name_type not in self.name_count:
            self.name_count[name_type] = 0
        else:
            self.name_count[name_type] += 1

        if self.name_count[name_type] == 0:
            return name_type
        else:
            return f"{name_type}_{self.name_count[name_type]}"


class ModelCheckpoint:
    def __init__(
        self, filepath, monitor="val_loss", verbose=0, save_best_only=False, mode="min"
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.best = None
        self.mode = mode

        if self.mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")

        if self.mode == "min":
            self.monitor_op = lambda x, y: x < y
            self.best = float("inf")
        else:
            self.monitor_op = lambda x, y: x > y
            self.best = float("-inf")

    def __call__(self, epoch, logs=None, model=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        current = current[-1]

        if current is None:
            raise ValueError(f"Monitor value '{self.monitor}' not found in logs")

        if self.save_best_only:
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print(
                        f"Epoch {epoch + 1}: {self.monitor} improved from {self.best} to {current}, saving model to {self.filepath}"
                    )
                self.best = current
                self._save_model(model)
            else:
                if self.verbose > 1:
                    print(
                        f"Epoch {epoch + 1}: {self.monitor} did not improve from {self.best}"
                    )
        else:
            if self.verbose > 0:
                print(f"Epoch {epoch + 1}: saving model to {self.filepath}")
            self._save_model(model)

    def _save_model(self, model):
        torch.save(model.state_dict(), self.filepath)


class Losses(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self,y_true=None, y_pred=None, inputs=None,aux=None, model=None):
       
        pass


# %%


class DeepONetCartesianProd(nn.Module):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation="tanh"):
        super().__init__()
        if isinstance(activation, dict):
            self.activation_branch = activation["branch"]
            self.activation_trunk = activation["trunk"]
        else:
            self.activation_branch = self.activation_trunk = activation
            nn.Tanh()
        self.activation_branch=nn.ReLU()
        self.activation_trunk=nn.Tanh()    
        self.trunk = self.build_net(layer_sizes_trunk, self.activation_trunk)
        self.branch = self.build_net(layer_sizes_branch, self.activation_branch)
        self.b = nn.Parameter(torch.tensor(0.0))
        self.name_generator = NameGenerator()
        self.logs = {}
        # if self.trunk.output_shape != self.branch.output_shape:
        #     raise AssertionError(
        #         "Output sizes of branch net and trunk net do not match."
        #     )
        # self.logs = {"Epoch": []}

    def build_net(self, layer_sizes, activation):
        # User-defined network
        if callable(layer_sizes[0]):
            return layer_sizes[0]
        # Fully connected network
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(activation)
        return nn.Sequential(*layers)

    def forward(self, inputs):
        x_func = inputs[0]  # branch input
        x_loc = inputs[1]  # trunk input

        x_func = self.branch(x_func)
        x_loc = self.trunk(x_loc)
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )

        y = self.merge_branch_trunk(x_func, x_loc)
        return y

    def merge_branch_trunk(self, x_func, x_loc):
        y = torch.einsum("bi,ni->bn", x_func, x_loc)
        y += self.b
        return y

    def compile(
        self,
        optimizer,
        lr=None,
        loss=[torch.nn.MSELoss()],
        loss_names=None,
        metrics=None,
        decay=None,
        loss_weights=None,
        external_trainable_variables=None,
    ):
        self.optimizer = optimizer(self.parameters(), lr=lr)
        loss_names_ = []
        if loss_names is None:
            for ls_fn in loss:
                ls_name = self.name_generator(ls_fn)
                loss_names_.append(ls_name)
        else:
            for ls_n, ls_fn in zip(loss_names, loss):
                if ls_n == "loss":
                    # loss name can not be 'loss'
                    ls_n = self.name_generator(ls_fn.__class__.__bases__[0].__name__)
                loss_names_.append(ls_n)
        if len(loss_names_) == 1:
            loss_names_ = ["loss"]  # key of total loss is 'loss'
        self.losses_fn = {key: val for key, val in zip(loss_names_, loss)}
        self.losses_vals = {key: [] for key in self.losses_fn}
        if len(self.losses_fn) > 1:
            self.losses_vals["loss"] = []

        self.val_losses_vals = {"val_" + key: [] for key in self.losses_fn}
        for key in self.losses_vals:
            if key not in self.logs:
                self.logs[key] = []
        for key in self.val_losses_vals:
            if key not in self.logs:
                self.logs[key] = []

        self.metrics = metrics
        self.decay = decay
        self.loss_weights = loss_weights
        self.external_trainable_variables = external_trainable_variables

    def collect_logs(self, losses_vals, batch_size):
        for key in losses_vals:
            self.logs[key].append(sum(losses_vals[key]) / batch_size)

    def print_logs(self, epoch, time):
        print(f"Epoch {epoch + 1} took {time:.2f}s", end=", ")
        for key, val in self.logs.items():
            if val:
                print(f"{key}: {val[-1]:.4e}", end=", ")
        print()

    def evaluLoss(self, losses_vals, y_true=None, y_pred=None, inputs=None,aux=None, model=None):
        losses_v = []
        pre = ""
        if "loss" not in losses_vals:
            pre = "val_"
        for loss_key, val in self.losses_fn.items():
            if len(inspect.signature(val.forward).parameters) == 2:
                loss = val(y_pred, y_true)
            else:
                loss = val( y_true=y_true,y_pred=y_pred,inputs=inputs,aux=aux, model=model)
            losses_v.append(loss)
            losses_vals[pre + loss_key].append(loss.item())
        total = torch.sum(torch.stack(losses_v))
        if len(self.losses_fn) > 1:
            losses_vals[pre + "loss"].append(total.item())
        return total

    def fit(
        self,
        train_loader,
        val_loader=None,
        device="cpu",
        epochs=10,
        callbacks=None,
        get_output=True
    ):

        self.to(device)

        for epoch in range(epochs):
            self.train()
            ts = timeit.default_timer()
            for key in self.losses_vals:
                self.losses_vals[key] = []

            # train
            for batch_idx, (inputs_, target) in enumerate(train_loader):
                target =  target.to(device)
                input_branch, input_trunk = (inputs_[0].to(device), inputs_[1].to(device))
                if len(inputs_)==3:
                    aux=inputs_[2].to(device)
                else:   
                    aux=None
                self.optimizer.zero_grad()
                inp = (input_branch, input_trunk)
                if get_output:
                    output = self(inp)
                else:
                    output = None
                loss = self.evaluLoss(self.losses_vals, target, output, inp,aux, self)
                loss.backward()
                self.optimizer.step()
            self.collect_logs(self.losses_vals, len(train_loader))

            # validate
            if val_loader is not None:
                self.eval()  # Set the model to evaluation mode
                for key in self.val_losses_vals:
                    self.val_losses_vals[key] = []
                with torch.no_grad():
                    for batch_idx, (inputs_, target) in enumerate(val_loader):
                        target = target.to(device)
                        if len(inputs_)==3:
                            aux=inputs_[2].to(device)
                        else:   
                            aux=None
                        input_branch, input_trunk = (
                            inputs_[0].to(device),
                            inputs_[1].to(device),
                        )
                        inp = (input_branch, input_trunk)
                        output = self(inp)
                        self.evaluLoss(self.val_losses_vals, target, output, inp,aux, self)
                self.collect_logs(self.val_losses_vals, len(val_loader))
            if callbacks is not None:
                callbacks(epoch, self.logs, self)

            te = timeit.default_timer()

            self.print_logs(epoch, (te - ts))
        return self.logs

    def predict(self, x, device):
        # Set the model to evaluation mode
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            if isinstance(x, torch.utils.data.DataLoader):
                predictions = []
                for inputs, _ in x:
                    input_branch, input_trunk = (
                        inputs[0].to(device),
                        inputs[1].to(device),
                    )
                    # Forward pass
                    preds = self((input_branch, input_trunk))
                    predictions.extend(preds.cpu().numpy())
                predictions = np.array(predictions)
            else:
                if not isinstance(x[0], torch.Tensor):
                    input_branch, input_trunk = (
                        torch.Tensor(x[0]).to(device),
                        torch.Tensor(x[1]).to(device),
                    )
                else:
                    input_branch, input_trunk = (x[0].to(device), x[1].to(device))
                predictions = self((input_branch, input_trunk))
                predictions = predictions.cpu().numpy()

        return predictions

    def save_logs(self, filebase):
        if self.logs is not None:
            if not os.path.exists(filebase):
                os.makedirs(filebase, exist_ok=True)
            his_file = os.path.join(filebase, "logs.json")
            with open(his_file, "w") as f:
                json.dump(self.logs, f)

    def load_logs(self, filebase):
        his_file = os.path.join(filebase, "logs.json")
        if os.path.exists(his_file):
            with open(his_file, "r") as f:
                self.logs = json.load(f)
        return self.logs

    def save_weights(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_weights(self, filepath):
        self.load_state_dict(torch.load(filepath))
        self.eval()


# %%


class TripleCartesianProd(Dataset):
    """Dataset with each data point as a triple. The ordered pair of the first two
    elements are created from a Cartesian product of the first two lists. If we compute
    the Cartesian product of the first two arrays, then we have a ``Triple`` dataset.

    This dataset can be used with the network ``DeepONetCartesianProd`` for operator
    learning.
    """

    def __init__(self, X_train, y_train, aux=None, transform=None):
        # Initialize dataset parameters, load data, etc.
        # TODO: add transform
        self.transform = transform
        self.X_branch, self.X_trunk = X_train
        self.y = y_train
        self.aux = aux
        if (
            len(self.X_branch) != self.y.shape[0]
            or len(self.X_trunk) != self.y.shape[1]
        ):
            raise ValueError(
                "The training dataset does not have the format of Cartesian product."
            )
        if aux is not None and (
            len(aux) != self.y.shape[0] or aux.shape[1] != self.y.shape[1]
        ):
            raise ValueError("The auxiliary data does not have the correct shape.")

    def __len__(self):
        # Return the size of the dataset
        return len(self.X_branch)

    def __getitem__(self, idx):
        # Retrieve and preprocess a sample from the dataset
        # Example: Assuming your data is a tuple (input_data, output_data)
        x_branch = self.X_branch[idx]
        output_data = self.y[idx]
        x_trunk = self.X_trunk
        # TODO: add transform
        # if self.transform:
        #     input_data = self.transform(input_data)
        #     output_data = self.transform(output_data)
        # TODO: make x_trunk lenght be 1
        if self.aux is not None:
            return (x_branch, x_trunk, self.aux[idx]), output_data

        else:
            return (x_branch, x_trunk), output_data

    @staticmethod
    def custom_collate_fn(batch):
        # Assuming data is a list of tuples (sample, label)
        # batch=[((inp,out)),...], len=batch_size
        # inp=(x_branch, x_trunk)
        input, out = zip(*batch)
        out = torch.tensor(np.array(out))
        if len(input[0])==2:
            input_branch, input_trunk = zip(*input)
            input_branch = torch.tensor(np.array(input_branch))
            input_trunk = torch.tensor(np.array(input_trunk[0]))
            return (input_branch, input_trunk), out
        elif len(input[0])==3:
            input_branch, input_trunk, aux = zip(*input)
            input_branch = torch.tensor(np.array(input_branch))
            input_trunk = torch.tensor(np.array(input_trunk[0]))
            aux=torch.tensor(np.array(aux))
            return (input_branch, input_trunk, aux), out
        else:   
            raise ValueError("Only accept 2 or 3 input data.")
            
            
     

# %%

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
