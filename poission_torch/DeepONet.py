# %%
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

# %%


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


class DeepONetCartesianProd(nn.Module):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation="tanh"):
        super().__init__()
        if isinstance(activation, dict):
            self.activation_branch = activation["branch"]
            self.activation_trunk = activation["trunk"]
        else:
            self.activation_branch = self.activation_trunk = activation
            nn.Tanh()
        self.activation_branch = nn.ReLU()
        self.activation_trunk = nn.Tanh()
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
        loss=None,
        loss_names=None,
        metrics=None,
        decay=None,
        loss_weights=None,
        external_trainable_variables=None,
    ):
        self.optimizer = optimizer(self.parameters(), lr=lr)
        # TODO: Add multiple loss function
        self.loss_fn = loss
        # TODO:
        self.metrics = metrics
        self.decay = decay
        self.loss_weights = loss_weights
        self.external_trainable_variables = external_trainable_variables

    def collect_logs(self, losses_vals={}, batch_size=1):
        for key in losses_vals:
            if key not in self.logs:
                self.logs[key] = []
            self.logs[key].append(sum(losses_vals[key]) / batch_size)

    def print_logs(self, epoch, time):
        print(f"Epoch {epoch + 1} took {time:.2f}s", end=", ")
        for key, val in self.logs.items():
            if val:
                print(f"{key}: {val[-1]:.4e}", end=", ")
        print()

    def train_step(self, data, device="cpu"):
        inputs_, y_true = data
        y_true = y_true.to(device)
        input_branch, input_trunk = (inputs_[0].to(device), inputs_[1].to(device))
        self.optimizer.zero_grad()
        y_pred = self((input_branch, input_trunk))
        # TODO: Add multiple loss function
        loss = self.loss_fn(y_pred, y_true)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def validate_step(self, data, device="cpu"):
        inputs_, y_true = data
        y_true = y_true.to(device)
        input_branch, input_trunk = (inputs_[0].to(device), inputs_[1].to(device))
        y_pred = self((input_branch, input_trunk))
        loss = self.loss_fn(y_pred, y_true)
        return {"val_loss": loss.item()}

    def fit(
        self, train_loader, val_loader=None, device="cpu", epochs=10, callbacks=None
    ):
        self.to(device)
        loss_vals = {}
        for epoch in range(epochs):
            # train
            self.train()
            ts = timeit.default_timer()
            loss_vals = {}
            for data in train_loader:
                loss = self.train_step(data, device)
                for key, value in loss.items():
                    if key not in loss_vals:
                        loss_vals[key] = []
                    loss_vals[key].append(value)
            self.collect_logs(loss_vals, len(train_loader))
            # validate
            if val_loader is not None:
                self.eval()
                loss_vals = {}
                with torch.no_grad():
                    for data in val_loader:
                        loss = self.validate_step(data, device)
                        for key, value in loss.items():
                            if key not in loss_vals:
                                loss_vals[key] = []
                            loss_vals[key].append(value)
                self.collect_logs(loss_vals, len(val_loader))
            # callbacks at end of epoch
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
            aux= self.aux[idx]
            return (x_branch, x_trunk), aux, output_data

        else:
            return (x_branch, x_trunk),self.aux, output_data

    @staticmethod
    def custom_collate_fn(batch):
        # Assuming data is a list of tuples (sample, label)
        # batch=[((inp,out)),...], len=batch_size
        # inp=(x_branch, x_trunk)
        input,aux, out = zip(*batch)
        out = torch.tensor(np.array(out))
        input_branch, input_trunk = zip(*input)
        input_branch = torch.tensor(np.array(input_branch))
        input_trunk = torch.tensor(np.array(input_trunk[0]))
        if aux[0] is None:
            data=((input_branch, input_trunk), out)
 
        else:
            aux=torch.tensor(np.array(aux))
            data=((input_branch, input_trunk), aux, out)
 
        return data
            
            
     

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



class PDELoss(torch.nn.modules.loss._Loss):
    '''indices: list of indices of the output that we want to apply the loss to
        generally is the points inside the doamin
    '''
    def __init__(self,indices,pde_evaluator, size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__(size_average, reduce, reduction)
        self.indices = indices
        self.pde_evaluator=pde_evaluator
    def forward(self, y_pred=None, inputs=None,aux=None, model=None):
        '''kwargs: dictionary of outputs of the model'''
        inputs[1].requires_grad_(True)
        input_=(inputs[0],inputs[1][self.indices,:])
        aux=aux[:,self.indices]
        losses= self.pde_evaluator(input_,model,aux)
        return torch.mean(torch.square(losses))
        
