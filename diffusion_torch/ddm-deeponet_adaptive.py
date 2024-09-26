#!/usr/bin/env python
# coding: utf-8

# In[1]:
# %run ddm-deeponet_adaptive.py 200 0 2 1 1 1
# %%
import sys
import os
import DeepONet
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.io as scio
import timeit
from torch.utils.data import Dataset, DataLoader

# In[]
# %run ddm-deeponet_adaptive.py 2000 0 2 1 1 1
prefix_filebase = "./saved_model"
str_dN, str_start, str_end = sys.argv[1:4]
str_k, str_c = sys.argv[4:-1]
str_caseID = sys.argv[-1]
# str_dN, str_start, str_end = '2000', '0', '2'
# str_k, str_c = '1', '1'
# str_caseID = '1'
# In[3]:

# In[3]:

fenics_data = scio.loadmat("./TrainingData/diffusion_gauss_cov20k.mat")
x_grid = fenics_data["x_grid"].squeeze().astype(np.float32)  # shape (Nx,)
t_grid = fenics_data["t_grid"].squeeze().astype(np.float32)
dx=x_grid[1]-x_grid[0]
dt=t_grid[1]-t_grid[0]
Nx, Nt = len(x_grid), len(t_grid)
m = Nx * Nt
x_grid, t_grid = np.meshgrid(x_grid, t_grid)
ICs_raw = fenics_data["ICs"].astype(np.float32) 
solutions_raw = fenics_data["solutions"].astype(np.float32)  # shape (N, Nt, Nx)
solutions_raw=solutions_raw.reshape(-1,m)


shift_solution, scaler_solution = np.mean(solutions_raw), np.std(solutions_raw) 
shift_ICs, scaler_ICs= np.mean(ICs_raw), np.std(ICs_raw)
solutions = (solutions_raw - shift_solution) / scaler_solution
ICs = (ICs_raw - shift_ICs) / scaler_ICs



num_test = 500
num_train = -num_test
u0_train = ICs[:num_train]
u0_train_raw = ICs_raw[:num_train]  
u0_testing = ICs[-num_test:]
u0_testing_raw = ICs_raw[-num_test:]
s_train = solutions[:num_train]
s_testing = solutions[-num_test:]
s_train_raw = solutions_raw[:num_train]
s_testing_raw = solutions_raw[-num_test:]

xy_train_testing = np.concatenate(
    [x_grid.reshape(-1, 1), t_grid.reshape(-1, 1)], axis=1
)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_trunk = torch.tensor(xy_train_testing).to(device)
x_trunk.requires_grad_(True)

x_train_branch = torch.tensor(u0_train).to(device)
y_train = torch.tensor(s_train).to(device)
aux_train = torch.tensor(u0_train_raw).to(device)

x_test = (torch.tensor(u0_testing).to(device), x_trunk)
y_test = torch.tensor(s_testing).to(device)
aux_test = torch.tensor(u0_testing_raw).to(device)
dataset_test = DeepONet.TripleCartesianProd(x_test, y_test)
test_loader = DataLoader(
    dataset_test,
    batch_size=dataset_test.__len__(),
    collate_fn=dataset_test.custom_collate_fn,
)
# %%

mse = nn.MSELoss()
model = DeepONet.DeepONetCartesianProd(
    [Nx, 200, 200, 200, 200, 200, 200],
    [2, 200, 200, 200, 200, 200, 200],
    activation={"branch": nn.ReLU(), "trunk": nn.Tanh()},
)
# initial_weights = model.get_weights()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
model.compile(optimizer=optimizer, loss=mse)
_=model.to(device)


# %%
alpha=0.01
def ResidualError(y, x, aux=None):
    # nb: batch size, =Nt*Nx
    # nx: dimension of trunck input, =2
    dYdX = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0] #(nb,nx)
    dYdx = dYdX[:, 0:1] # (nb,1)
    dYdt = dYdX[:, 1:2] # (nb,1)
    dYdxdX = DeepONet.jacobian(dYdx, x, create_graph=True) #(nb,nx)
    dYdx2=dYdxdX[:,0:1] # (nb,1)
    #
    y_inv=scaler_solution*y+shift_solution
    lhs=(dYdt*scaler_solution+y_inv-alpha*dYdx2*scaler_solution)
    lhs=lhs.reshape(Nt,Nx)
    aux_norm=torch.norm(aux) 
    l2s=[torch.norm(lhs[i,:]-aux.squeeze())/aux_norm for i in range(lhs.shape[0])]
    
    res=torch.mean(torch.stack(l2s))
    return res


res_op = DeepONet.EvaluateDeepONetPDEs(ResidualError,model=model)


def ErrorMeasure(X, rhs):
    res = res_op(X, aux=rhs)
    res = res.detach().cpu().numpy()
    # res = np.squeeze(res)
    # rhs_np = rhs.detach().cpu().numpy()
    # return np.linalg.norm(res, axis=1) / np.linalg.norm(rhs_np, axis=1)
    return res

def sampling(iter, dN, pre_filebase):
    all_data_idx = np.arange(len(x_train_branch))
    currTrainDataIDX__ = None
    if iter == 0:
        currTrainDataIDX__ = np.random.choice(
            a=all_data_idx, size=dN, replace=False
        )
    else:
        pre_train_data_idx = np.genfromtxt(
            os.path.join(pre_filebase, "trainDataIDX.csv"), dtype=int, delimiter=","
        )
        # potential training data
        potential_train_data_idx = np.delete(all_data_idx, pre_train_data_idx)
        print('befor error measure')
        torch.cuda.empty_cache()
        LR = ErrorMeasure(
            (x_train_branch[potential_train_data_idx], x_trunk),
            aux_train[potential_train_data_idx]
        )
        probility = np.power(LR, k) / np.power(LR, k).mean() + c
        probility_normalized = probility / np.sum(probility)
        new_training_data_idx = np.random.choice(
            a=potential_train_data_idx,
            size=dN,
            replace=False,
            p=probility_normalized,
        )
        currTrainDataIDX__ = np.concatenate((pre_train_data_idx, new_training_data_idx))
    return currTrainDataIDX__


# %%
numCase_add = int(str_dN)
k = float(str_k)
c = float(str_c)

iter_start, iter_end = int(str_start), int(str_end)
project_name = "adapt_k" + str_k + "c" + str_c + "dN" + str_dN + "case" + str_caseID
filebase = os.path.join(prefix_filebase, project_name)
start_time = timeit.default_timer()
# %%

if iter_start != 0:
    pre_filebase = os.path.join(filebase, "iter" + str(iter_start - 1))
    model.load_weights(os.path.join(pre_filebase, "model.ckpt"), device)
    model.load_logs(pre_filebase)

for iter in range(iter_start, iter_end):
    print("====iter=======:%d" % iter)
    pre_filebase = os.path.join(filebase, "iter" + str(iter - 1))
    current_filebase = os.path.join(filebase, "iter" + str(iter))
    os.makedirs(current_filebase, exist_ok=True)
    # restore_path = find_checkpoint_2restore(pre_filebase)
    currTrainDataIDX = sampling(iter, numCase_add, pre_filebase)

    np.savetxt(
        os.path.join(current_filebase, "trainDataIDX.csv"),
        currTrainDataIDX,
        fmt="%d",
        delimiter=",",
    )

    curr_u_train = x_train_branch[currTrainDataIDX]
    x_train = (curr_u_train, x_trunk)
    curr_y_train = y_train[currTrainDataIDX]
    current_s_train=s_train[currTrainDataIDX]
    dataset_train = DeepONet.TripleCartesianProd(x_train, curr_y_train)
    train_loader = DataLoader(
        dataset_train,
        batch_size=64,
        shuffle=True,
        collate_fn=dataset_train.custom_collate_fn,
    )
    checkpoint_fname = os.path.join(current_filebase, "model.ckpt")
    checkpoint_callback = DeepONet.ModelCheckpoint(
        checkpoint_fname, monitor="val_loss", save_best_only=True
    )
    # model.set_weights(initial_weights)
    h = model.fit(
        train_loader,
        test_loader,
        epochs=1500,
        #callbacks=checkpoint_callback,
    )
    checkpoint_callback._save_model(model)
    model.save_logs(filebase)
    model.load_weights(checkpoint_fname, device)
    y_pred = model.predict(test_loader)
    error_test = np.linalg.norm(s_testing - y_pred, axis=1) / np.linalg.norm(
        s_testing, axis=1
    )
    np.savetxt(
        os.path.join(current_filebase, "TestL2Error.csv"),
        error_test,
        fmt="%.4e",
        delimiter=",",
    )
    
    y_pred_true=model.predict(train_loader)
    error_train = np.linalg.norm(current_s_train - y_pred_true, axis=1) \
        / np.linalg.norm(current_s_train, axis=1)
    np.savetxt(
        os.path.join(current_filebase, "TrainL2Error.csv"),
        error_test,
        fmt="%.4e",
        delimiter=",",
    )
    
    stop_time = timeit.default_timer()
    print("training Run time so far: ", round(stop_time - start_time, 2), "(s)")
    # fig=plt.figure()
    # plt.hist(error_test)
    # plt.xlabel("L2 relative error")
    # plt.ylabel("frequency")
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["loss"], label="loss")
ax.plot(h["val_loss"], label="val_loss")
ax.legend()
ax.set_yscale("log")


# Plotting Results
# %%
x_validate = x_test
y_validate = s_testing_raw
u0_validate = u0_testing_raw

# input_branch, input_trunk = x_validate[0], x_validate[1]
# input_trunk.requires_grad_(True)

y_pred_out = model(x_validate)
y_pred = y_pred_out.detach().cpu().numpy()
y_pred = y_pred * scaler_solution + shift_solution

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
sort_idx = np.argsort(error_s)
min_index = sort_idx[0]
max_index = sort_idx[-1]
median_index = sort_idx[len(sort_idx) // 2]

# Print the indexes
print("Index for minimum element:", min_index)
print("Index for maximum element:", max_index)
print("Index for median element:", median_index)

min_median_max_index = np.array([min_index, median_index, max_index])
nr, nc = 1, 3
fig = plt.figure(figsize=(18, 5))
for i, index in enumerate(min_median_max_index):
    y_validate_nx_ny = y_validate[index].reshape(Nt, Nx)
    s_pred_nx_ny = y_pred[index].reshape(Nt, Nx)

    ax = plt.subplot(nr, nc, i + 1)
    # py.figure(figsize = (14,7))
    num_curv = 5
    step = (Nt - 16) / (num_curv + 1)
    curv = [int(16 + (i + 1) * step) for i in range(num_curv)]
    curv[-1] = Nt - 1
    for j, c in enumerate(curv):
        if j == 0:
            ax.plot(x_grid[c, :], y_validate_nx_ny[c, :], "b", label="True")
            ax.plot(x_grid[c, :], s_pred_nx_ny[c, :], "r--", label="Predicted")
        else:
            ax.plot(x_grid[c, :], y_validate_nx_ny[c, :], "b")
            ax.plot(x_grid[c, :], s_pred_nx_ny[c, :], "r--")
    ax.legend()
    ax.set_title(f"b = %.2f" % u0_validate[index, 0])
plt.tight_layout()

# %%

# res = ErrorMeasure(x_validate,aux_train)

# correlation = np.corrcoef(error_s, res)[0, 1]
# print("Pearson correlation coefficient:", correlation)
# gap=2
# fig = plt.figure()
# ax=plt.subplot(1,1,1)
# ax.plot(error_s[sort_idx][::gap],(res)[sort_idx][::gap],'o')
# # Fit a straight line
# coefficients = np.polyfit(error_s[sort_idx], (res)[sort_idx], 1)
# line = np.poly1d(coefficients)

# # Plot the line
# ax.plot(error_s[sort_idx], (res)[sort_idx], 'o')
# ax.plot(error_s[sort_idx], line(error_s[sort_idx]), color='red')

# # Add labels and title
# ax.set_xlabel('L2 Relative Error')
# ax.set_ylabel('Residual Error')
# ax.set_title('L2 Relative Error vs Residual Error')
# %%