#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys
import os
import DeepONet_tf as DeepONet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.io as scio
import tensorflow as tf
import h5py
import timeit
# dde.backend.set_default_backend("tensorflow")
# dde.config.set_default_float("float64")
# %run ddm-deeponet_adaptive_tf.py 200 0 2 0 1 0
# In[]

prefix_filebase = "./saved_model"
str_dN, str_start, str_end = sys.argv[1:4]
str_k, str_c = sys.argv[4:-1]
str_caseID = sys.argv[-1]

# In[3]:
# fenics_data = scio.loadmat("./TrainingData/poisson_gauss_cov20k.mat")
# x_grid = fenics_data["x_grid"].astype(np.float32)  # shape (Ny, Nx)
# y_grid = fenics_data["y_grid"].astype(np.float32)
# source_terms_raw = fenics_data["source_terms"].astype(np.float32).reshape(-1, Nx * Ny)  # shape (N, Ny* Nx)
# solutions_raw = fenics_data["solutions"].astype(np.float32).reshape(-1, Nx * Ny)  # shape (N, Ny* Nx)

hf=h5py.File("./TrainingData/poisson_2XY_gauss_cov40K.h5", 'r')
x_grid=hf['x_grid'][:].astype(np.float32)  # shape (Nx,)\
y_grid = hf["y_grid"][:].astype(np.float32)
Nx,Ny=x_grid.shape[1],x_grid.shape[0]
m = Nx * Ny
source_terms_raw = hf["source_terms"][:].astype(np.float32).reshape(-1, Nx * Ny) 
solutions_raw = hf["solutions"][:].astype(np.float32).reshape(-1, Nx * Ny) # shape (N, Nt, Nx)
hf.close()



shift_solution, scaler_solution = np.mean(solutions_raw), np.std(solutions_raw)
shift_source, scaler_source = np.mean(source_terms_raw), np.std(source_terms_raw)
# shift_solution, scaler_solution = np.min(solutions_raw), np.max(solutions_raw)-np.min(solutions_raw)
# shift_source, scaler_source = np.min(source_terms_raw), np.max(source_terms_raw)-np.min(source_terms_raw)
# shift_solution, scaler_solution = 0, (np.max(solutions_raw)-np.min(solutions_raw))*0.5
# shift_source, scaler_source = 0, (np.max(source_terms_raw)-np.min(source_terms_raw))*0.5
# shift_solution, scaler_solution = 0, 1
# shift_source, scaler_source = 0, 1
solutions = (solutions_raw - shift_solution) / scaler_solution
source_terms = (source_terms_raw - shift_source) / scaler_source


num_train = -1000
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

# %%

x_test = (u0_testing, xy_train_testing)
y_test = s_testing
data_test = DeepONet.TripleCartesianProd(x_test, y_test, shuffle=False)
# %%

data_train = None
model = DeepONet.DeepONetCartesianProd(
    [m, 100, 100, 100, 100, 100, 100],
    [2, 100, 100, 100, 100, 100, 100],
    {"branch": "relu", "trunk": "tanh"},
)
initial_weights = model.get_weights()
laplace_op = DeepONet.EvaluateDeepONetPDEs(model, DeepONet.laplacian)

def ErrorMeasure(X, rhs, method="AD"):
    if method == "AD":
        lhs = -0.01 * (laplace_op(X)) * scaler_solution
        lhs = np.squeeze(lhs)
        return np.linalg.norm(lhs - rhs, axis=1) / np.linalg.norm(rhs, axis=1)
    elif method == "FD":
        with tf.device("/CPU:0"):
            y = model.predict(X)
            y = y.reshape(-1, Ny, Nx) * scaler_solution + shift_solution
            lsh = DeepONet.laplacian_FD(y, dx, dy).reshape(-1, (Ny - 2) * (Nx - 2))
            rhs__ = rhs.reshape(-1, Ny, Nx)[:, 1:-1, 1:-1]
            rsh__ = rhs__.reshape(-1, (Ny - 2) * (Nx - 2))
            return np.linalg.norm(lsh - rsh__, axis=1) / np.linalg.norm(rsh__, axis=1)
    else:
        raise ValueError("method should be AD or FD")

def L2RelativeError(x_validate, y_validate):
    y_pred = model.predict(x_validate)
    y_pred = y_pred * scaler_solution + shift_solution
    error_s = np.linalg.norm(y_validate - y_pred, axis=1) / np.linalg.norm(
        y_validate, axis=1
    )
    return error_s, y_pred

def sampling(iter, dN, pre_filebase):
    all_data_idx = np.arange(len(u0_train))
    currTrainDataIDX__ = None
    if iter == 0:
        currTrainDataIDX__ = np.random.choice(a=all_data_idx, size=dN, replace=False)
    else:
        pre_train_data_idx = np.genfromtxt(
            os.path.join(pre_filebase, "trainDataIDX.csv"), dtype=int, delimiter=","
        )
        # potential training data
        potential_train_data_idx = np.delete(all_data_idx, pre_train_data_idx)

        LR = ErrorMeasure(
            (u0_train[potential_train_data_idx], xy_train_testing),
            u0_train_raw[potential_train_data_idx],method="AD"
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

project_name = (
     "adapt_k" + str_k + "c" + str_c + "dN" + str_dN + "case" + str_caseID
)
filebase = os.path.join(prefix_filebase, project_name)
start_time = timeit.default_timer()

if iter_start != 0:
    pre_filebase = os.path.join(filebase, "iter" + str(iter_start - 1))
    model.load_history(pre_filebase)
    model.load_weights(os.path.join(pre_filebase, "model.ckpt"))

correla_file=os.path.join(filebase, "correlation.csv")
if os.path.exists(correla_file):
    correla_hist = pd.read_csv(correla_file).to_dict(orient="list")
else:
    correla_hist = {"num_sample":[], "spearman": [], "pearson": []}

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

    curr_u_train = u0_train[currTrainDataIDX]
    curr_x_train = (curr_u_train, xy_train_testing)
    curr_y_train = s_train[currTrainDataIDX]
    curr_y_train_raw = s_train_raw[currTrainDataIDX]
    data_train = DeepONet.TripleCartesianProd(curr_x_train, curr_y_train, batch_size=64)
    optm = tf.keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(optimizer=optm, loss="mse")
    checkpoint_fname = os.path.join(current_filebase, "model.ckpt")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_fname,
        save_weights_only=True,
        monitor="loss",
        verbose=0,
        save_freq="epoch",
        save_best_only=True,
        mode="min",
    )
    # model.set_weights(initial_weights)
    h = model.fit(
        data_train.dataset,
        validation_data=data_test.dataset,
        epochs=1000,
        verbose=2,
        callbacks=[model_checkpoint],
    )
    model.save_history(filebase)
    model.load_weights(checkpoint_fname)

    error_test, _ = L2RelativeError(data_test.X_data, s_testing_raw)
    error_train, _= L2RelativeError(curr_x_train, curr_y_train_raw)
    np.savetxt(
        os.path.join(current_filebase, "TestL2Error.csv"),
        error_test,
        fmt="%.4e",
        delimiter=",",
    )
    
    np.savetxt(
        os.path.join(current_filebase, "TrainL2Error.csv"),
        error_train,
        fmt="%.4e",
        delimiter=",",
    )
    
    res_test = ErrorMeasure(
            data_test.X_data,
            u0_testing_raw
        )
    r_spearman, _ = spearmanr(error_test, res_test)
    print(f"Spearman's rank correlation coefficient: {r_spearman}")
    pearson_coe = np.corrcoef(error_test, res_test)[0, 1]
    print("Pearson correlation coefficient:", pearson_coe)
    
    correla_hist["num_sample"].append(len(currTrainDataIDX))
    correla_hist["spearman"].append(r_spearman)
    correla_hist["pearson"].append(pearson_coe)
    
    stop_time = timeit.default_timer()
    print("training Run time so far: ", round(stop_time - start_time, 2), "(s)")

sys.exit(0)
# Plotting Results
# %%
u0_validate = u0_train_raw[currTrainDataIDX]
y_validate = s_train_raw[currTrainDataIDX]
x_validate = (u0_train[currTrainDataIDX], xy_train_testing)
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

# %%
laplace_op_val_ = laplace_op((x_validate[0][min_median_max_index], x_validate[1]))
# %%
laplace_op_val = -0.01 * laplace_op_val_ * scaler_solution


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
ErrorMeasure(
    (x_validate[0][min_median_max_index], x_validate[1]),
    u0_validate[min_median_max_index],
)
