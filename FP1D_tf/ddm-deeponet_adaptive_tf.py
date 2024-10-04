# In[1]:


import sys
import os
import DeepONet_tf as DeepONet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.io as scio
# from deepxde import utils
# import deepxde as dde
# from deepxde.backend import tf
import tensorflow as tf
import timeit
from IPython.display import HTML
import scipy.integrate as sciint
physical_devices = tf.config.list_physical_devices()
for dev in physical_devices:
    print(dev)
print("tf version:", tf.__version__)
# %run ddm-deeponet_adaptive_tf.py 50 0 2 0 0 0
# In[]

prefix_filebase = "./saved_model"
# diff_method = "FD"
str_dN, str_start, str_end = sys.argv[1:4]
str_k, str_c = sys.argv[4:-1]
str_caseID = sys.argv[-1]
# str_N0, str_dN, str_start, str_end="200","200","0","2"
# str_k, str_c = "0.0", "0.0"
# str_caseID = "0"
# In[3]:

fenics_data = scio.loadmat("./TrainingData/FP1Drand_all.mat")

x_grid_file=fenics_data["x_data"].squeeze()
T_data_file=fenics_data["Temp_data"].squeeze()
alpha_data_file=fenics_data["alpha_data"].squeeze()
t_data_file=fenics_data["t_data"].squeeze()
pcs_data_raw=fenics_data["process_condition"].astype(np.float32)
x_grid_raw=x_grid_file.astype(np.float32)
scaler_x=np.max(x_grid_raw)
x_grid=x_grid_raw/scaler_x
Nx=len(x_grid)
# %%
T_data_raw=[]
alpha_data=[]
tx_data=[]
t_data_raw=np.array([0])
t_data=np.array([0])
Tmaxs,Tmins=[],[]
for i in range(len(t_data_file)):
    t_temp=t_data_file[i].squeeze().astype(np.float32)
    if len(t_temp)>len(t_data_raw):
        t_data_raw=t_temp
    
    T_data_raw.append(T_data_file[i].reshape(-1,1).astype(np.float32))
    alpha_data.append(alpha_data_file[i].reshape(-1,1).astype(np.float32))
    Tmaxs.append(np.max(T_data_raw[i]))
    Tmins.append(np.min(T_data_raw[i]))
    if len(alpha_data_file[i])!=len(t_temp):
        raise ValueError("The number of time steps in alpha_data and t_arr_results do not match")   
    if len(T_data_file[i])!=len(t_temp):
        raise ValueError("The number of time steps in T_data and t_arr_results do not match")
scaler_t=np.max(t_data_raw) # ()
t_data=t_data_raw/scaler_t # (Nt,)
x_,t_=np.meshgrid(x_grid,t_data) # (Nt, Nx)
dx=x_grid_raw[1]-x_grid_raw[0]
dt=t_data_raw[1]-t_data_raw[0]
tx_data=(np.concatenate((t_.reshape(-1,1),x_.reshape(-1,1)),axis=1))  # (Nt*Nx,2)
Tmax,Tmin=max(Tmaxs),min(Tmins)
scaler_T,shift_T=(Tmax-Tmin),Tmin
T_data = [(T - shift_T) / scaler_T for T in T_data_raw] #[(Nt*Nx,1)]
Talpha_data=[np.concatenate((T_data[i],alpha_data[i]),axis=1) for i in range(len(T_data))] #[(Nt*Nx,2)],(T, alpha)
Talpha_data_raw=[np.concatenate((T_data_raw[i],alpha_data[i]),axis=1) for i in range(len(T_data_raw))]
T0_min,T0_max=np.min(pcs_data_raw[:,0]),np.max(pcs_data_raw[:,0])
scaler_T0,shift_T0=T0_max-T0_min,T0_min
alpha0_min,alpha0_max=np.min(pcs_data_raw[:,1]),np.max(pcs_data_raw[:,1])
scaler_alpha0,shift_alpha0=alpha0_max-alpha0_min,alpha0_min
pcs_data=(pcs_data_raw-np.array([shift_T0,shift_alpha0]))/np.array([scaler_T0,scaler_alpha0]) # (Nb,2)

# T_data,T_data_raw: list of np with shape(1,nx*nt), shift_T,scaler_T: float
# alpha_data, list of np with shape(1,nx*nt)
# t_data, t_data_raw:  1D np, scaler_t: float
#x_grid, x_grid_raw: 1D np, scaler_x: float
# tx_data: 2D np, meshgrid of x_grid and t_data, [t,x] format
# pcs_data, pcs_data_raw: 2D np, [T0,alpha0] format

# %%
def pad_sequences(arrays, pad_value=0):
    # Determine the maximum number of rows
    max_rows = max(arr.shape[0] for arr in arrays)
    
    # Initialize a list to store the padded arrays
    padded_arrays = []
    
    for arr in arrays:
        # Create a new array with the same number of columns but with max_rows number of rows
        padded_arr = np.full((max_rows, arr.shape[1]), pad_value, dtype=arr.dtype)
        padded_arr[:arr.shape[0], :] = arr
        padded_arrays.append(padded_arr)
    
    return np.stack(padded_arrays)

padding_value=-1000
Talpha_data_padded=pad_sequences(Talpha_data,padding_value)

mask = (Talpha_data_padded != padding_value).astype('float32')
Talpha_data_padded=Talpha_data_padded*mask
# %% 
num_testing=1000
num_train=-num_testing

pcs_train=pcs_data[:num_train]
Talpha_train=Talpha_data_padded[:num_train]
mask_train=mask[:num_train]
Talpha_raw_train=Talpha_data_raw[:num_train]


pcs_testing=pcs_data[-num_testing:]
Talpha_testing=Talpha_data_padded[-num_testing:]
mask_testing=mask[-num_testing:]
Talpha_raw_testing=Talpha_data_raw[-num_testing:]
# %%
x_train = (pcs_train, tx_data)
y_train = Talpha_train
x_testing = (pcs_testing, tx_data)
y_testing = Talpha_testing

data_test = DeepONet.TripleCartesianProd(x_testing, y_testing,aux_data=mask_testing, shuffle=False)

# %%

# %%

data_train = None
numNode=200
model = DeepONet.DeepONetCartesianProd(
    [2, numNode, numNode,numNode,numNode,numNode,2*numNode],
    [2, numNode, numNode,numNode,numNode,numNode,2*numNode],
    {"branch": "relu", "trunk": "tanh"},num_outputs=2,
)
# %%
def L2RelativeError(x_validate, y_validate):
    mask_validate=x_validate[2].astype(bool)

    y_pred_out = model.predict(x_validate)
    y_pred_raw=np.ones_like(y_pred_out)
    y_pred_raw[:,:,0]=y_pred_out[:,:,0]*scaler_T+shift_T
    y_pred_raw[:,:,1]=y_pred_out[:,:,1]
    y_pred = [y_pred_raw[i][mask_validate[i,:,0]] for i in range(y_pred_raw.shape[0])]
    error_s = []
    for i in range(len(y_validate)):
        error_t = np.linalg.norm(y_validate[i][:,0] - y_pred[i][:,0]) / np.linalg.norm(
            y_validate[i][:,0]
        )
        error_a = np.linalg.norm(y_validate[i][:,1] - y_pred[i][:,1]) / np.linalg.norm(
            y_validate[i][:,1]
        )
        error_s_tmp=0.5*(error_t+error_a)
        error_s.append(error_s_tmp)
    return np.stack(error_s), y_pred # error_s:(N,)
    
# %%
def derivative_op(y,x):
    # y shape=(np,2)
    # x shape=(np,2) [t,x ]
    T=y[:,0:1] # (np,1)
    alpha=y[:,1:2] #(np,1)
    T_X=DeepONet.jacobian(T,x) # (np,2)
    T_x=T_X[:,1:2] # (np,1)
    T_t=T_X[:,0:1]*scaler_T/scaler_t # (np,1)
    T_xX=DeepONet.jacobian(T_x,x) # (np,2)
    T_xx=T_xX[:,1:2]*scaler_T/scaler_x**2 # (np,1)
    T_raw=(T*scaler_T+shift_T)
    alpha_t=DeepONet.jacobian(alpha,x)[:,0:1]/scaler_t
    return T_raw,alpha,T_t,T_xx,alpha_t
op = DeepONet.EvaluateDeepONetPDEs(model, derivative_op)

def ErrorMeasure(inputs,op):
    A, Er, m, n, Ca, alpha_c = 8.55e15, 110750, 0.77, 1.72, 14.48, 0.405
    rho, kappa, Cp, H=980.0, 0.152, 1600.5, 350000.0
    R = 8.314

    data_all=op(inputs,stack=False)
    masks_all=inputs[2]
    res_all=[]
    idt1,idt2=30,-5
    idx1,idx2=10,-10
    for data,masks in zip(data_all,masks_all):
        T,alpha,T_t,T_xx,alpha_t=data
        indices=np.where(masks[:,0]==1.0)[0]
        T=T.numpy()[indices].reshape(-1,Nx)[idt1:idt2,idx1:idx2]
        alpha=alpha.numpy()[indices].reshape(-1,Nx)[idt1:idt2,idx1:idx2]
        T_t=T_t.numpy()[indices].reshape(-1,Nx)[idt1:idt2,idx1:idx2]
        T_xx=T_xx.numpy()[indices].reshape(-1,Nx)[idt1:idt2,idx1:idx2]
        alpha_t=alpha_t.numpy()[indices].reshape(-1,Nx)[idt1:idt2,idx1:idx2]
        res1=Cp/H*T_t-kappa/(rho*H)*T_xx-alpha_t
        r1=np.linalg.norm(res1)/np.linalg.norm(alpha_t)
        exp_term = A*np.exp(-Er / (R * T))
        alpha=np.maximum(alpha,0.0)
        alpha=np.minimum(alpha,1.0)
        f_alpha = (1-alpha)**m*alpha**n/(1 + np.exp(Ca * (alpha - alpha_c)))
        rea_rhs =  exp_term * f_alpha
        res2=rea_rhs-alpha_t
        r2=np.linalg.norm(res2)/np.linalg.norm(alpha_t)
        res_all.append(r1*0.5+r2*0.5)
    return np.array(res_all)
 
# %%


def sampling(iter, dN, pre_filebase):
    all_data_idx = np.arange(len(x_train[0]))
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
            (x_train[0][potential_train_data_idx], x_train[1], mask_train[potential_train_data_idx]),op
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
     "adapt_k" + str_k + "c" + str_c + "dN" + str_dN +  "case" + str_caseID
)
filebase = os.path.join(prefix_filebase, project_name)
start_time = timeit.default_timer()

if iter_start != 0:
    pre_filebase = os.path.join(filebase, "iter" + str(iter_start - 1))
    model.load_history(pre_filebase)
    model.load_weights(os.path.join(pre_filebase, "model.ckpt"))

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

    curr_x_train = (x_train[0][currTrainDataIDX], tx_data)
    curr_y_train = y_train[currTrainDataIDX]
    curr_mask_train = mask_train[currTrainDataIDX]
    data_train = DeepONet.TripleCartesianProd(curr_x_train, curr_y_train,aux_data=curr_mask_train, batch_size=64)
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
    
    error_test,_ =L2RelativeError((x_testing[0],x_testing[1],mask_testing),
                                  Talpha_raw_testing)
    np.savetxt(
        os.path.join(current_filebase, "TestL2Error.csv"),
        error_test,
        fmt="%.4e",
        delimiter=",",
    )
    
    curr_y_train_raw = [Talpha_raw_train[i] for i in currTrainDataIDX]
    error_train,_=L2RelativeError((curr_x_train[0],curr_x_train[1],curr_mask_train),
                                  curr_y_train_raw)
    np.savetxt(
        os.path.join(current_filebase, "TrainL2Error.csv"),
        error_train,
        fmt="%.4e",
        delimiter=",",
    )
    
    
    stop_time = timeit.default_timer()
    print("training Run time so far: ", round(stop_time - start_time, 2), "(s)")