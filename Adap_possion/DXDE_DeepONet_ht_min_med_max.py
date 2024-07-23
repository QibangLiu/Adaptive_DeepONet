#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from deepxde import utils
import deepxde as dde
from deepxde.backend import tf
#dde.backend.set_default_backend("tensorflow")

# In[]
filebase='/scratch/bbpq/qibang/repository/Adap_data_driven_possion/saved_model/AllData'
if not os.path.exists(filebase):
    os.makedirs(filebase)


# In[3]:

Nx = 128
Ny = 128
m = Nx * Ny
###  N number of samples 1000
###  m number of points 40
###  P number of output sensors (coordinates) 1600
### x_train is a tuple of u0(N,m) and output sensores, all coordinates xy_train_testing(P,2)
### y_train is target solutions (our s) u(N,P)

seed = 123 
tf.keras.backend.clear_session()
#tf.keras.utils.set_random_seed(seed)

#dde.config.set_default_float("float64")

u0_train = np.load('DATA3_5000/data_u0_train_ht.npy').astype(np.float32)

u0_testing = np.load('data_u0_testing_ht.npy').astype(np.float32)

#s_train = np.load('s_train_10K.npy').astype(np.float32)
s_train = np.load('DATA3_5000/data_s_train_ht.npy').astype(np.float32)

s_testing = np.load('data_s_testing_ht.npy').astype(np.float32)
xy_train_testing = np.load('xy_train_test_ht.npy').astype(np.float32)

print('u0_train.shape = ', u0_train.shape)
print('type of u0_train = ', type(u0_train))
print('u0_testing.shape = ',u0_testing.shape)
print('s_train.shape = ',s_train.shape)
print('s_testing.shape = ',s_testing.shape)
print('xy_train_testing.shape', xy_train_testing.shape)
# %%
x_train = (u0_train, xy_train_testing)
y_train = s_train 
x_test = (u0_testing[:3], xy_train_testing)
y_test = s_testing[:3]

# %%
#x_train, y_train = get_data("train_IC2.npz")
#x_test, y_test = get_data("test_IC2.npz")


data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)
#print('x_train.shape = ', np.array(x_train).shape)
#print("type of x_train", type(x_train))
#print('x_train = ', x_train)
#print('y_train.shape = ', y_train.shape)
#print('x_test.shape = ', x_test.shape)
#print('y_test.shape = ', y_test.shape)
#print("type of data", type(data))
# %%
net = dde.maps.DeepONetCartesianProd(
    [m, 100, 100, 100, 100, 100, 100], [2, 100, 100, 100, 100, 100, 100], "relu", "Glorot normal"
)



model = dde.Model(data, net)
model.compile(
    "adam",
    lr=1e-3,
    decay=("inverse time", 1, 1e-4),
    metrics=["mean l2 relative error"],
)
# IC1
# losshistory, train_state = model.train(epochs=100000, batch_size=None)
# IC2

losshistory, train_state = model.train(epochs=200000, batch_size=64,model_save_path=filebase) # QB:epochs=200000

y_pred = model.predict(data.test_x)
print('y_pred.shape =', y_pred.shape)
##np.savetxt("y_pred_deeponet.dat", y_pred[0].reshape(nt, nx))
##np.savetxt("y_true_deeponet.dat", data.test_y[0].reshape(nt, nx))
##np.savetxt("y_error_deeponet.dat", (y_pred[0] - data.test_y[0]).reshape(nt, nx))

# %%

error_s = []
for i in range(50):
    error_s_tmp = np.linalg.norm(y_test[i] - y_pred[i]) / np.linalg.norm(y_test[i])
    error_s.append(error_s_tmp)
error_s = np.stack(error_s)
print("error_s = ", error_s)


# %%

#### Plotting Results 
# %%
import matplotlib.pyplot as plt
import pylab as py
# Defining custom plotting functions
def my_contourf(x,y,F,ttl):
    cnt = py.contourf(x,y,F,12,cmap = 'jet')
    py.colorbar()
    py.xlabel('x'); py.ylabel('y'); py.title(ttl)
    return 0

rand_num = np.random.randint(Nx)
print(rand_num)
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)


min_index = np.argmin(error_s)
max_index = np.argmax(error_s)
median_index = np.median(error_s).astype(int)

# Print the indexes
print("Index for minimum element:", min_index)
print("Index for maximum element:", max_index)
print("Index for median element:", median_index)


min_median_max_index = np.array([min_index, median_index, max_index])


for index in min_median_max_index:

    u0_testing_nx_ny = u0_testing[index].reshape(Nx,Ny)
    s_testing_nx_ny = y_test[index].reshape(Nx,Ny)
    s_pred_nx_ny = y_pred[index].reshape(Nx,Ny)

    fig = plt.figure(figsize=(18,5))
    plt.subplot(1,3,1)
    #py.figure(figsize = (14,7))
    my_contourf(x,y,u0_testing_nx_ny.T,r'Source Distrubution')
    plt.tight_layout()
    plt.subplot(1,3,2)
    #py.figure(figsize = (14,7))
    my_contourf(x,y,s_testing_nx_ny,r'Reference Solution')
    plt.tight_layout()
    plt.subplot(1,3,3)
    #py.figure(figsize = (14,7))
    my_contourf(x,y,s_pred_nx_ny,r'Predicted Solution')
    plt.tight_layout()
    if index == min_index:
        plt.savefig("temperature_min_error_5000_3.jpg", dpi=300)
    if index == median_index:
        plt.savefig("temperature_median_error_5000_3.jpg", dpi=300)
    if index == max_index:
        plt.savefig("temperature_max_error_5000_3.jpg", dpi=300)
    #plt.savefig("temperature_sample{}_5000_3.jpg".format(index), dpi=300)
    plt.show()





# %%
