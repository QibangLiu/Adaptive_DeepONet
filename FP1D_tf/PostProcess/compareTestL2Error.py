# In[]
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from natsort import natsorted
import os
import json

# In[]
def GetL2Error(filebase,epsilon=0.0,filename="TestL2Error.csv"):
    iters = natsorted(next(os.walk(filebase))[1])
    iters_path = [os.path.join(filebase, iter) for iter in iters]
    num_samples = []
    mean_L2 = []
    max_L2 = []
    std_L2 = []
    L2_errors = []
    num_outliers=[]
    trainDataIDXs=[]
    for iter in iters_path:
        if os.path.exists(os.path.join(iter, filename)):
            trainDataIDX = np.genfromtxt(
                os.path.join(iter, "trainDataIDX.csv"), dtype=int, delimiter=","
            )
            num_samples.append(trainDataIDX.shape[0])
            trainDataIDXs.append(trainDataIDX)
            L2error1 = np.genfromtxt(
                os.path.join(iter, filename), dtype="float32", delimiter=","
            )
            L2error=L2error1[L2error1 > epsilon]
            if len(L2error)==0:
                L2error=np.array([])
            mean_L2.append(np.sum(L2error))
            max_L2.append(np.max(L2error1))
            std_L2.append(np.std(L2error))
            L2_errors.append(L2error1)
            num_outliers.append(len(L2error))
            
    return num_samples,num_outliers, np.array(mean_L2), np.array(max_L2), np.array(std_L2),L2_errors,trainDataIDXs


def get_labels(paras):
    labels = []
    for para in paras:
        if para[0] == "0":
            label = "Random"
        else:
            label = "k=" + para[0] #+ ", c=" + para[1]
        labels.append(label)
    return labels


# %%
prefix_filebase = "../saved_model/"
caseID = 0
dN="50"
paras = [("0", "0", dN), ("1", "0", dN), ("2", "0", dN),("4", "0", dN)]
#paras = [("0", "1", "800"), ("1", "1", "800")]
labels = get_labels(paras)
filebases = []
for para in paras:
    project_name = (#"PI-"+
        "adapt_k" + para[0] + "c" + para[1] + "dN" + para[2] + "case"+str(caseID)
    )
    filebases.append(os.path.join(prefix_filebase, project_name))

epsilon=0.03
numS,num_outliers, means, maxs, stds = [], [], [], [], []
L2_errors_all=[]
trainDataIDXs=[]
for filebase in filebases:
    numS1,n_outliers, mean1, max1, std1,L2error,trainidxs = GetL2Error(filebase,epsilon,"TestL2Error.csv")
    numS.append(numS1)
    num_outliers.append(n_outliers)
    means.append(mean1)
    maxs.append(max1)
    stds.append(std1)
    L2_errors_all.append(L2error)
    trainDataIDXs.append(trainidxs)
# %%
fig = plt.figure(figsize=(18, 4))
ax = plt.subplot(1, 3, 1)
for num, meanv, label in zip(numS, means, labels):
    ax.plot(num, (meanv) * 100, "-o", label=label)
ax.set_xlabel("Num of samples")
ax.set_ylabel("Sum of outliers' $L_2$ relative error [%]")
ax.set_title(f"$L_2 > {epsilon*100}\%$" )
ax.legend()
ax.set_yscale("log")
ax = plt.subplot(1, 3, 2)
for num, n_outliers, label in zip(numS, num_outliers, labels):
    ax.plot(num, n_outliers, "-o", label=label)
ax.set_xlabel("Num of samples")
ax.set_ylabel("Num of outliers ")
ax.set_yscale("log")
ax.legend()
ax = plt.subplot(1, 3, 3)
for num, maxv, label in zip(numS, maxs, labels):
    ax.plot(num, (maxv) * 100, "-o", label=label)
ax.set_xlabel("Num of samples")
ax.set_ylabel("Max $L_2$ relative error [%] ")
ax.legend()
ax.set_yscale("log")

# %%

fenics_data = scio.loadmat("../TrainingData/FP1D_all.mat")
pcs_data_raw=fenics_data["process_condition"].astype(np.float32)
# %%
pcs=[]
for idxs in trainDataIDXs:
    pcs.append(pcs_data_raw[idxs[-1],:])
fig = plt.figure(figsize=(8, 8))
for i in range(1,5):
    ax=plt.subplot(2,2,i)
    ax.scatter(pcs[i-1][:,0],pcs[i-1][:,1],c='r',s=0.5,label='Adaptive')
    ax.set_title(labels[i-1])
# %%
adap_algo=0
num=numS[adap_algo]
L2_errors=L2_errors_all[adap_algo]

fig=plt.figure(figsize=(25,30))
nc,nr=5,6
for i,l2error in enumerate(L2_errors):
    ax=plt.subplot(nc,nr,i+1)
    ax.hist(L2_errors[i],bins=20)
    ax.set_xlabel("L2 error")
    #ax.set_xticks(np.arange(0,0.24,0.04))
    ax.set_ylabel("Frequency")
    ax.set_title("num of training samples: "+str(num[i]))
# %%
