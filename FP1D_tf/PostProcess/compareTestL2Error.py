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
def GetL2Error(filebase,filename="TestL2Error.csv"):
    iters = natsorted(next(os.walk(filebase))[1])
    iters_path = [os.path.join(filebase, iter) for iter in iters]
    num_samples = []
    mean_L2 = []
    max_L2 = []
    std_L2 = []
    trainDataIDXs=[]
    for iter in iters_path:
        if os.path.exists(os.path.join(iter, filename)):
            trainDataIDX = np.genfromtxt(
                os.path.join(iter, "trainDataIDX.csv"), dtype=int, delimiter=","
            )
            num_samples.append(trainDataIDX.shape[0])
            trainDataIDXs.append(trainDataIDX)
            L2error = np.genfromtxt(
                os.path.join(iter, filename), dtype="float32", delimiter=","
            )
            mean_L2.append(np.mean(L2error[522:525]))
            max_L2.append(np.max(L2error[522:525]))
            print(filebase,np.argmax(L2error))
            std_L2.append(np.std(L2error[522:525]))
    return num_samples, np.array(mean_L2), np.array(max_L2), np.array(std_L2),trainDataIDXs


def get_labels(paras):
    labels = []
    for para in paras:
        if para[0] == "0":
            label = "Random"
        else:
            label = "k=" + para[0] + ", c=" + para[1]
        labels.append(label)
    return labels


# %%


prefix_filebase = "../saved_model/"
caseID = 0
dN='50'
paras = [("0", "1", dN), ("1", "1", dN), ("2", "1", dN), ("2", "0", dN)]
labels = get_labels(paras)
filebases = []
for para in paras:
    project_name = (
        "adapt_k" + para[0] + "c" + para[1] + "dN" + para[2] + "case"+str(caseID)
    )
    filebases.append(os.path.join(prefix_filebase, project_name))


numS, means, maxs, stds = [], [], [], []
trainDataIDXs=[]
for filebase in filebases:
    numS1, mean1, max1, std1,idxs = GetL2Error(filebase)
    numS.append(numS1)
    means.append(mean1)
    maxs.append(max1)
    stds.append(std1)
    trainDataIDXs.append(idxs)
# %%
fig = plt.figure(figsize=(12, 4))
ax = plt.subplot(1, 2, 1)
for num, meanv, label in zip(numS, means, labels):
    ax.plot(num, (meanv) * 100, "-o", label=label)
ax.set_xlabel("num of samples")
ax.set_ylabel("Mean L2 error [%]")
ax.legend()
ax.set_yscale("log")
ax = plt.subplot(1, 2, 2)
# eb=ax.errorbar(numS1,(mean1)*100,yerr=(std1)*100,lolims=True,fmt='--o',label='random')
# eb[-1][0].set_linestyle('--')
# eb=ax.errorbar(numS2,(mean2)*100,yerr=(std2)*100,lolims=True,fmt='--o',label='adaptive')
# eb[-1][0].set_linestyle('--')
for num, maxv, label in zip(numS, maxs, labels):
    ax.plot(num, (maxv) * 100, "-o", label=label)
ax.set_xlabel("num of samples")
ax.set_ylabel("Max L2 error [%] ")
ax.legend()
ax.set_yscale("log")


# %%

fenics_data = scio.loadmat("../TrainingData/FP1D_all.mat")
pcs_data_raw=fenics_data["process_condition"].astype(np.float32)
# %%
pcs=[]
for idxs in trainDataIDXs:
    pcs.append(pcs_data_raw[idxs[-7],:])
fig = plt.figure(figsize=(8, 8))
for i in range(1,5):
    ax=plt.subplot(2,2,i)
    ax.scatter(pcs[i-1][:,0],pcs[i-1][:,1],c='r',s=0.5,label='Adaptive')
    ax.set_title(labels[i-1])
# %%
