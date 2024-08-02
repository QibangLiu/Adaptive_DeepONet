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
    for iter in iters_path:
        if os.path.exists(os.path.join(iter, filename)):
            trainDataIDX = np.genfromtxt(
                os.path.join(iter, "trainDataIDX.csv"), dtype=int, delimiter=","
            )
            num_samples.append(trainDataIDX.shape[0])
            L2error = np.genfromtxt(
                os.path.join(iter, filename), dtype="float32", delimiter=","
            )
            mean_L2.append(np.mean(L2error))
            max_L2.append(np.max(L2error))
            std_L2.append(np.std(L2error))
    return np.array(num_samples).astype('float32'), np.array(mean_L2), np.array(max_L2), np.array(std_L2)


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
caseIDs=[0,1,2]
paras = [("0", "1", "400"), ("1", "1", "400"), ("2", "1", "400"), ("2", "0", "400")]
#paras = [("0", "1", "800"), ("1", "1", "800")]
labels = get_labels(paras)
filebases = []
for para in paras:
    filebase_case = []
    for caseID in caseIDs:
        project_name = (#"PI-"+
            "adapt_k" + para[0] + "c" + para[1] + "dN" + para[2] + "case"+str(caseID)
        )
        filebase_case.append(os.path.join(prefix_filebase, project_name))
    filebases.append(filebase_case)

numS, means, maxs, stds = [], [], [], []
for filebase_case in filebases:
    numS_case, means_case, maxs_case, stds_case = [], [], [], []
    for filebase in filebase_case:
        numS1, mean1, max1, std1 = GetL2Error(filebase)
        numS_case.append(numS1)
        means_case.append(mean1)
        maxs_case.append(max1)
        stds_case.append(std1)
    len_case =max([len(numS1) for numS1 in numS_case])
    for i in range(len(numS_case)):
            numS_case[i]=np.pad(numS_case[i],(0,len_case-len(numS_case[i])),constant_values=np.nan)
            means_case[i]=np.pad(means_case[i],(0,len_case-len(means_case[i])),constant_values=np.nan)
            maxs_case[i]=np.pad(maxs_case[i],(0,len_case-len(maxs_case[i])),constant_values=np.nan)
            stds_case[i]=np.pad(stds_case[i],(0,len_case-len(stds_case[i])),constant_values=np.nan)
    numS_case=np.nanmean(np.array(numS_case),axis=0).astype(int)
    means_case=np.nanmean(np.array(means_case),axis=0)
    maxs_case=np.nanmean(np.array(maxs_case),axis=0)
    stds_case=np.nanmean(np.array(stds_case),axis=0)
    numS.append(numS_case)
    means.append(means_case)
    maxs.append(maxs_case)
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
def load_history(filebase):
    his_file = os.path.join(filebase, "history.json")
    if os.path.exists(his_file):
        with open(his_file, "r") as f:
            fit_history = json.load(f)
    return fit_history


# %%
# fig = plt.figure()
# ax = plt.subplot(1, 1, 1)
# sid=800
# ax.plot(h["loss"][sid:sid+1600], label="loss")
# ax.plot(h["val_loss"][sid:sid+1600], label="val_loss")
# ax.legend()
# ax.set_yscale("log")
# %%
