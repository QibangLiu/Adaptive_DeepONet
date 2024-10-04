# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# %%

# Define the mean vector and covariance matrix
sigma=0.2

# Create a grid of (x, y) points
x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
def f(x, y):
    return (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-(x**2+y**2) / (2 * sigma**2))

z = f(x, y)

X, Y = np.meshgrid(np.linspace(-1, 1, 5000), np.linspace(-1, 1, 5000))
XY = np.column_stack((X.ravel(), Y.ravel()))

LR=f(XY[:, 0], XY[:, 1])

# %%
K=[0,1,2,3,4]
IDs=[]
for k in K:
    probility = np.power(LR, k) / np.power(LR, k).mean()
    probility_normalized = probility / np.sum(probility)
    selectIDs= np.random.choice(
                a=len(LR),
                size=1000,
                replace=False,
                p=probility_normalized,
            )
    IDs.append(selectIDs)

# %%
nr,nc=2,3
fig=plt.figure(figsize=(nc*5,nr*5))
ax=plt.subplot(nr,nc,1)
c=ax.contourf(x, y, z, cmap='viridis')
plt.colorbar(c, ax=ax)
ax.set_title(r"Residual Error $L_R$")
ax.set_xlabel('x')
ax.set_ylabel('y')


for i,k in enumerate(K):
    ax=plt.subplot(nr,nc,i+2)
    ax.scatter(XY[IDs[i],0],XY[IDs[i],1],s=1)
    ax.set_title(r"$k=%d$"%k)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
plt.tight_layout()
# %%
