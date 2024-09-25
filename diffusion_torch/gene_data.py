# %%
import sys
import numpy as np
import scipy.fftpack

import numpy as np
import matplotlib.pyplot as plt
#from scipy.ndimage import gaussian_filter
import scipy.io as scio
from matplotlib import rcParams
import time
import numpy as np
import os


import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx.nls.petsc import NewtonSolver
import ufl
from ufl import TestFunction, TrialFunction, FiniteElement
from dolfinx.fem import (functionspace, Function, dirichletbc,
                         Expression, Constant, locate_dofs_geometrical)
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

import dolfinx as dfx
import basix
import time
from petsc4py import PETSc
import gstools as gs
# %%
# %%
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
numProc = comm.Get_size()
if comm_rank == 0:
    print("number of cores: ", numProc)
    print('current_time (done): ',  time.ctime())

dfx.log.set_log_level(dfx.log.LogLevel.ERROR)  # OFF,INFO,WARNING,ERROR

# %%
nx = 127
xl, xr = (0.0, 1.0)
T = 1.0 
nSteps = 1000
dt = T/nSteps
order = 2

msh =dfx.mesh.create_interval(comm, nx, points=[(xl), (xr)])
V = functionspace(msh, ("Lagrange", order))
coord_local = V.tabulate_dof_coordinates()[:,:msh.topology.dim]
sortidx = np.argsort(coord_local[:,0])

# boundary conditions
def bc_left(x):
    return np.isclose(x[0], xl)
def bc_right(x):
    return np.isclose(x[0], xr)
dofs_L = dfx.fem.locate_dofs_geometrical(V, bc_left)
dofs_R = dfx.fem.locate_dofs_geometrical(V, bc_right)


# initial conditions    
def gaussian_covariance_field(size=128):
    x = np.linspace(0,1,size)
    val=np.random.randint(1,50)
    model = gs.Gaussian(dim=1, var=val, len_scale=0.2)
    srf = gs.SRF(model)
    srf((x), mesh_type='structured')
    return srf.field

u_n=Function(V)
u_=Function(V)
v=TestFunction(V)
u=TrialFunction(V)
source_term = Function(V)
def femsim(u0):
    #u_n.interpolate(lambda x: -0.5*np.sin(np.pi*x[0]))
    u_n.x.array[sortidx]=0
    u_.x.array[sortidx]=0
    source_term.x.array[sortidx]=u0
    alpha=0.01
    diff_eq=(u-u_n)*v*ufl.dx+alpha*dt*ufl.inner(ufl.grad(v), ufl.grad(u))*ufl.dx\
        -dt*v*(source_term-u)*ufl.dx
    a_diff, L_diff = dfx.fem.form(ufl.lhs(diff_eq)), dfx.fem.form(ufl.rhs(diff_eq))
    bilinear_form = dfx.fem.form(a_diff)
    linear_form = dfx.fem.form(L_diff)    
    
  
    
    bc_l = dfx.fem.dirichletbc(u0[0], dofs_L,V)
    bc_r = dfx.fem.dirichletbc(u0[-1], dofs_R,V)
    bcs = []
    
    Amat = assemble_matrix(bilinear_form, bcs=[])
    Amat.assemble()
    solver = PETSc.KSP().create(msh.comm)  # Krylov methods.
    solver.setOperators(Amat)

    opts = PETSc.Options()
    opts["ksp_type"] = "preonly" # cg, 
    opts["pc_type"] = "lu"  # gamg, lu
    # opts.setValue('ksp_atol', 1E-10)  #default 1e-50
    #opts.setValue('ksp_rtol', 1E-14)  # default 1e-5
    solver.setFromOptions()
    #solver.setInitialGuessNonzero(True)
    b = create_vector(linear_form)
    
    t = 0
    solution=[]
    u_gather = comm.gather(u_.x.array[:], root=0)
    u_gather=comm.bcast(u_gather, root=0)
    u_gather=np.concatenate(u_gather).reshape(-1)
    solution.append(u_gather)
    tarr=[t]
    for n in range(nSteps):
        # if n>0:
        #     bcs = []
        # update time
        t += dt
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form)
        apply_lifting(b, [bilinear_form], [[]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [])
        
        solver.solve(b, u_.vector)
        u_.x.scatter_forward()
        if n%10==0:
            print(f"Step {n}, time {t}")
            u_gather = comm.gather(u_.x.array[:], root=0)
            u_gather=comm.bcast(u_gather, root=0)
            u_gather=np.concatenate(u_gather).reshape(-1)
            solution.append(u_gather[sortidx])
            tarr.append(t)
        # update previous solution
        u_n.x.array[:] = u_.x.array[:]
    tarr=np.array(tarr)
    solution=np.array(solution)
    return solution,tarr
# %%
print("start the simulation")
solutions,u0s=[],[]
for i in range(5000):
    if i%100==0:
        print("i: ",i)
    u0=gaussian_covariance_field(len(u_n.x.array[:]))
    solution,tarr=femsim(u0)

    solutions.append(solution)
    u0s.append(u0)

filebase = ['./TrainingData', 'Allen-Cahn_gauss_cov'+sys.argv[1]]
os.makedirs(filebase[0], exist_ok=True)
training_data = {'x_grid':coord_local[sortidx,0],'t_grid':tarr,'u0s': u0s, 'solutions': solutions}
femFile = os.path.join(filebase[0], filebase[1]+'.mat')
scio.savemat(femFile, training_data)

# # %%
# plt.contourf(coord_local[:,0],np.linspace(0,1,len(solution)), solution,levels=100)
# # %%
# fig=plt.figure()
# ax=plt.subplot(1,1,1)   
# Nt=len(tarr)
# num_curv = 8
# step = (Nt - 0) / (num_curv + 1)
# curv = [int(0 + (i + 1) * step) for i in range(num_curv)]
# curv[-1] = Nt - 1
# x_grid=coord_local[sortidx,0]
# for j, c in enumerate(curv):
#         ax.plot(x_grid, solution[c, :], label="t=%.2f" % tarr[c])
# ax.legend()

# %%

# %%
