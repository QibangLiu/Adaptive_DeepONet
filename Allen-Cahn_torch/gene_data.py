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
    return srf.field/10

u_n=Function(V)

def femsim(u0):
    #u_n.interpolate(lambda x: -0.5*np.sin(np.pi*x[0]))
    u_n.x.array[sortidx]=u0
    u=Function(V)
    v=TestFunction(V)
    alpha=0.01
    u_x=u.dx(0)
    u.x.array[:]=u_n.x.array[:]
    F=(u-u_n)*v*ufl.dx+alpha*dt*ufl.inner(ufl.grad(v), ufl.grad(u))*ufl.dx\
        +5.0*dt*u**3*v*ufl.dx-5.0*dt*u*v*ufl.dx
    bc_l = dfx.fem.dirichletbc(u0[0], dofs_L,V)
    bc_r = dfx.fem.dirichletbc(u0[-1], dofs_R,V)
    bcs = [bc_l,bc_r]
    
    t = 0
    solution=[]
    u_gather = comm.gather(u.x.array[:], root=0)
    u_gather=comm.bcast(u_gather, root=0)
    u_gather=np.concatenate(u_gather).reshape(-1)
    solution.append(u_gather)
    tarr=[t]
    for n in range(nSteps):
        # if n>0:
        #     bcs = []
        problem = dfx.fem.petsc.NonlinearProblem(F, u,bcs)
        # update time
        t += dt

        problem = dfx.fem.petsc.NonlinearProblem(F, u,bcs)
        #dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        solver = NewtonSolver(comm, problem)
        # Set Newton solver options
        #solver.convergence_criterion = "incremental"
        solver.error_on_nonconvergence = False
        solver.convergence_criterion = "incremental" #"residual" #"incremental"
        solver.atol = 1e-14
        solver.rtol = 1e-14
        solver.max_it = 1000
        solver.report = True
        # We can customize the linear solver used inside the NewtonSolver by
        # modifying the PETSc options
        ksp = solver.krylov_solver
        opts = PETSc.Options()  # type: ignore
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "preonly"
        opts[f"{option_prefix}pc_type"] = "lu"
        opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        ksp.setFromOptions()
        r = solver.solve(u)
        if not r[1]:
            print("diverge break!!!")
            break
        if n%10==0:
            print(f"Step {n}, time {t}, residual {r}")
            u_gather = comm.gather(u.x.array[:], root=0)
            u_gather=comm.bcast(u_gather, root=0)
            u_gather=np.concatenate(u_gather).reshape(-1)
            solution.append(u_gather[sortidx])
            tarr.append(t)
        # update previous solution
        u_n.x.array[:] = u.x.array[:]
    tarr=np.array(tarr)
    solution=np.array(solution)
    return solution,tarr,r
# %%
print("start the simulation")
solutions,u0s=[],[]
for i in range(5000):
    if i%100==0:
        print("i: ",i)
    u0=gaussian_covariance_field(len(u_n.x.array[:]))
    solution,tarr,r=femsim(u0)
    if r[1]:
        solutions.append(solution)
        u0s.append(u0)

filebase = ['./TrainingData', 'Allen-Cahn_gauss_cov'+sys.argv[1]]
os.makedirs(filebase[0], exist_ok=True)
training_data = {'x_grid':coord_local[sortidx,0],'t_grid':tarr,'u0s': u0s, 'solutions': solutions}
femFile = os.path.join(filebase[0], filebase[1]+'.mat')
scio.savemat(femFile, training_data)

# %%
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
