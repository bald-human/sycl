import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time
import os

# modules that are specific to the setup
# see supporting files, if needed
from mesh        import Mesh
from slopes      import MonCen, MonCen2
from Riemann     import HLL

import muscl_step

if True:
    from MUSCL_done import MUSCL
else:
    from MUSCL_pybind       import MUSCL



# allow 40 figures to be opened
mpl.rcParams['figure.max_open_warning'] = 40

# stop the code if there are floating point errors
np.seterr(over='raise',invalid='raise',under='ignore');

n = (64,64,64) # resolution of the mesh
center = [0,0,0] # center of the mesh
size = [2,2,2]   # physical size of the mesh
cs = 1
gamma = 1.2
Cdt = 0.2

# create the mesh with resolution "n"
m = Mesh(n=n, center=center, size=size, gamma=gamma, cs=cs)

# create a solver object for advancing hydrodynamics
muscl = MUSCL(Cdt=Cdt, Solver=HLL, Slope=MonCen)

# initial conditions
d0 = 1     # density
u0 = [0,0,0] # velocity
p0 = 1     # pressure

m.D[...] = d0                # density
for idim in range(m.ndim):
    m.M[idim] = d0*u0[idim]  # momentum

if not m.isothermal:         # total energy
    m.E[...] = p0 / (m.gamma - 1) + 0.5*d0*(u0[0]**2 + u0[1]**2)

# adding a blast wave
power = 2
w = 3
e0 = 1e3
d0 = 2
blast = np.exp(-np.abs(m.r/(w*m.ds[0]))**power)
blast_int = np.sum(blast * m.ds.prod()) # unormalised total energy of the blast wave
m.E[...] = m.E[...] + e0 * blast / blast_int
m.D[...] = m.D[...] + d0 * blast / blast_int

dtmax = 1e-2
tend = 0.02
max_step = 100
if m.ndim == 3:
    mid = m.n[2]//2 # midplane

used = 0
last = 0
stepfreq = 10

counter = 0
while m.t < tend and m.step < max_step:
    # to make one step environment to implement and test pybind inmplementations
    if counter == 8:
        print("end of simulation")
        break
    print(m.step)
    now = time()
    dt_c = muscl_step.Courant(m.D, m.E, m.M, m.ds, m.n, m.isothermal, m.ndim, m.cs, m.gamma, Cdt)
    dt = min(dt_c,dtmax)
    if m.t + dt > tend:
        dt = tend - m.t
    muscl_step.Calc_Step(m.vars, m.D, m.E, m.M, m.ds, m.n, m.nv, m.ndim, m.iD, m.iE, m.iM, dt, Cdt, m.cs, m.gamma, m.isothermal)
    m.t = m.t + dt
    m.step = m.step + 1
    used += time()-now
    counter += 1

    if m.step % stepfreq == 0:
        musc = 1e6*(used - last) / (np.prod(m.n) * stepfreq) # micro-seconds per cell update
        print("step = {}, t = {:4g}, dt = {:4g} mus/pt = {:3f}".format(m.step,m.t,dt,musc))
        last = used

    def are_np_same(array_a, array_b):
        if not (array_a.shape == array_b.shape):
            raise RuntimeError("shapes are misaligned")
        
        if np.all(array_a == array_b):
            print("array_a and array_b are the same")
            return True
        else:
            print("arrays are not the same")
            return False
#
#output_dir = r"C:\Users\T-Bone\python_work\sycl-hydro\pybinding"
#os.makedirs(output_dir, exist_ok=True)  # ensure the directory exists
#output_path = os.path.join(output_dir, "vars_c.npy")
#np.save(output_path, m.vars)



print("Total time = {:3f}".format(used))
musc = 1e6*used / (np.prod(m.n) * m.step) # micro-seconds per cell update
print("Average micro-seconds per cell update = {:3f}".format(musc))