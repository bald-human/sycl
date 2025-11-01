from slopes      import MonCen
from Riemann     import HLL
import numpy as np

import muscl_step

def stat(f,name=''):
    """ Print statistics on a single variable """
    print(name+" min={:6.3g} max={:6.3g} mean={:6.3g} std={:6.3g}".format(f.min(),f.max(),f.mean(),f.std()))

class MUSCL():
    """ The Monotonic Upwind Scheme for Conservation Laws has these steps:
        1. Compute primitive variables
        2. Compute slopes of primitive variables
        3. Compute predicted states at time t+dt/2 
        Repeat 4+5+6 for all coordinate directions:
        4. Compute left and right face values
        5. Compute fluxes using Riemann solver
        6. Update the conserved variables
    """
    def __init__(self, Cdt=0.2, Solver=HLL, Slope=MonCen):
        """ Initialize a MUSCL solver """
        self.verbose  = False # set to true for verbose output
        self.Cdt      = Cdt
        self.Solver   = Solver
        self.Slope    = Slope

    def Courant(self,m):
        print("courant")
        """ Compute the Courant condition """
        # signal speed on the grid
        dt = muscl_step.Courant(m.D, m.M, m.ds, m.n, m.v_sound, m.isothermal, m.ndim, m.cs, self.Cdt)
        return dt

    def Step(self,m,dt):
        def are_np_same(array_a, array_b):
            if not (array_a.shape == array_b.shape):
                raise RuntimeError("shapes are misaligned")
            
            if np.all(array_a == array_b):
                print("array_a and array_b are the same")
                return True
            else:
                print("arrays are not the same")
                return False
        
        """ 
        Full time update of the MUSCLmethod
        m: mesh related variables; dt: time step 
        """
        print("step")

        ## 1) Primitive variables D,P,v -- @t and cell centered
        #prim = step_one.make_prim(m.D, m.M, m.n, m.ndim, m.nv, m.iD, m.iE, m.iMl, m.pressure, m.isothermal)
        #invD = step_one.invD(m.D, m.n)
        #print("here")
        #
        ## 2) Slopes for primitive variables shape (nv,ndim,n,n,n) -- @t and cell centered
        #dprim = step_two.step_two(prim, m.n, m.nv, m.ndim, m.ds)
        #
        ## 3) predicted solution @t+dt/2 shape (nv,n,n,n)
        ## shorthand for variables and slopes
        #"""have to test isothermal case"""
        #predict = step_two.step_three(prim, dprim, invD, m.n, m.nv, m.ndim, m.iM, m.iD, m.iE, m.cs, m.isothermal, dt, m.gamma)
        #
        #vars = calc_flux.flux(m.vars ,predict, dprim, m.ds, m.n, m.nv, m.ndim, m.iD, m.iE, m.iM, m.isothermal, m.cs, m.gamma, dt, solver.HLL)
        vars = muscl_step.Calc_step(m.vars, m.D, m.E, m.M, m.ds, m.n, m.nv, m.ndim, m.iD, m.iE, m.iM, dt, self.Cdt, m.cs, m.gamma, m.isothermal)
        #are_np_same(vars, vars_2)
        
        # Check for zero or negative densities
        if np.any(m.vars[m.iD] <= 0):
            print("⚠️ Warning: Non-positive density in m.vars detected!")
            print("Min density:", m.vars[m.iD].min())
            #print("Indices where D <= 0:", np.argwhere(m.vars[m.iD] <= 0))
        print(vars.shape)
        print("bye,bye")