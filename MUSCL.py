from slopes      import MonCen
from Riemann     import HLL
import numpy as np

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
        """ Compute the Courant condition """
        # signal speed on the grid
        if m.isothermal:
            cs = m.cs
        else:
            cs = m.v_sound()
        iD = 1 / m.D
        vmax = np.empty(m.ndim)
        for idim in range(m.ndim):
            vmax[idim] = (cs + np.abs(m.M[idim])*iD).max()
        min = (m.ds / vmax).min()
        dt = self.Cdt * min                  # time step
        return dt

    def Step(self,m,dt):
        """ Full time update of the MUSCL method
            m: mesh related variables; dt: time step """

        dir = ['x','y','z']

        # 1) Primitive variables D,P,v -- @t and cell centered
        invD = 1 / m.D
        prim = [None]*m.nv
        prim[m.iD] = m.D
        if not m.isothermal:
            prim[m.iE] = m.pressure()
        for idim in range(m.ndim):
            prim[m.iMl[idim]] = m.M[idim] * invD
        if self.verbose: stat(prim[m.iD],'prim(D)')

     
        def dump_stats(name, A):
            print(f"{name:12s} min={A.min():12.6e}   max={A.max():12.6e}")

        #dump_stats("prim[D]",   prim[m.iD])



        # 2) Slopes for primitive variables shape (nv,ndim,n,n,n) -- @t and cell centered
        dprim = np.empty((m.nv,m.ndim)+m.n)
        for iv in range(m.nv):
            self.Slope(prim[iv],dprim[iv])
            for idim in range(m.ndim):
                dprim[iv][idim] *= (1 / m.ds[idim])
        if self.verbose:
            for idim in range(m.ndim):
                stat(dprim[m.iD,idim],'dprim(D)-'+dir[idim])

        # 3) predicted solution @t+dt/2 shape (nv,n,n,n)
        # shorthand for variables and slopes
        predict = np.empty((m.nv,)+m.n)  # predicted solution @t+dt/2
        D = prim[m.iD]; dD = dprim[m.iD] # density
        v = prim[m.iM]; dv = dprim[m.iM] # velocities dv[i,j] = dv_i/dx_j
        if m.isothermal:                 # pressure
            dP = m.cs**2 * dD
        else:
            P = prim[m.iE]; dP = dprim[m.iE]
        
        # div(velocity)
        div_v = np.trace(dv,axis1=0,axis2=1)

        # drhodt = - v.grad(rho) - rho (div.v)
        predict[m.iD] = D - 0.5*dt*(np.sum(v*dD,axis=0) + D*div_v)

        # dP/dt = -v*grad(P) - gamma*P*div(v)
        if not m.isothermal:
            predict[m.iE] = P - 0.5*dt*(np.sum(v*dP,axis=0) + m.gamma*P*div_v)

        # dv_i/dt = - v.grad(v_i) - 1/rho grad(P)
        for i,iM in enumerate(m.iMl):
            predict[iM] = v[i] - 0.5*dt*(np.sum(v*dv[i],axis=0) + invD*dP[i])
        if self.verbose: stat(predict[m.iD],'predict(D)')

        # flux contributions from all directions
        facel, facer = [None]*m.nv, [None]*m.nv  # Left and right face values
        for idim in range(m.ndim):
            # 4) left and right face values (+-ds/2) @t+dt/2 _at_cell_interface_ with shape (2,nv,n,n,n)
            hds = 0.5*m.ds[idim]
            for iv in range(m.nv):
                facel[iv] =         predict[iv] + hds*dprim[iv,idim]               # left face
                facer[iv] = np.roll(predict[iv] - hds*dprim[iv,idim],-1,axis=idim) # right face
            if self.verbose: stat(facel[m.iD],'facel(D)-'+dir[idim])
            if self.verbose: stat(facer[m.iD],'facer(D)-'+dir[idim])

            # 5) Reorder variables with the perpedicular velocity component as first index.
            #    After call to Solver order is restored again using references.
            if m.isothermal:
                fwd = [m.iD] + list(np.roll(m.iMl,-idim))
                bck = [m.iD] + list(np.roll(m.iMl,idim))
            else:
                fwd = [m.iD,m.iE] + list(np.roll(m.iMl,-idim))
                bck = [m.iD,m.iE] + list(np.roll(m.iMl,idim))
            ql = [facel[ifwd] for ifwd in fwd] # make ql, qr a list of references to avoid copying
            qr = [facer[ifwd] for ifwd in fwd]
            f1d = self.Solver(ql,qr,m)
            flux = []
            for ibck in bck: flux.append(f1d[ibck]) # reference back in place

            # 6) Update the conserved variables
            for iv in range(m.nv):
                m.vars[iv] -= (dt/m.ds[idim])*(flux[iv] - np.roll(flux[iv],1,axis=idim))
            if self.verbose: stat(m.vars[m.iD],'updated(D)')
        

        '''
        dump_stats("prim[0]",   prim[0])
        dump_stats("prim[1]",   prim[1])
        dump_stats("prim[2]",   prim[2])
        dump_stats("prim[3]",   prim[3])
        dump_stats("prim[4]",   prim[4])
        dump_stats("dprim[0]",   dprim[0])
        dump_stats("dprim[1]",   dprim[1])
        dump_stats("dprim[2]",   dprim[2])
        dump_stats("dprim[3]",   dprim[3])
        dump_stats("dprim[4]",   dprim[4])        
        dump_stats("predict[0]", predict[0])
        dump_stats("predict[1]", predict[1])
        dump_stats("predict[2]", predict[2])
        dump_stats("predict[3]", predict[3])
        dump_stats("predict[4]", predict[4])
        dump_stats("facer[0]",  facer[0])
        dump_stats("facer[1]",  facer[1])
        dump_stats("facer[2]",  facer[2])
        dump_stats("facer[3]",  facer[3])
        dump_stats("facer[4]",  facer[4])
        
        
        #dump_stats("facer[D]",  facer[m.iD])
        #dump_stats("flux[D]",   flux[m.iD])
        dump_stats("vars[0]",   m.vars[0])
        dump_stats("vars[1]",   m.vars[1])
        dump_stats("vars[2]",   m.vars[2])
        dump_stats("vars[3]",   m.vars[3])
        dump_stats("vars[4]",   m.vars[4])
            
        '''
            
        #raise ValueError("stoooop")
