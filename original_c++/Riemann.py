import numpy as np

def prim2cons(q,m):
    # Convert primitive variables to conserved variables
    # c is a list of np.arrays to avoid copy when possible
    c = [None]*m.nv
    c[m.iD] = q[m.iD]                              # Density
    # compute total energy if not isothermal inline
    if not m.isothermal:
        Etot = q[m.iMl[0]]**2
        for i in range(1,m.ndim):
            Etot +=  q[m.iMl[i]]**2
        Etot *= 0.5*q[m.iD]
        Etot += (1/(m.gamma-1))*q[m.iE]
        c[m.iE] = Etot # Total energy
    for iM in m.iMl: c[iM] = q[m.iD]*q[iM]         # Momentum
    return c

def hydro_flux(c,q,m):
    """ Compute hydro flux for conserved variables """
    F = [None]*m.nv
    v = q[m.iMl[0]]                                # Normal velocity
    F[m.iD] = c[m.iMl[0]]                          # Density flux = D v = first momentum component
    for iM in m.iMl: F[iM] = c[iM]*v               # Velocity part of momentum flux F_v[i] = D v[i] v_norm
    if not m.isothermal:
        F[m.iE] = (c[m.iE] + q[m.iE])*v            # Energy flux = (E + P) v
        F[m.iMl[0]] += q[m.iE]                     # Add pressure to normal part of momentum flux
    else:
        F[m.iMl[0]] += m.cs**2*c[m.iD]             # Add pressure to normal part of momentum flux
    return F

def HLL(ql,qr,m):
    """ Solve for the HLL flux given two HD state vectors (left and right of the interface)
        q = [left,right][Density, Pressure, vO, v1, v2], where vO is normal to the interface
        between left and right state, and v1, v2 are parallel to the interface """
    cl = prim2cons(ql,m)                # Convert to conserved variables
    cr = prim2cons(qr,m)
    Fl = hydro_flux(cl,ql,m)            # Compute hydro flux
    Fr = hydro_flux(cr,qr,m)
    if not m.isothermal:                # maximum signal speed
        c2_left  = m.gamma*ql[m.iE]/ql[m.iD]
        c2_right = m.gamma*qr[m.iE]/qr[m.iD]
        c_max = np.sqrt(np.maximum(c2_left,c2_right))
    else:
        c_max = m.cs
    # maximum wave speeds to the left and right (guaranteed to have correct sign)
    vl = ql[m.iMl[0]]; vr = qr[m.iMl[0]]  # normal velocity
    SL = (np.minimum(np.minimum(vl,vr)-c_max,0)) # <= 0.
    SR = (np.maximum(np.maximum(vl,vr)+c_max,0)) # >= 0.

    # HLL flux based on wavespeeds. If SL < 0 and SR > 0 (sub-sonic) then mix appropriately. 
    iSRL = 1 / (SR - SL)
    SRL  = SR*SL
    flux = np.empty((m.nv,)+m.n)
    for iv in range(m.nv):
        flux[iv] = (SR*Fl[iv] - SL*Fr[iv] + SRL*(cr[iv] - cl[iv]))*iSRL
    return flux
