import numpy as np
from numba import jit

def deriv(f):
    """ Central slope (e.g. derivative*dx) """
    slopes=np.empty((f.ndim,)+f.shape)
    for i in range(f.ndim):
        slopes[i]=0.5*(np.roll(f,-1,axis=i) - np.roll(f,1,axis=i))
    if f.ndim==1: slopes.squeeze()
    return slopes

@jit(nopython=True,cache=True)
def MonCen(f,slopes):
    """ Monotonized central slope limiter """
    s = f.shape
    if f.ndim==1:
        ls = 0.5*(f - np.roll(f,1))
        for ix in range(s[0]-1):
            w = ls[ix]*ls[ix+1]
            slopes[ix] = 2*w/(ls[ix]+ls[ix+1]) if w>0 else 0
        w = ls[-1]*ls[0]
        slopes[-1] = 2*w/(ls[-1]+ls[0]) if w>0 else 0

    elif f.ndim==2:
        # x-axis
        for ix in range(s[0]):
            ixp = (ix+1) % s[0]
            for iy in range(s[1]):
                iyp = (iy+1) % s[1]
                lsx = 0.5*(f[ix,iy] - f[ix-1,iy])
                rsx = 0.5*(f[ixp,iy] - f[ix,iy])
                wx = lsx*rsx
                slopes[0,ix,iy] = 2*wx/(lsx+rsx) if wx>0 else 0
                lsy = 0.5*(f[ix,iy] - f[ix,iy-1])
                rsy = 0.5*(f[iyp,iy] - f[ix,iy])
                wy = lsy*rsy
                slopes[1,ix,iy] = 2*wy/(lsy+rsy) if wy>0 else 0

    elif f.ndim==3:
        for ix in range(s[0]):
            ixp = (ix+1) % s[0]
            for iy in range(s[1]):
                iyp = (iy+1) % s[1]
                for iz in range(s[2]):
                    izp = (iz+1) % s[2]
                    lsx = 0.5*(f[ix,iy,iz] - f[ix-1,iy,iz])
                    rsx = 0.5*(f[ixp,iy,iz] - f[ix,iy,iz])
                    wx = lsx*rsx
                    slopes[0,ix,iy,iz] = 2*wx/(lsx+rsx) if wx>0 else 0
                    lsy = 0.5*(f[ix,iy,iz] - f[ix,iy-1,iz])
                    rsy = 0.5*(f[ix,iyp,iz] - f[ix,iy,iz])
                    wy = lsy*rsy
                    slopes[1,ix,iy,iz] = 2*wy/(lsy+rsy) if wy>0 else 0
                    lsz = 0.5*(f[ix,iy,iz] - f[ix,iy,iz-1])
                    rsz = 0.5*(f[ix,iy,izp] - f[ix,iy,iz])
                    wz = lsz*rsz
                    slopes[2,ix,iy,iz] = 2*wz/(lsz+rsz) if wz>0 else 0                

def MonCen2(f,slopes):
    """ Monotonized central slope limiter """
    slopes[...] = 0
    for i in range(f.ndim):
        ls = f-np.roll(f,1,axis=i)
        rs = np.roll(ls,-1,axis=i)
        w  = ls*rs > 0
        if f.ndim==1:
            slopes[w] = 2*w*ls*rs/(ls+rs)
        else:
            slopes[i,w] = 2*ls[w]*rs[w]/(ls[w]+rs[w])