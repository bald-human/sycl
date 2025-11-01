import matplotlib.pyplot as pl
import numpy as np

def imshow(f,title='',**kwargs):
    """ Show an image, with preferred orientation """
    pl.imshow(np.transpose(f),origin='lower',**kwargs)
    pl.colorbar()
    pl.title('{}  max:{:.4f}  min:{:.4f}'.format(title,f.max(),f.min()));

def imshows(f,title='',**kwargs):
    f=np.array(f)
    if f.ndim==2:
        imshow(f,title,**kwargs)
    else:
        n=f.shape[0]
        rows=1+(n-1)//3
        cols=min(n,3)
        pl.figure(figsize=(cols*5,rows*4))
        for i in range(n):
            pl.subplot(rows,cols,1+i)
            if type(title) is np.ndarray or type(title) is list:
                t=title[i]
            else:
                t=title
            imshow(f[i,:,:],t,**kwargs);