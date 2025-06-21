"""Utilities for simple sky estimation."""

import numpy as np
import scipy.stats
from scipy.special import legendre_p

def binkxk(arr, k):
    """Bin-averaging utility for 2D array, kxk"""

    (ny,nx) = np.shape(arr)
    nyo = ny//k; nxo=nx//k

    return np.mean(arr[:k*nyo,:k*nxo].reshape((nyo,k,nxo,k)), axis=(1,3))

def smooth_mode(arr, pc=25., pksmooth=0.5, niter=3):
    """Find the mode of the histogram of arr.

    Inputs
    -------
    arr : the numpy array
    pc : (optional, default=25.) percentile cut for the histogram
    pksmooth : (optional, default=0.5) number of sigmas to smooth
    niter : (optional, default=3) number of peak-finding iterations

    Returns
    --------
    (best fit mode, width of weighting function)

    Comments
    --------
    Ignores nans
    """

    # initial setup of the center and sigma of the distribution
    c1 = np.nanpercentile(arr,pc)
    c2 = np.nanpercentile(arr,50.)
    c3 = np.nanpercentile(arr,100.-pc)
    gauss_iqr_in_sigmas = scipy.stats.norm.ppf((100.-pc)/100.)*2
    ctr = c2
    sigma = (c3-c1)/gauss_iqr_in_sigmas

    N = 21
    for k in range(niter):
        histsmooth = np.zeros(N)
        z = ctr + np.linspace(-1,1,N)*sigma
        for i in range(1,N-1):
            w = np.exp(-.5*( (z[i]-arr)/(pksmooth*sigma) )**2)
            histsmooth[i] = np.sum(np.where(np.isnan(w),0.,w))

        # now find the peak of the histogram
        i_peak = np.argmax(histsmooth)
        # now we want to fit a quadratic to these 3 points
        b = ( histsmooth[i_peak+1]-histsmooth[i_peak-1] )/2.
        a = ( histsmooth[i_peak+1]+histsmooth[i_peak-1] )/2. - histsmooth[i_peak]
        ctr = z[i_peak] + (z[1]-z[0])*(-b/2./a)

    return (ctr,sigma*pksmooth)

def medfit(arr, N=8, order=2):
    """Fits a low-order polynomial to a 2D array.

    Inputs
    -------
    arr : 2D numpy array
    N : (optional) regions to break into on each dimension
    order : order of polynomial to fit

    Returns
    --------
    coef : polynomial coefficients (numpy array)
    arrmed : fit to the median (numpy array, same shape as arr)

    Comments
    ---------
    arrmed[y,x] = sum coef_ij P_i(u) P_j(v) where u = 2*x/nx-1, v = 2*y/ny-1
    (so u and v are scaled to -1 .. +1)
    coef ordering is:
    0,0   0,1   0,2   ...   0,order-1   0,order
    1,0   1,1   1,2   ...   1,order-1
    ...
    order,0

    total number of coefficients is (order+1)*(order+2)//2
    """

    # get indices for the center, as close as we can get
    (ny,nx) = np.shape(arr)
    kx = nx//N; ky = ny//N
    px = (nx%N)//2; py = (ny%N)//2
    u_ = 2*( px-.5 + kx*np.linspace(.5,N-.5,N) )/nx - 1
    v_ = 2*( py-.5 + ky*np.linspace(.5,N-.5,N) )/ny - 1
    u,v = np.meshgrid(u_,v_)

    # now get the medians
    meds = np.nanmedian(arr[py:py+N*ky,px:px+N*kx].reshape((N,ky,N,kx)), axis=(1,3))

    # and now get the coefficients
    # this is not efficient for large N, but we don't anticipate such a use case
    nc = (order+1)*(order+2)//2
    basis = np.zeros((nc,N,N))
    k=0
    for i in range(order+1):
        temp = legendre_p(i,u)
        for j in range(order+1-i):
            basis[k,:,:] = temp*legendre_p(j,v)
            k+=1

    # now build the linear system solution, Ax=b
    A = np.zeros((nc,nc))
    b = np.zeros(nc)
    for ipix in range(N):
        for jpix in range(N):
            if not np.isnan(meds[jpix,ipix]):
                A += np.outer(basis[:,jpix,ipix],basis[:,jpix,ipix])
                b += meds[jpix,ipix]*basis[:,jpix,ipix]
    x = np.linalg.solve(A,b)

    # now get the Legendre polynomials on a pixel grid
    LPX = np.zeros((order+1,nx))
    LPY = np.zeros((order+1,ny))
    u_ = np.linspace(-1,1-2/nx,nx)
    v_ = np.linspace(-1,1-2/ny,ny)
    for i in range(order+1):
        LPX[i,:] = legendre_p(i,u_)
    for j in range(order+1):
        LPY[j,:] = legendre_p(j,v_)

    # and this is where the components are summed up
    arrmed = np.zeros((ny,nx))
    k = 0
    for i in range(order+1):
        for j in range(order+1-i):
            arrmed += x[k]*np.outer(LPY[j,:],LPX[i,:])
            k+=1

    return x,arrmed.astype(arr.dtype)
