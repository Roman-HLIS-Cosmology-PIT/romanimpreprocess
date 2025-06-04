import numpy as np
import sys
import asdf
from roman_datamodels.dqflags import pixel

"""IPC utilities"""

def ipc_fwd(image, kernel, gain=None):
    """Carries out an IPC operation on the image.

    image should be a 2D numpy array of size (ny,nx)
    kernel should be a 4D numpy array of size (3,3,ny,nx)

    This returns a 2D numpy array:
    output[y,x] = sum_{dy,dx} input[y-dy,x-dx] kernel[1+dy,1+dx,y-dy,x-dx]

    Natively works in e, but if gain is provided then works in DN (does g^-1 K g).
    """

    im = image
    if gain is not None: im = gain*image

    # start with the center image
    output = im*kernel[1,1,:,:]

    # nearest neighbors
    # dy=1, dx=0
    output[1:,:] += im[:-1,:]*kernel[2,1,:-1,:]
    # dy=-1, dx=0
    output[:-1,:] += im[1:,:]*kernel[0,1,1:,:]
    # dy=0, dx=1
    output[:,1:] += im[:,:-1]*kernel[1,2,:,:-1]
    # dy=0, dx=-1
    output[:,:-1] += im[:,1:]*kernel[1,0,:,1:]

    # diagonals
    # dy=1, dx=1
    output[1:,1:] += im[:-1,:-1]*kernel[2,2,:-1,:-1]
    # dy=1, dx=-1
    output[1:,:-1] += im[:-1,1:]*kernel[2,0,:-1,1:]
    # dy=-1, dx=1
    output[:-1,1:] += im[1:,:-1]*kernel[0,2,1:,:-1]
    # dy=-1, dx=-1
    output[:-1,:-1] += im[1:,1:]*kernel[0,0,1:,1:]

    if gain is not None: output /= gain

    return(output)

def ipc_rev(image, kernel, order=2, gain=None):
    """Inverse operation of ipc_fwd to the given order. Grows the footprint of each pixel to (2*order+1,2*order+1).

    If gain is provided, then does g^-1 K^-1 g image instead of K^-1 image. (Equivalent: operate on DN instead of e.)
    """

    image2 = image
    if gain is not None: image2 = gain*image
    output = np.copy(image2)
    for j in range(order):
        output = output + image2 - ipc_fwd(output,kernel)
    if gain is not None: output /= gain
    return(output)

"""LINEARITY UTILITIES"""

def _lin(z,coefs,linextrap=True):
    """Helper function to evaluate Legendre-based function.

    z = rescaled signal (modified DN), shape (ny,nx)
    coefs = Legendre polynomial coefficients, shape(p_order+1,ny,nx)
    linextrap = linearly extrapolate beyond end of range? (boolean)

    Returns:
    phi = sum_l coefs_l P_p(z), shape (ny,nx)
    exflag = extrapolated beyond |z|=1?
    """

    exflag = np.abs(z)>1 # are we extrapolating?
    phi = np.copy(coefs[0,:,:])
    poly_prev = np.ones_like(phi)
    poly = np.copy(z)
    for L in range(1,np.shape(coefs)[0]):
        if linextrap:
            phi += coefs[L,:,:]*np.where(exflag,np.sign(z)**L*(1+L*(L+1)/2.*(np.abs(z)-1)),poly)
        else:
            phi += coefs[L,:,:]*poly
        # Legendre polynomial recursion relation
        poly_next = (2*L+1)/(L+1)*z*poly - L/(L+1)*poly_prev
        poly_prev = poly
        poly = poly_next

    return phi, exflag

def linearity(S, linearity_file, origin=(0,0)):
    """Performs a linearity correction.

    Inputs:
    S = input data shape (ny,nx)
    linearity_file = asdf file with linearity data
    origin = (x,y) of the lower-left corner of S in the convention of the file

    So for example, if you have a block S that corresponds to region [128:132,256:260] then you
    would give origin = (256,128)

    Returns:
    Slin, shape (ny,nx), also in DN
    dq = uint32 flag array
    """

    (dy,dx) = np.shape(S)
    ymin = origin[1]
    ymax = ymin+dy
    xmin = origin[0]
    xmax = xmin+dx

    with asdf.open(linearity_file) as F:
        Smin = F['roman']['Smin'][ymin:ymax,xmin:xmax]
        Smax = F['roman']['Smax'][ymin:ymax,xmin:xmax]
        phi, exflag = _lin(-1+2*(S-Smin)/(Smax-Smin),F['roman']['data'][:,ymin:ymax,xmin:xmax])
        dq = np.copy(F['roman']['dq'][ymin:ymax,xmin:xmax])
    dq |= np.where(exflag, pixel.NO_LIN_CORR, 0).astype(np.uint32) # flag with bad linearity correction
    return phi, dq

def multilin(S, linearity_file, origin=(0,0), do_not_flag_first=True, attempt_corr=None):
    """Performs a linearity correction, but with multiple groups.

    Inputs:
    S = input data shape (ngrp,ny,nx)
    linearity_file = asdf file with linearity data
    origin = (x,y) of the lower-left corner of S in the convention of the file
    do_not_flag_first = don't flag the first read if it is out of range (useful for reset-read frames that we won't use anyway)
    attempt_corr = if provided, an array of the same shape as S that is True if we want to try the correction
                   and False otherwise (the idea is that we want to be able to *not* flag a pixel that is saturated)

    So for example, if you have a block S that corresponds to region [128:132,256:260] then you
    would give origin = (256,128)

    Returns:
    Slin, shape (ngrp,ny,nx), also in DN
    dq = uint32 flag array (right now 2D)
    """

    (ngrp,dy,dx) = np.shape(S)
    ymin = origin[1]
    ymax = ymin+dy
    xmin = origin[0]
    xmax = xmin+dx

    # accept everything if attempt_corr not provided
    if attempt_corr is None:
        attempt_corr = np.ones((ngrp,dy,dx),dtype=bool)

    phi = np.zeros(np.shape(S), dtype=np.float32)
    with asdf.open(linearity_file) as F:
        Smin = F['roman']['Smin'][ymin:ymax,xmin:xmax]
        Smax = F['roman']['Smax'][ymin:ymax,xmin:xmax]
        Sref = F['roman']['Sref'][ymin:ymax,xmin:xmax]
        dq = np.copy(F['roman']['dq'][ymin:ymax,xmin:xmax])
        for j in range(ngrp):
            z = -1+2*(S[j,:,:]-Smin)/(Smax-Smin)
            if j==0 and do_not_flag_first:
                z = np.clip(z,-1,1)
            phi[j,:,:], exflag = _lin(z,F['roman']['data'][:,ymin:ymax,xmin:xmax])
            phi[j,:,:] = np.where(dq & (pixel.NO_LIN_CORR|pixel.REFERENCE_PIXEL)==0, phi[j,:,:], S[j,:,:]-Sref)

            # flag reads with bad linearity correction
            if not(j==0 and do_not_flag_first):
                dq |= np.where(np.logical_and(exflag,attempt_corr[j,:,:]), pixel.NO_LIN_CORR, 0).astype(np.uint32)

    return phi, dq

def invlinearity(Slin, linearity_file, origin=(0,0)):
    """Calculates the inverse linearity.
    This is most likely to be used in simulations.

    Inputs:
    Slin = input data shape (ny,nx)
    linearity_file = asdf file with linearity data
    origin = (x,y) of the lower-left corner of S in the convention of the file

    So for example, if you have a block S that corresponds to region [128:132,256:260] then you
    would give origin = (256,128)

    Returns:
    S, shape (ny,nx), in DNlin (with Slin=0 corresponding to the bias level Sref)
    exflag = boolean, extrapolated?
    """

    (dy,dx) = np.shape(Slin)
    ymin = origin[1]
    ymax = ymin+dy
    xmin = origin[0]
    xmax = xmin+dx

    with asdf.open(linearity_file) as F:
        z = np.zeros_like(Slin)
        # binary search, robust over the range -1 < z < +1
        # (which should encapsulate anything; also automatically saturates)
        for j in range(1,25):
            phi, exflag = _lin(z,F['roman']['data'][:,ymin:ymax,xmin:xmax], linextrap=False)
                # linextrap=False saves some time
            z += np.where(phi<Slin, 1/2**j, -1/2**j)
        Smin = F['roman']['Smin'][ymin:ymax,xmin:xmax]
        Smax = F['roman']['Smax'][ymin:ymax,xmin:xmax]
        S = Smin + (Smax-Smin)/2.*(1+z)

    return S, exflag

def correct_cube(data,ipc_file,mylog,gain_file=None):
    """IPC corrects a full data cube (data) in place.

    It can operate on a data cube in DN if a gain_file is passed.
    """

    if ipc_file is None:
        mylog.append('No IPC file specified, skipping ...\n')
        return

    with asdf.open(ipc_file) as F:
        kernel = F['roman']['data']
        mylog.append('IPC kernel center range --> {:f},{:f}\n'.format(np.amin(kernel[1,1,:,:]), np.amax(kernel[1,1,:,:])))
        (ngrp,ny,nx) = np.shape(data)
        nb = (8192+(nx-np.shape(kernel)[-1])//2)%16
        mylog.append(' ..., {:d} groups, excluding {:d} border pixels\n'.format(ngrp,nb))
        if gain_file is None:
            g = 1.
        else:
            with asdf.open(gain_file) as G:
                g = np.copy(G['roman']['data'][nb:ny-nb,nb:nx-nb])
        for i in range(ngrp):
                data[i,nb:ny-nb,nb:nx-nb] = ipc_rev(data[i,nb:ny-nb,nb:nx-nb]*g, kernel)/g

"""IPC + inverse linearity forward modeling tools"""

class IL:
    """IPC+Inverse linearity class. This exists to wrap invlinearity in a way that
    is consistent with romanisim.

    If built with ipc_file=None, then skips the IPC.

    Optionally can start with some number of electrons (start_e, number or array) in the well.
    Useful for reset noise.

    Methods:
    __init__
    set_dq
    apply
    """

    def __init__(self, linearity_file, gain_file, ipc_file, start_e=0.):
        self.linearity_file = linearity_file
        self.gain_file = gain_file
        self.ipc_file = ipc_file
        self.start_e = start_e
        # need the .dq attribute
        with asdf.open(self.linearity_file) as f:
            self._dq = np.copy(f['roman']['dq'])

    def set_dq(self, ngroup=1, nborder=4):
        """This is so that the data quality flags can be propagated."""
        (ny,nx) = np.shape(self._dq)
        self.dq = np.zeros((ngroup,ny-2*nborder,nx-2*nborder), dtype=np.uint32)
        self.dq[:,:,:] = self._dq[None,nborder:ny-nborder,nborder:nx-nborder]

    def apply(self, counts, electrons=False, electrons_out=False):
        """Converts a linearized signal to a non-linear signal.

        Inputs could be:
        electrons (electrons = True)
        DN_lin (electrons = False)

        Outputs could be:
        g*(S-Sref) (electrons_out = True)
        DN_raw (electrons_out = False)
        """

        print('apply', electrons, electrons_out, np.shape(counts))
        print(counts[:6,:6])
        sys.stdout.flush()

        # this uses a 4d IPC file
        if self.ipc_file is not None:
            with asdf.open(self.ipc_file) as f:
                counts_conv = ipc_fwd(counts+self.start_e, f['roman']['data'])
        else:
            counts_conv = counts+self.start_e

        # gain factors for in and out
        (nyc,nxc) = np.shape(counts)
        g_in = 1.0
        g_out = 1.0
        if electrons or electrons_out:
            with asdf.open(self.gain_file) as f:
                # extract the gain (with reference pixels clipped if needed)
                g = f['roman']['data']
                (nyg,nxg) = np.shape(g)
                if nyg>nyc:
                    nb = (nyg-nyc)//2
                    g = g[nb:-nb,nb:-nb]
            if electrons: g_in = g
            if electrons_out: g_out = g

        # what to strip off the counts array
        nb = (8192-nyc//2)%16
        S,_ = invlinearity(counts_conv/g_in, self.linearity_file, origin=(nb,nb))

        if not electrons_out: return S

        # below here, we know electrons_out is on.
        with asdf.open(self.linearity_file) as F:
            return g_out*(S - F['roman']['Sref'][nb:nb+nyc,nb:nb+nxc])

def test__lin():
    """Simple test function."""
    z = np.linspace(-1.5,1.5,31).reshape((1,31))
    coefs = np.zeros((4,1,31))
    coefs[3,:,:] = 1.
    phi,_ = _lin(z,coefs)
    print(phi)

def test_lin_ilin(linearity_file):
    """Some simple linearity tests."""

    ymin = 260; ymax = 262
    xmin = 140; xmax = 143
    dy = ymax-ymin
    dx = xmax-xmin
    with asdf.open(linearity_file) as F:
        print('Smin', F['roman']['Smin'][ymin:ymax,xmin:xmax])
        print('Smax', F['roman']['Smax'][ymin:ymax,xmin:xmax])
        S = F['roman']['Sref'][ymin:ymax,xmin:xmax] + np.linspace(0,dx*dy-1,dx*dy).reshape((dy,dx))
    Slin, dq = linearity(S, linearity_file, origin=(xmin,ymin))
    Sfwd, exflag = invlinearity(Slin, linearity_file, origin=(xmin,ymin))

    print('coefs:')
    with asdf.open(linearity_file) as F:
        print(F['roman']['data'][:,ymin:ymax,xmin:xmax])
    print('signal [DN_raw]:')
    print(S)
    print('inverted signal [DN_lin]:')
    print(Slin)
    print('recovered signal [DN_raw]:')
    print(Sfwd)
    print('flags')
    print(dq,exflag)

def test_IL(linearity_file, gain_file, ipc_file):
    """Some tests for the inverse linearity class."""

    ILTEST = IL(linearity_file, gain_file, ipc_file)
    n = 4088
    NE = np.zeros((n,n), dtype=np.float32)
    ymin = 260; ymax = 262
    xmin = 140; xmax = 143
    print(ILTEST.apply(NE,electrons=True,electrons_out=False)[ymin:ymax,xmin:xmax])
    NE[::3,::3] = 2.0e3
    print(ILTEST.apply(NE,electrons=True,electrons_out=False)[ymin:ymax,xmin:xmax])

if __name__=="__main__":
    """Test function"""

    if len(sys.argv)<3:
        print('call with python linearity.py <linearity_file> <gain_file> [<ipc_file>].')
        exit()

    test__lin()
    test_lin_ilin(sys.argv[1])
    if len(sys.argv)<4:
        test_IL(sys.argv[1], sys.argv[2], None)
    else:
        test_IL(sys.argv[1], sys.argv[2], sys.argv[3])
