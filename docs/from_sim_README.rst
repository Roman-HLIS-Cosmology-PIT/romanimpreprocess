Detailed algorithms for generating L1 images
##################################################

This page describes in a bit more detail how ``sim_to_isim`` works.

Usage
====================================

The ``sim_to_isim.py`` script generates simulated Roman ASDF images from the OpenUniverse simulations (we may add more input formats in the future). Its calling format is::

  python3 -m romanimpreprocess.from_sim.sim_to_isim config.yaml

You can also call this from python using the code::

    import yaml
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    romanimpreprocess.from_sim.sim_to_isim.run_config(config)

which allows you to make lots of images as needed.

Interaction between the codes
---------------------------------------

The code calls ``romanisim``, but it passes some packaged algorithms into it, so it is more forming a combined workflow than acting as a wrapper. Aside from the need to add the read pattern to ``parameters.read_pattern`` (which is unavoidable), we were able to do this without any monkey-patching (because I dislike monkey-patching).

Fields in the configuration
====================================

The configuration is a Python dictionary (normally read from a YAML file).

**Required fields:**

* ``IN``: The input FITS image to simulate.

* ``OUT``: The output Level 1 ASDF image. Must end in ``.asdf``.

* ``READS``: The read pattern. This is a flattened list of an even number of integers, for example the pattern::

    READS: [0, 1, 1, 2, 2, 4, 4, 10, 10, 26, 26, 32, 32, 34, 34, 35]

  means that the groups that are being averaged are ``range(0,1)``, ``range(1,2)``, ``range(2,4)``, ``range(4,10)``, etc. If every read is being used in a group average, then the second integer in each pair is the same as the first one of the next pair. If there are dropped frames, then this won't be the case, e.g., if we had written ``...10, 10, 25, 26, 32, ...`` instead, then frame 25 would be dropped.

**Optional fields:**

* ``SEED``: Integer seed for the random number generator; technically optional but it's a really bad idea to leave it out and generate more than one file.

* ``FITSOUT``: If True, also write a FITS version of the data cube (with file extension ``_asdf_to.fits`` instead of ``.asdf``). (This has only the data, not the metadata, and was primarily intended so that the output is easily viewable in ds9.)

* ``CALDIR``: This is a dictionary of calibration files. Defaults are used if ``CALDIR`` is not provided. The files are unique to that SCA (so you will need 18 sets of files to describe the full WFI focal plane).

* ``CNORM``: A normalization constant that scales the overall throughput of the simulation (default: 1.0, since this is relative to the "ideal" effective area curve in the OpenUniverse simulation).

* ``NO_AMP33``: If True, then does not try to read reference output information from the calibration files.

Calibration file requirements
====================================

The following ASDF calibration files are required if ``CALDIR`` is set (or optional if indicated). Each one has a format and units:

CALDIR['dark']
-----------------------------------------

Dark data. There are several arrays in it:

* ``f['roman']['data']``: ngroup x 4096 x 4096 array, float32, DN:

  A group-averaged dark data cube (should be averaged over many darks, but with no subtractions). The reference pixels are important here.

* ``f['roman']['dark_slope']``: 4096 x 4096 array, float32, DN/s:

  The dark current map. The reference pixel data in this array is ignored.

CALDIR['gain'] 
-------------------

The gain array. The tree contains:

* ``['roman']['data']``: 4096 x 4096 array, float32, e/DN:

  The per-pixel gain (a constant array can be provided if per-pixel gain is not available). The reference pixel data in this array is ignored.

CALDIR['ipc4d'] 
-------------------

The IPC kernel.

* ``['roman']['data']``: 3 x 3 x 4088 x 4088 array, float32, dimensionless:

  The IPC kernel K, in the sense that if there are ``N[y,x]`` electrons in pixel (x,y), then ``K[dy,dx,y,x]*N[y,x]`` electrons appear to be in pixel (x+dx,y+dy). This should be normalized in the sense that ``np.sum(K[:,:,y,x])==1``. Note that the reference pixels are not included. Kernel contributions that go off the edge of the science array should be zero (but ``sim_to_isim`` doesn't explicitly check this).


CALDIR['linearitylegendre']
--------------------------------

The linearity data. The tree contains:

* ``['roman']['data']``: (order+1) x 4096 x 4096, float32, mixed units:

* ``['roman']['Smin']``: 4096 x 4096, float32, DN:

* ``['roman']['Smax']``: 4096 x 4096, float32, DN:

  This is a cube of Legendre polynomial coefficients and the bounds. The relation between raw signal (S) and linearized signal (Slin) in pixel (x,y) is via a two-step transformation::

    # re-scale S into the domain of the Legendre polynomials:
    # S=Smin --> z=-1
    # S=Smax --> z= 1
    z = -1 + 2*(S[y,x]-Smin[y,x])/(Smax[y,x]-Smin[y,x])

    # pseudocode for Legendre polynomial sum, we don't actually implement it this way
    Slin[y,x] = np.sum([data[L,y,x] * legendre(L)(z) for L in range(order+1)])

  The reference pixels are included in the array but not used.

* ``['roman']['dq']``: 4096 x 4096 array, uint32, dimensionless:

  The data quality flags from the linearity determination.

* ``['roman']['Sref']``: 4096 x 4096 array, float32, DN:

  The signal in DN that corresponds to "0 e in well". Note that unlike a CCD, where a charge packet in the silicon may truly be "empty", in Roman detectors there are always many free charges on the p-type side of the photodiode (the exact number can't be measured) and so charge in the well is always relative to some level.

CALDIR['read'] 
-------------------

The read noise cube. The tree contains:

* ``['roman']['data']``: 4096 x 4096 array, float32, DN:

  The standard deviation of the read noise (for a single read).

* ``['roman']['resetnoise']``: 4096 x 4096 array, float32, DN:

  The standard deviation of the reset noise.

* ``['roman']['anc']['C_PINK']`` and ``['roman']['anc']['U_PINK']``: float, DN:

  The amplitudes for correlated and uncorrelated (across the 32 readout channels) 1/f noise.

* ``['roman']['amp33']``: (optional) If specified, contains information needed to simulate the reference output. The contents are:

  * ``['roman']['amp33']['valid']``: True

  * ``['roman']['amp33']['med']``: 4096 x 128 array, float32, DN: median of the reference output

  * ``['roman']['amp33']['std']``: 4096 x 128 array, float32, DN: standard deviation of the reference output

  * ``['roman']['amp33']['M_PINK']`` and ``['roman']['amp33']['RU_PINK']``: float: parameters for 1/f noise in the reference output.


CALDIR['biascorr'] (optional)
-----------------------------------------

If provided, this file contains information on how to correct dark current + non-linearity information to get the correct median level. It should contain:

* ``f['roman']['data']``: ngroup x 4088 x 4088 array, float32, DN:

  The difference between the predicted dark signal (from the dark current slope and linearity curve) and the observed darks. This is to be added to the simulation outputs.

* ``f['roman']['t0']``: float, s:

  The time from reset to the reference level (i.e., what corresponds to "0 e in the well").

Code structure
=======================

The ``Image2D`` class is the main object you will encounter. 

Initialization
-------------------

``Image2D`` can be initialized from a simulated image::

    x = Image2D('anlsim', fname='Roman_WAS_truth_F184_14747_10.fits')

The ``__init__`` function takes a file type, currently ``'anlsim'``, but which is designed to be extendable in the future if we get another simulation file type. This fills in the 2D image data (``x.image``), but also the filter, date, and pointing/WCS information. Note that the input file already has PSF/pixelization, but is noiseless.

*Comment:* The OpenUniverse 2024 simulation is in the Detector frame. The flip to convert to the Science frame is performed in the initialization function.

Simulation
-----------------------------

The Roman images can be simulated using the ``simulate`` method::

    x.simulate(use_read_pattern, caldir)

This is an expanded version of ``romanisim.image.simulate`` that calls lower-level ``romanisim`` routines. It first constructs a blank image (i.e., containing only dark and sky, but no astronomical objects), using information in ``caldir`` (except if ``caldir`` is None, in which case defaults are used). Then Poisson-distributed counts are added based on the 2D image. The construction of the simulated ramps is carried out either by ``romanisim.l1.make_l1`` (if you are using the default ``caldir=None``) or by ``make_l1_fullcal`` (if you are providing ``caldir``). Note that ``make_l1_fullcal`` calls the ``romanisim.l1.apportion_counts_to_resultants`` and ``romanisim.l1.add_read_noise_to_resultants`` routines. However, it passes its own function for converting from electrons to DN to ``romanisim.l1.apportion_counts_to_resultants``: this is a ``romanimpreprocess.utils.ipc_linearity.IL`` object, and incorporates IPC, gain,  some offsets (see below), and inverse linearity.

Either way, this process only produces the ngroup x 4088 x 4088 cube of the science pixels. The reference pixel padding is added by a call to ``romanisim.l1.make_asdf``. If ``caldir`` is provided, then the reference pixels are filled in (including their own read and reset noise) by ``fill_in_refdata_and_1f``. Note that the correlated (banded) noise is also added here since it spans across the reference pixels.

*Comment:* The correlations across the reference output (``amp33`` in the L1 ASDF file) aren't included yet. Also the correlated noise is pure 1/f right now, so the alternating column noise is not simulated. As such, the current setup would not be able to realistically test the new reference pixel subtraction schemes.

A simulated slope fit and L2 image cube is generated (following the workflow in ``romanisim``), but we're not doing anything with those at the moment.

Writing files
-------------------

The Level 1 data file is written to ASDF with the call::

  x.L1_write_to(config['OUT'])

We also write the header (with flipping of the WCS if needed) with the suffix ``_asdf_wcshead.txt``, and if ``FITSOUT`` is true then we write the data cube with the suffix ``_asdf_to.fits``.

Methodology
=======================

The ordering of operations assumed here is as follows.

Reset
---------

We implement a Gaussian reset noise.

The mean reset level is set to a negative number of elementary charges so that it will integrate up to 0 on average in the dark at the time used to compute the "0 e" level (that would usually be the first stable frame after the reset, but it doesn't have to be).

Charge accumulation
-----------------------------------

Accumulation of signal into pixels (in elementary charges, e) is simulated first. Right now this is a Poissonian process.

* An implementation choice is that the sky+dark is built up first, and then the astronomical scene is added later. This is fine because for the Poisson distribution "add and then draw" is the same as "draw and then add".

* Another implementation choice is that the total counts up through the last frame are drawn first, and then they are apportioned in between reads. This is fine because again "Poisson and then binomial" is the same as "draw multiple Poisson variables".

* The brighter-fatter effect and quantum yield are not yet implemented.

Inter-pixel capacitance (IPC)
----------------------------------

IPC is implemented next, using the per-pixel map (although in practice that map might be constant in super-pixels).

*Comment:* It is not clear that it is entirely distinct from nonlinearity and gain, since some contributions to the nonlinearity and gain happen in the pixel and thus are "simultaneous" with IPC: the collected charges and even the boundary of the depletion zone are moving around to minimize free energy, and IPC means that this process is not independent across pixels). But because of how IPC and gain are measured, we do them before non-linearity.

Gain
---------

By definition, this is a simple conversion from elementary charges to linearized digital numbers ("DN_lin") in the same pixel.

Non-linearity
-----------------------

The non-linearity curve is computed using the Legendre polynomial cube. Pixels that go up to Smax (saturation level) will be clipped.

Uncorrelated read noise
---------------------------

The uncorrelated read noise term is added next. This is Gaussian white noise.

Bias
---------------------

A bias (from ``biascorr``) can be added to make sure that the median dark comes out right. In building the calibration files, this is really computed from the median dark minus what you get by running the dark current through the non-linearity curve, so it accomplishes this by construction.

The ``biascorr`` is usually small (except in the read-reset frame), but this step does make the hot pixels appear a bit more realistic (e.g., in cases where they are initially hot but then the dark current decreases before they saturate). But I don't think we really want to use the hot pixels for science so this aspect may not matter. Similarly, I don't think we want to use the read-reset frame directly for science, although it probably contains some useful calibration information.

Reference pixels
--------------------

The reference pixel padding is done next. These pixels have reset and read noise, but don't respond to the sky and are not included in the IPC calculation (since empirically we don't see IPC involving the edge pixels).

Correlated read noise
----------------------------

The correlated read noise (both 1/f components that are common across all channels and independent) are generated by FFT'ing a Gaussian random vector whose length is twice the readout.

* Read noise that is correlated across multiple frames is not yet implemented.

