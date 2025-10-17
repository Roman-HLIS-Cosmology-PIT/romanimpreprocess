Detailed algorithms for converting L1 to L2 images
########################################################

This page describes in a bit more detail how ``gen_cal_image`` works.

Usage
====================================

The ``gen_cal_image.py`` script converts L1 to L2 images. It carries out the pipeline steps associated with the image processing (linearity, ramp fitting and jump detection, IPC, dark, flat). Its calling format is::

  python3 -m romanimpreprocess.L1_to_L2.gen_cal_image config.yaml

or if you want to also generate noise images::

  # this calls gen_cal_image internally, so you don't need that as an extra step
  python3 -m romanimpreprocess.L1_to_L2.gen_noise_image romanimpreprocess/sample_Step1.yaml

You can also call this from python using the code::

    import yaml
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    romanimpreprocess.L1_to_L2.gen_cal_image.calibrateimage(config)

or to generate noise images as well::

    import yaml
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    romanimpreprocess.L1_to_L2.gen_cal_image.calibrateimage(config | {'SLICEOUT':True}) # add slice information
    romanimpreprocess.L1_to_L2.gen_noise_image.generate_all_noise(config)

which allows you to process many images in a scripted way.

Interaction between the codes
---------------------------------------

In its present form, the code does not wrap romancal (the original intention) but rather runs some of the key algorithms itself (calling stcal where appropriate). It also makes use of the utilities in roman_datamodels.

Fields in the configuration
====================================

The configuration is a Python dictionary (normally read from a YAML file).

**Required fields for all uses:**

- ``IN``: File name for the input L1 data cube (ASDF format).

- ``OUT``: File name for the output L2 image (ASDF format).

- World coordinate system: must specify one of:

  - ``FITSWCS``: File name with FITS header text (e.g., as written ``sim_to_isim``).

  - Other options are coming soon; at the very least, I need to add gwcs input.

- ``CALDIR``: A directory of calibration files (all ASDF). The calibration files have the same meaning as described in the `from_sim README <../from_sim/>`_. This has both required and optional entries:

  - Required: dark, flat, gain, ipc4d, linearitylegendre, mask, read, saturation

  - Optional: biascorr

**Fields required only for noise generation:**

- ``NOISE``: The information on which noise fields to generate.

  - ``LAYER``: A list of noise layers to generate, e.g., ``['RP', 'RS2']`` (a description of the allowed codes is in development).

  - ``TEMP``: Scratch storage location for intermediate steps in simulated noise files as they are being generated. (This only needs to exist while the script is running, so temporary storage on the compute node, as provided by many HPC systems, should work.)

  - ``SEED``: Random number seed (positive integer).

  - ``OUT``: Output noise file (.asdf). The output ASDF tree has a ``'config'`` branch with the configuration file and a ``'noise'`` branch with the float32 numpy array of the noise images.

**Optional fields:**

- ``RAMP_OPT_PARS``: Parameters for which to optimize the ramp weights (slope in DN/s, gain in e/DN, sigma_read in DN).

- ``JUMP_DETECT_PARS``: Threshold parameters for the jump detection algorithm (we may ultimately enable more than one choice).

- ``FITSOUT``: If True, also writes a FITS output with the ``.asdf`` ending replaced with ``_asdf_to.fits``.

- ``SKYORDER``: If provided, then subtracts a median-based sky model that is a 2D polynomial of the given order.

- ``NOISE_PRECISION``: The precision to store noise fields, as number of bits in IEEE 754 floating point convention. Options are 16 and 32 (default = 32).

A sample file would be::

    ---
    # Input file
    IN: 'sim1.asdf'
    OUT: 'sim2P.asdf'
    FITSWCS: 'sim1_asdf_wcshead.txt'
    CALDIR:
      linearitylegendre: '/fs/scratch/PCON0003/cond0007/cal/roman_wfi_linearitylegendre_DUMMY20250521_SCA10.asdf'
      gain: '/fs/scratch/PCON0003/cond0007/cal/roman_wfi_gain_DUMMY20250521_SCA10.asdf'
      dark: '/fs/scratch/PCON0003/cond0007/cal/roman_wfi_dark_DUMMY20250521_SCA10.asdf'
      read: '/fs/scratch/PCON0003/cond0007/cal/roman_wfi_read_DUMMY20250521_SCA10.asdf'
      ipc4d: '/fs/scratch/PCON0003/cond0007/cal/roman_wfi_ipc4d_DUMMY20250521_SCA10.asdf'
      flat: '/fs/scratch/PCON0003/cond0007/cal/roman_wfi_pflat_DUMMY20250521_SCA10.asdf'
      biascorr: '/fs/scratch/PCON0003/cond0007/cal/roman_wfi_biascorr_DUMMY20250521_SCA10.asdf'
      mask: '/fs/scratch/PCON0003/cond0007/cal/roman_wfi_mask_DUMMY20250521_SCA10.asdf'
      saturation: '/fs/scratch/PCON0003/cond0007/cal/roman_wfi_saturation_DUMMY20250521_SCA10.asdf'
    SKYORDER: 2
    RAMP_OPT_PARS:
      slope: 0.4
      gain: 1.8
      sigma_read: 7.
    JUMP_DETECT_PARS:
      SthreshA: 5.5
      SthreshB: 4.5
      IthreshA: 0.6
      IthreshB: 600.
    FITSOUT: True
    NOISE:
      LAYER: ['RP', 'RS2']
      TEMP: /fs/scratch/PCON0003/cond0007/MyNoise.asdf
      SEED: 15000
      OUT: 'sim2P_noise.asdf'
    ...

Summary of algorithms
=====================================

The principal algorithms used in this version of the code are as follows. Some implementations are "Internal" (in ``gen_cal_image``). Others point to other files in this repository (as indicated) or are called from external libraries (e.g., stcal). Note that some choices are provisional and will change as better algorithms become available.

.. list-table:: Algorithms in romanimpreprocess
   :widths: 25 50 25
   :header-rows: 1

   * - Step
     - Algorithm \& reference file(s)
     - Implementation
   * - Initialization
     - Read metadata from L1 image and ``'mask'`` file
     - Internal, ``initializationstep``
   * - Saturation check
     - Compare each group to ``'saturation'`` file (with checks for groups with some reads saturated) 
     - wrap algorithm from stcal (``flag_saturated_pixels``)
   * - Reference pixel correction
     - Simple interpolation from reference pixels \& reference output
     - ``utils.reference_subtraction``
   * - Bias correction
     - Simple subtraction, ``'biascorr'`` file
     - Internal
   * - (Classical) linearity
     - Legendre polynomial fit, coefficients in ``'linearitylegendre'``
     - ``utils.ipc_linearity``
   * - Dark current subtraction
     - Simple subtraction, ``'dark'`` (uses ``dark_slope`` array)
     - Internal, ``subtract_dark_current``
   * - Inter-pixel capacitance
     - De-convolution with kernel from ``'ipc4d'``
     - ``utils.ipc_linearity``
   * - Ramp fitting
     - Simplified version of optimal fit `(Casertano et al. 2022) <https://www.stsci.edu/files/live/sites/www/files/home/roman/_documents/Roman-STScI-000394_DeterminingTheBestFittingSlope.pdf>`_ with ramp slope used in weighting fixed.
     - ``utils.fitting``
   * - Jump detection
     - Flagging with single \& double differences `(Sharma & Casertano 2024) <https://ui.adsabs.harvard.edu/abs/2024PASP..136e4504S/abstract>`_, but with no attempt at correction or fitting multiple ramps.
     - ``utils.fitting``
   * - Flat field
     - The flat field is IPC-deconvolved; ``'flat'`` is used, but so is ``'ipc4d'``.
     - ``utils.flatutils``

*Note*: The ``'gain'`` file is used as ancillary data in many steps whenever a threshold is in elementary charges instead of DN.


Some steps are not carried out in this code:

* World Coordinate System determination (we read from another file, this isn't fit by this code; in this case the PIT plans to start by importing the SOC WCS solution)

* absolute calibration (i.e., from flattened DN_lin/s to MJy/sr)

Noise realizations
######################

You can generate simulated noise realizations *as well as* the calibrated images with the ``gen_noise_image`` script. For example::

    from romanimpreprocess.L1_to_L2 import gen_noise_image
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    gen_noise_image.calibrateimage(config | {'SLICEOUT': True})
    gen_noise_image.generate_all_noise(config)

Here ``'SLICEOUT':True`` tells ``calibrateimage`` to save the information on which resultants are used to construct the slope image, so that ``generate_all_noise`` can pull from the correct distribution.

You can tell ``gen_noise_image`` which noise realizations to generate by putting a ``NOISE`` block in the configuration file::

  NOISE:
    LAYER: ['RP', 'RS2']
    TEMP: /fs/scratch/PCON0003/cond0007/MyNoise.asdf
    SEED: 15000
    OUT: 'sim2P_noise.asdf'

Here:

* ``LAYER`` is a list of which noise layers to generate (see below for the codes).

* ``TEMP`` is a temporary file location (it is recommended to use the on-node temporary storage on an HPC cluster).

* ``SEED`` is the random number generator seed (integer).

* ``OUT`` is the location of the output file.

Noise layer code system
=========================

The noise layer string (e.g., ``'RS2'``) indicates which noise elements should be included. Each command begins with a capital letter indicating the type of command, and in some cases is followed by other characters (lower case letters, numbers, underscores) that provide arguments.

The types of commands are:

* ``R``: Generate simulated read noise (including both white and 1/f components). These realizations are generated as 3D images (resultant,y,x) in Level 1 space. If the 'a' flag is set (``'Ra'``) then this is passed through the pipeline by differencing; schematically::

    L1_to_L2(data_3D+simulated_noise_3D) - L1_to_L2(data_3D)

  Otherwise a "bias+noise" field is generated and processed (so no subtraction is necessary).

  The noise can be clipped based on the median and interquartile range at some number of equivalent sigmas with the 'z' directive, e.g., ``'Raz4.5'`` will clip at 4.5 sigma. This is useful if you want to be able to feed another noise layer through the pipeline without re-computing the outlier mask, and thus it is recommended for use with feeding noise realizations to PyIMCOM, etc.

* ``O``: Generate noise realizations intended for pseudo-Poisson bias corrections. To be used by Gabe et. al in prep. (Recommended to not turn on both O and P simultaneously.)

* ``P``: Generate Poisson noise. This must come after ``R`` (if present). The variants of this command that are currently supported are:

  * ``Pbr`` : re-sampled Poisson noise with background (sky level) only.

  * ``Pr`` : re-sampled Poisson noise including sources as well as background (i.e., "total" Poisson noise).

* ``S``: Perform sky subtraction on the noise realizations of the given order, e.g., ``'S2'`` removes a 2nd order polynomial from the noise realization, ``'S0'`` removes a constant, etc.

* ``C``: Comment (does not affect the noise generated). This can also be used to give statistically equivalent noise layers unique designations so that they can be referred to later, e.g., by PyIMCOM. So if you wanted 3 read noise layers with a constant subtracted off, you could write::

    LAYER: ['RS0C0', 'RS0C1', 'RS0C2']

  Of course, since this is a comment, you could also name them however you want as long as you don't use capital letters::

    LAYER: ['RS0Cmickey_mouse', 'RS0Cdonald_duck', 'RS0Cgoofy']

