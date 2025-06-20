Detailed algorithms for converting L1 to L2 images
########################################################

This page describes in a bit more detail how ``gen_cal_image`` works.

Usage
====================================

The ``gen_cal_image.py`` script converts L1 to L2 images. It carries out the pipeline steps associated with the image processing (linearity, ramp fitting and jump detection, IPC, dark, flat). Its calling format is::

  python3 -m romanimpreprocess.L1_to_L2.gen_cal_image config.yaml

You can also call this from python using the code::

    import yaml
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    romanimpreprocess.L1_to_L2.gen_cal_image(config)

which allows you to process many images in a scripted way.

Interaction between the codes
---------------------------------------

In its present form, the code does not wrap romancal (the original intention) but rather runs some of the key algorithms itself (calling stcal where appropriate). It also makes use of the utilities in roman_datamodels.

Fields in the configuration
====================================

The configuration is a Python dictionary (normally read from a YAML file).

**Required fields:**

- ``IN``: File name for the input L1 data cube (ASDF format).

- ``OUT``: File name for the output L2 image (ASDF format).

- World coordinate system: must specify one of:

  - ``FITSWCS``: File name with FITS header text (e.g., as written ``sim_to_isim``).

  - Other options are coming soon; at the very least, I need to add gwcs input.

- ``CALDIR``: A directory of calibration files (all ASDF). The calibration files have the same meaning as described in the `from_sim README <../from_sim/>`_. This has both required and optional entries:

  - Required: dark, flat, gain, ipc4d, linearitylegendre, mask, read, saturation

  - Optional: biascorr

**Optional fields:**

- ``RAMP_OPT_PARS``: Parameters for which to optimize the ramp weights (slope in DN/s, gain in e/DN, sigma_read in DN).

- ``JUMP_DETECT_PARS``: Threshold parameters for the jump detection algorithm (we may ultimately enable more than one choice).

- ``FITSOUT``: If True, also writes a FITS output with the ``.asdf`` ending replaced with ``_asdf_to.fits``.

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
