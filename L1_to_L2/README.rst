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
    JUMP_DETECT_PARS:
      SthreshA: 5.5
      SthreshB: 4.5
      IthreshA: 0.6
      IthreshB: 600.
    FITSOUT: True
    ...

Code structure
====================================

*Under construction*
