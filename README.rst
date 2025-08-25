HLWAS Image Preprocessing Tools
###############################

This package handles basic pre-processing of Roman data prior to feeding it to PyIMCOM. The ultimate aim will be for it to handle both simulations and 
real Level 1/2 data.

See also the `readthedocs <https://romanimpreprocess.readthedocs.io/en/latest/>`_ page.

Dependencies
************

You will need to install:

* `galsim <https://galsim-developers.github.io/GalSim/_build/html/overview.html>`_: This is a general-purpose package used for image simulations 
  (romanisim is built on top of it).

* `romanisim <https://github.com/spacetelescope/romanisim>`_: This is a Roman-specific image simulator (in development; testing here is with commit 
  ``910af8f``, from August 2025). ``romanimpreprocess.from_sim`` uses the ``romanisim`` 
  functions rather than calling the whole script, since we are generating ramps from a 2D image rather than a catalog.

* `romancal <https://roman-pipeline.readthedocs.io/en/latest/>`_: The Roman pipeline to go from Level 1 (uncalibrated, 3D) to Level 2 (calibrated, 2D) 
  data. *Note*: this is not used for scripts in ``romanimpreprocess.from_sim``. Note that the ``romancal`` installation also installs some other modules 
  that ``romanimpreprocess`` calls directly (specifically: ``stcal`` and ``roman_datamodels``).

The ``romanimpreprocess`` workflows do not explicitly call the Calibration Reference Data System (CRDS), although both ``romanisim`` and ``romancal`` 
have the ability to do so. Rather, we are specifying calibration reference files in the YAML configurations.

If you want to *generate* your own calibration files from flats and darks (as opposed to using externally provided ones) then you will want to download 
`solid-waffle <https://github.com/hirata10/solid-waffle>`_.


Conventions
***********

In a "glue" script such as ``romanimpreprocess``, the linking of the conventions used by different tools can be a bit overwhelming. The current summary 
of the conventions is `here <docs/conventions.pdf>`_.

Converting an OpenUniverse simulation to L1 format
**************************************************

You can convert an OpenUniverse truth image file by running the ``sim_to_isim`` script::

  python3 -m romanimpreprocess.from_sim.sim_to_isim config.yaml

The simplest configuration ``config.yaml`` that you can run is as follows::

  ---
  # Input file
  IN: '/fs/scratch/PCON0003/cond0007/anl-run-in- prod/truth/Roman_WAS_truth_F184_14747_10.fits'
  OUT: 'sim1.asdf'
  READS: [0, 1, 1, 2, 2, 4, 4, 10, 10, 26, 26, 32, 32, 34, 34, 35]
  SEED: 500
  ...

Here:

* ``IN`` is the input file.
* ``OUT`` is the output file.
* ``READS`` is the Multi-Accum table (here the group averages are ``range(0,1)``, ``range(1,2)``, etc.)
* ``SEED`` is the random number generator seed (if this is missing then a default is used, but this is definitely not recommended!).

This will generate both the output (simulated L1) file, and the truth WCS file, in this case ``sim1_asdf_wcshead.txt`` (which exists in the simulation 
but wouldn't be part of the real data).

It is also possible to include a dictionary of calibration reference files::

  # reference files -- see sample_Step0.yaml
  CALDIR:
    linearitylegendre: '/fs/scratch/PCON0003/cond0007/cal/roman_wfi_linearitylegendre_DUMMY20250521_SCA10.asdf'
    gain: '/fs/scratch/PCON0003/cond0007/cal/roman_wfi_gain_DUMMY20250521_SCA10.asdf'
    dark: '/fs/scratch/PCON0003/cond0007/cal/roman_wfi_dark_DUMMY20250521_SCA10.asdf'
    read: '/fs/scratch/PCON0003/cond0007/cal/roman_wfi_read_DUMMY20250521_SCA10.asdf'
    ipc4d: '/fs/scratch/PCON0003/cond0007/cal/roman_wfi_ipc4d_DUMMY20250521_SCA10.asdf'
    biascorr: '/fs/scratch/PCON0003/cond0007/cal/roman_wfi_biascorr_DUMMY20250521_SCA10.asdf'

(Note that some of these have additional data: see `the from_sim Readme <docs/from_sim_README.rst>`_ for a detailed list of requirements.)

General notes
=============

The OpenUniverse 2024 simulations can be found at `the IPAC site <https://irsa.ipac.caltech.edu/data/theory/openuniverse2024/overview.html>`_.

The OpenUniverse simulations are in coordinates native to the detector array: y=1 is along the "bar" on one side (the other 3 sides of the H4RG array can 
be packed much closer together), and x=1 is on the side corresponding to Channel 1. The L1 data products are in Science frame coordinates, in which all 
detectors are oriented the same way. Additionally, there is a parity inversion between the two systems. See `the Roman documentation at STScI 
<https://roman-docs.stsci.edu/data-handbook-home/wfi-data-format/coordinate-systems>`_ for a description of the coordinates. The 
``romanimpreprocess.from_sim.sim_to_isim`` script performs the appropriate flips, for both the simulated L1 file (``*.asdf``) and the WCS 
(``*_asdf_wcshead.txt``).

Advanced options
================

*I'll insert a link here for advanced options.*

Converting an L1 to L2 image
*****************************************************

You can (partially) convert an L1 image (unprocessed data) to an L2 image (2D with instrument artifacts cleaned or flagged) by running the 
``gen_cal_image`` script. Some "big picture" known issues at this point are:

- Only the internal steps (propagating bad pixel flags, saturation and cosmic ray flagging, linearity/IPC, dark, flat, bias) work right now. The WCS has 
  to be externally provided, and outputs, while flattened, are still in instrumental units (DN/s).

- Not all of the metadata and error arrays populate correctly in this version. We're working on it!

The script can be run via::

  python3 -m romanimpreprocess.L1_to_L2.gen_cal_image config_L1_to_L2.yaml

The simplest configuration ``config_L1_to_L2.yaml`` that you can run is as follows::

  ---
  # Input file
  IN: 'sim1.asdf'
  OUT: 'sim2.asdf'
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
  ...

Here:

* ``IN`` is the input (L1) file.
* ``OUT`` is the output (L2) file.
* The WCS and format is externally provided by one of the \*WCS keywords (in this case: ``FITSWCS``).
* ``CALDIR`` is a directory of calibration files to use (``romanimpreprocess`` uses this in place of the ``*.imap`` files used in the SOC tools,
  but it would be straightforward for the calling script to write the imap files into a configuration YAML).

This will generate the output (simulated L2) file, with the provided WCS (in this case
``sim1_asdf_wcshead.txt``) included.

See `the L1_to_L2 Readme <L1_to_L2/>`_ for detailed instructions and all the options.

Utilities
*********

The ``utils/`` folder includes some utilities that are intended to be called by the pipelines, but also that users might find useful for postprocessing, 
visualization, or other applications. See the `utilities page <utils/>`_ for more details.

A few useful test scripts are in `tests <tests/README.rst>`_.

Information for specific runs
*****************************

You can find information on specific runs we have done (or are doing) as follows. The code in these directories will be updated in the future when we 
build new calibration files; but it deals with specific input formats that may change (as opposed to the rest of the repository that is intended to be 
fully general):

* Summer 2025 run (with FPT tests + OpenUniverse): `here <runs/summer2025run/>`_.
