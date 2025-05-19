HLWAS Image Preprocessing Tools
#############################################

This package handles basic pre-processing of Roman data prior to feeding it to PyIMCOM. The ultimate aim will be for it to handle both simulations and real Level 1/2 data.

Dependencies
******************************

You will need to install:

* `galsim <https://galsim-developers.github.io/GalSim/_build/html/overview.html>`_: This is a general-purpose package used for image simulations (romanisim is built on top of it).

* `romanisim <https://github.com/spacetelescope/romanisim>`_: This is a Roman-specific image simulator (in development; testing here is with commit ``e8d2cb5``, as of May 2025 the pypi version didn't have all the functionality we are using). ``romanimpreprocess.from_sim`` uses the ``romanisim`` functions rather than calling the whole script, since we are generating ramps from a 2D image rather than a catalog.

* `romancal <https://roman-pipeline.readthedocs.io/en/latest/>`_: The Roman pipeline to go from Level 1 (uncalibrated, 3D) to Level 2 (calibrated, 2D) data. *Note*: this is not used for scripts in ``romanimpreprocess.from_sim``.

The ``romanimpreprocess`` workflows do not explicitly call the Calibration Reference Data System (CRDS), although both ``romanisim`` and ``romancal`` have the ability to do so. Rather, we are specifying calibration reference files in the YAML configurations.

Converting an OpenUniverse simulation to L1 format
*****************************************************

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

This will generate both the output (simulated L1) file, and the truth WCS file, in this case ``sim1_asdf_wcshead.txt`` (which exists in the simulation but wouldn't be part of the real data).

General notes
======================

The OpenUniverse 2024 simulations can be found at `the IPAC site <https://irsa.ipac.caltech.edu/data/theory/openuniverse2024/overview.html>`_.

The OpenUniverse simulations are in coordinates native to the detector array: y=1 is along the "bar" on one side (the other 3 sides of the H4RG array can be packed much closer together), and x=1 is on the side corresponding to Channel 1. The L1 data products are in Science frame coordinates, in which all detectors are oriented the same way. Additionally, there is a parity inversion between the two systems. See `the Roman documentation at STScI <https://roman-docs.stsci.edu/data-handbook-home/wfi-data-format/coordinate-systems>`_ for a description of the coordinates. The ``romanimpreprocess.from_sim.sim_to_isim`` script performs the appropriate flips, for both the simulated L1 file (``*.asdf``) and the WCS (``*_asdf_wcshead.txt``).

Advanced options
======================

*I'll insert a link here for advanced options.*
