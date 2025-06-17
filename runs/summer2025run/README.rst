Scripts to generate calibration files: Summer 2025 run
##################################################################

Overview and Purpose
===============================

This is a set of simple scripts to generate reference files from darks and flats using ``solid-waffle``. It is being used for the Summer 2025 image simulation run. Most of this material we expect to be useful later, including for in-flight calibrations, but the file formats will be different (this run used Focal Plane Test data). Therefore we intend for this set of scripts to remain as a record of the Summer 2025 run after it is complete, and for later runs to be in their own subdirectories of ``romanimpreprocess.runs``.

Note that only some calibration information can be extracted from darks and flats: this includes dark current, bias, noise, saturation, linearity, gain, inter-pixel capacitance (IPC), and the high spatial frequency flat ("P-flat"). Some other important calibrations (e.g., distortion, bandpass, low spatial frequency flat, point spread function, absolute flux) are only possible with other data sources. For the Summer 2025 runs, these are pulled from simulations or other data sources. But in any case, it is probably going to be helpful for the future to be able to convert solid-waffle outputs into a format that is interoperable with romanisim and (at least partially) with the calibration reference data formats used by the SOC.

Usage
===============================

Everything here is controlled by the script ``make_sca_files.job.template`` (you will want to make your own copy, ``make_sca_files.job`` with the variables set according to your platform). This is a bash script that can be submitted via slurm::

    sbatch --export=USE_SCA=10 make_sca_files.job

The ``$USE_SCA`` variable must be set for that job; it is an integer between 01 and 18 (leading zeros required).

The script is a bash script, with a bunch of parameters to export. One of these specifies the read pattern: for example, ``export READ_PATTERN_NAME="IM_107_8"`` points to a read pattern in the yaml file ``settings_IM_107_8.yaml``. Another is a tag that will label all the output files. The remaining exports in the script point to directories with data and where ``solid-waffle`` is located on your system. They also point to the ``$TARGET_DIR`` where the output files will be written.

You will need to place the data files in the directories you indicate (probably on a scratch disk).

Inputs and outputs
===============================

The inputs to this run include:

* dark exposures (in ``$DARK_DIR``)
* high-intensity flat exposures for linearity (in ``$FLAT_DIR``)
* low-intensity flat exposures for gain and to extend the linearity curve (in ``$LOFLAT_DIR``)

The outputs are ASDF calibration reference files. These are mostly in the form of standard calibration reference files that can be used by romanisim and romancal to ensure compatibility, but there are some differences noted below where we need to specify different information. Additionally, the file format naming convention is::

  roman_wfi_<file type>_<tag>_SCA<number>.asdf

where the ending ensures that it will never be confused with a SOC calibration file.

The output file types and contents of the ``roman`` branch are as follows (differences from the SOC format are noted in **bold**):

- ``biascorr`` **This is a correction that needs to be applied in the simulation to have the correct mean, beyond what we get by adding the dark current and non-linearity curve. Without this, a pixel with negligible dark current would have the same signal at all time, but there is an additional electronic bias especially in the read-reset frame. This may not matter very much since we exclude that frame from the ramp fitting, but we want to get as much right as possible.**
    - ``data`` (difference of dark data minus what we get from running dark_slope through the inverse linearity curve, ngroup x 4088 x 4088 DN)
- ``dark``
    - ``data`` (averages of darks, ngroup x 4096 x 4096, DN)
    - ``dq`` (4096 x 4096, uint32 flags)
    - ``dark_slope`` (dark current, 4096 x 4096, DN/s)
    - ``dark_slope_err`` (4096 x 4096, DN/s)
- ``gain``
    - ``data`` (4096 x 4096, e/DN_lin)
    - ``dq`` (4096 x 4096, uint32 flags)
- ``ipc4d`` **Extended to 4D to allow for spatially varying IPC kernel, as seen during testing.**
    - ``data`` (IPC kernel, active pixels only, 3 x 3 x 4088 x 4088, unitless)
    - ``dq`` (4088 x 4088, uint32 flags)
- ``linearitylegendre`` **The linearity curve is represented in Legendre polynomial coefficients over the given range. This is stable even if the linearity curve has some wiggles in it. Also we don't use an explicit inverse linearity curve, since we pass a class to romanisim that numerically inverts the linearity curve when needed.**
    - ``data`` (linearization coefficients (order+1) x 4096 x 4096)
    - ``dq`` (4096 x 4096, uint32 flags)
    - ``Smin`` (min signal for polynomial fit, 4096 x 4096, DN)
    - ``Smax`` (max signal for polynomial fit, 4096 x 4096, DN)
    - ``Sref`` (reference signal level that corresponds to 0 DN_lin, 4096 x 4096)
    - ``dark`` (dark signal subtracted from flat, 4096 x 4096, DN_lin/s)
    - ``pflat`` (pixel-level flat field, nrampgroups x 4096 x 4096, DN_lin/s)
    - ``ramperr`` (max residual from linearity fit to each light level, nlevels x 4096 x 4096, DN)
- ``mask``
    - ``dq`` (4096 x 4096, uint32 flags)
- ``pflat`` **This is read separately in a few places so we copied it out of linearitylegendre. This is only the pixel-level flat right now.**
    - ``data`` (pixel level flat field, 4096 x 4096, median rescaled to 1)
    - ``dq`` (4096 x 4096, uint32 flags)
- ``read`` **The resetnoise array is also included so that we can implement a random reset value in the simulation. We also added noise information for the reference output.**
    - ``anc`` (dictionary of correlated noise parameters: at least the parameters ``U_PINK`` and ``C_PINK`` for uncorrelated and correlated 1/f noise amplitudes)
    - ``data`` (1 sigma read noise per pixel, 4096 x 4096, DN)
    - ``resetnoise`` (1 sigma reset noise per pixel, 4096 x 4096, DN)
    - ``amp33`` (dictionary):
        - ``med`` (median reference output, 4096 x 128, DN)
        - ``std`` (sigma of reference output, 4096 x 128, DN)
        - ``M_PINK`` and ``RU_PINK`` (reference output 1/f noise parameters)
- ``saturation``
    - ``data`` (saturation level on raw data, 4096 x 4096, DN)
    - ``dq`` (4096 x 4096, uint32 flags)

Detailed steps
===============================

We now discuss the specific steps in ``make_sca_files.job.template``.

Reformatting the files
---------------------------------

The first step (after setting the environment variables) is to convert the files::

  # make single FITS files of the darks and flats
  # The number of frames to use is indicated in each command.
  cd $SCRIPT_DIR
  pwd
  python convert_dark.py $DARK_DIR 56 $TARGET_DIR $USE_SCA
  python convert_flt.py $FLAT_DIR 56 $TARGET_DIR $USE_SCA
  python convert_loflt.py $LOFLAT_DIR 12 $TARGET_DIR $USE_SCA

The nature of these scripts varies depending on how the data is formatted; this selection is for the Focal Plane Test data, where each frame is stored in a separate FITS file and needs to be merged. We don't expect this in the future since in-flight flats and darks are going to be formatted in the Level 1 format (though there may be other formatting necessary).

In each case, there is a directory containing the files (``$DARK_DIR``, ``$FLAT_DIR``, or ``$LOFLAT_DIR``); a number of frames to use; a target location; and the SCA number.

Running the flat autocorrelation analysis
--------------------------------------------

There are tools in ``solid-waffle`` to analyze the autocorrelations of the flats and estimate IPC and gain. The script runs these in parallel in groups of 10 flats::

  # set up solid-waffle
  # This is for using 50 exposures.
  cd $SCRIPT_DIR
  pwd
  echo "" > summary_files_$USE_SCA.txt; rm summary_files_$USE_SCA.txt # suppress warning
  perl write_solid-waffle_config.pl $TARGET_DIR $USE_SCA  1 10 > config1_$USE_SCA.txt
  perl write_solid-waffle_config.pl $TARGET_DIR $USE_SCA 11 20 > config2_$USE_SCA.txt
  perl write_solid-waffle_config.pl $TARGET_DIR $USE_SCA 21 30 > config3_$USE_SCA.txt
  perl write_solid-waffle_config.pl $TARGET_DIR $USE_SCA 31 40 > config4_$USE_SCA.txt
  perl write_solid-waffle_config.pl $TARGET_DIR $USE_SCA 41 50 > config5_$USE_SCA.txt
  # run solid-waffle
  # This step can be parallelized
  cd $SOLID_WAFFLE_DIR
  python test_run.py $SCRIPT_DIR/config1_$USE_SCA.txt > $TARGET_DIR/sw-SCA$USE_SCA-P1.log &
  python test_run.py $SCRIPT_DIR/config2_$USE_SCA.txt > $TARGET_DIR/sw-SCA$USE_SCA-P2.log &
  python test_run.py $SCRIPT_DIR/config3_$USE_SCA.txt > $TARGET_DIR/sw-SCA$USE_SCA-P3.log &
  python test_run.py $SCRIPT_DIR/config4_$USE_SCA.txt > $TARGET_DIR/sw-SCA$USE_SCA-P4.log &
  python test_run.py $SCRIPT_DIR/config5_$USE_SCA.txt > $TARGET_DIR/sw-SCA$USE_SCA-P5.log &
  wait
  # cleanup files (these are stored in cal)
  rm $SCRIPT_DIR/config?_$USE_SCA.txt

The configurations are written by the ``write_solid-waffle_config.pl`` script. There are comments in that script for each line of the configuration. Note that the first echo command prints a list of summary files that later stages of the script can extract.

Making the gain files
----------------------------

The script ``make_gain_file.py`` extracts the information from the solid-waffle summary files, averages the results, and writes ASDF gain and IPC files::

  # now print the gain files
  cd $SCRIPT_DIR
  pwd
  python make_gain_file.py summary_files_$USE_SCA.txt $USE_SCA 
  $TARGET_DIR/roman_wfi_gain_$TAG\_SCA$USE_SCA.asdf
  # this is no longer needed
  rm summary_files_$USE_SCA.txt

Making the noise files
---------------------------

This part runs solid-waffle's noise script (a slight update of the one used in `Troxel et al. <https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.2801T/abstract>`_) The call is::

  # run the noise script
  cd $SOLID_WAFFLE_DIR
  pwd
  python noise_run.py -f 6 -i $TARGET_DIR/99999999_SCA$USE_SCA\_Noise_001.fits -o $TARGET_DIR/noise_SCA$USE_SCA.fits -n 100 -t 2 -cd 5.0 -rh 7 -tn 34

and the options are::

  # here:
  # 6 = file format (consistent with FPS)
  # $TARGET_DIR/99999999_SCA$USE_SCA\_Noise_001.fits = name of first noise file
  # $TARGET_DIR/noise_SCA$USE_SCA.fits = output file
  # 100 = number of darks
  # 2 = frame to start dark current determination (Fortran ordered since it is a FITS file)
  # 5.0 = cutoff for 'low CDS' pixel
  # 7 = row overhead (placeholder, not actually propagated into the reference files)
  # 34 = number of frames to use for 'total' noise and dark current determination

Then the dark files are built from the output information (no major calculations here, but lots of pulling out data and putting it into the format usable by romanisim)::

  # build the dark file
  cd $SCRIPT_DIR
  pwd
  python make_dark_file.py IM_107_8 $TARGET_DIR/99999999_SCA$USE_SCA\_Noise_001.fits $TARGET_DIR/noise_SCA$USE_SCA.fits $USE_SCA  $TARGET_DIR/roman_wfi_dark_$TAG\_SCA$USE_SCA.asdf

Flat, linearity and saturation information
---------------------------------------------

This part analyzes the flat fields to produce linearity tables. It also produces some useful ancillary outputs including a saturation file and a P-flat.

*Note that the P-flat produced here isn't directly usable for science, since the illumination isn't the same as from astronomical sources. But it does have the small-scale structure and should produce a much more realistic flat to challenge the analysis tools than leaving out the flat entirely.*

We begin by configuring and running ``solid-waffle``'s linearity tools::

  # build the linearity files
  cd $SCRIPT_DIR
  pwd
  perl write_linearity_config.pl $TARGET_DIR $USE_SCA $TAG > linearity_pars_$USE_SCA.json
  cd $SOLID_WAFFLE_DIR
  pwd
  python linearity_run.py $SCRIPT_DIR/linearity_pars_$USE_SCA.json

The important adjustable parameters in the configuration file are described in the comments in ``write_linearity_config.pl``. This produces the large ``linearitylegendre`` output file (in ASDF format). Some information is pulled out from this file by the post-processing script::

  # post-process these to get pflat and saturation
  cd $SCRIPT_DIR
  pwd
  python postprocess_calfiles.py 
  $TARGET_DIR/roman_wfi_linearitylegendre_$TAG\_SCA$USE_SCA.asdf $USE_SCA
  python makemask.py $TARGET_DIR/roman_wfi_mask_$TAG\_SCA$USE_SCA.asdf $USE_SCA


Conventions
===============

Here we note some aspects of the conventions assumed for input files, and used for output files.

Reference frames
-----------------------

All output data is in the Science Frame. The Focal Plane Test data is in the Detector Frame, and the ``convert_*.py`` scripts perform the conversion.

Legendre polynomial cubes
----------------------------

Linearity data are stored in Legendre polynomial format for numerical stability. The key information is in the ``data``, ``Smin``, and ``Smax`` leaves of the linearitylegendre file. To take a 2D numpy image S (in raw DN) and convert to linearized DN, you have the steps:

* First, compute z, which packages the range Smin<S<Smax into -1<z<1. That is,

  (1+z)/2 = (S-Smin)/(Smax-Smin).

* Then we have Slin = sum_{L=0}^{p_order} ``data[L,:,:]`` * P_L(z)

Note that the linearization also takes out an intercept: the reference level Sref (in raw DN) maps to a linearized signal of 0 DN_lin.
