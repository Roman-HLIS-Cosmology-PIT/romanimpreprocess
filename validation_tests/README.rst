Testing scripts
#####################################

This directory contains some simple scripts to derive properties of the simulation --> L1 --> L2 workflow.

Statistics of the realizations
==================================

The calling format is (this is an example)::

    python3 -m romanimpreprocess.tests.many_realizations sample_Step0.yaml sample_Step1.yaml 5 /tmp/runs

The arguments are:

* The YAML configuration for simulation --> L1 (here ``sample_Step0.yaml``)

* The YAML configuration for L1 --> L2 (here ``sample_Step1.yaml``)

* The number of realizations to run (here N=5; note that the seed is incremented internally by the script so you get a different realization)

* A temporary directory where large files can be saved (here ``/tmp/runs``; on OSC you can use the ``$TMPDIR`` environment variable)

The output is an 8x4096x4096 FITS data cube. The slices of the cube are:

#. The ideal slope based on the input simulation (flattened DN/s)

#. The median of the L1 difference images (group -1 minus group 1), in DN/s

#. The median of the L2 slope images

#. The number of realizations where that pixel was unmasked.

#. The mean of the unmasked realizations (in DN/s).

#. The standard deviation of the unmasked realizations (in DN/s).

#. The difference (mean of unmasked realizations minus the ideal image) - right now this includes the sky. Also in DN/s.

#. The median of the error maps (in DN/s).

The flag -1000 is used to indicate failures (e.g., means with no pixels).
