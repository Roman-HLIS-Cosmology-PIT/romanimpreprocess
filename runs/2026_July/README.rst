July 2026 run information
#########################

This set of runs is controlled by the script:

.. code-block:: bash

    # replace "06" with the SCA you are using ... "01" through "18"
    sbatch --export=USE_SCA=06 make_sca_files_2026Jul.job

The files used are downloaded from MAST into the ``$MAST_DIR`` directory. The full list of files used is on the `Wiki page <https://github.com/Roman-HLIS-Cosmology-PIT/romanimpreprocess/wiki/Files-used-for-July-2026-run>`_.

The Slurm script can be modified with the appropriate input and output directories via the ``export`` lines. It calls the appropriate scripts and writes its own solid-waffle configurations.
