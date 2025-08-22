"""
Constants and settings used in this package.
"""

## Dimensionality information ##

# Detector array parameters
nside = 4096
nborder = 4
nchannel = 32

# Useful combinations
nside_active = nside - 2 * nborder
channelwidth = nside // nchannel
nside_augmented = nside + channelwidth

## See the LaTeX documentation for how these are used. ##

Omega_ideal = 2.8440360952308436e-13  # this is (0.11 arcsec)^2 in sr
h_Planck = 6.62607015e-24  # J s (now exact) # noqa: N816
g_ideal = 1.458  # e/DN for the flattened digital numbers; arbitrary, sets zero-point of output
