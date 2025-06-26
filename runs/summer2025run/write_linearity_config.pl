($TARGET_DIR, $SCA, $TAG) = @ARGV;

print qq~\{\n~;

# This can be un-commented to write only a portion of the image for de-bugging
#print qq~"STOP": 2,\n~;

# the SCA number
$SCAint = int($SCA);
print qq~"SCA": $SCAint,\n~;

# this section gives the information on the ramps that you want to use
# in the linearity fit.
# you can include multiple groups --- usually, a regular flat (Flat) and
# a gain flat (LoFlat), with the latter filling in the lower counts
print qq~"RAMPS": [\n~;

# in each case, take NRAMP ramp files with the numbering beginning at START
# TSTART is the first time to use in the ramp (starting at 1)

# this one is the high-intensity flat
print qq~\{\n~;
print qq~  "FORMAT": 6,\n~;
print qq~  "FILE": "$TARGET_DIR/99999999_SCA$SCA\_Flat_001.fits",\n~;
print qq~  "START": 1,\n~;
print qq~  "NRAMP": 50,\n~;
print qq~  "TSTART": 2\n~;
print qq~\},\n~;

# this one is the low-intensity flat
print qq~\{\n~;
print qq~  "FORMAT": 6,\n~;
print qq~  "FILE": "$TARGET_DIR/99999999_SCA$SCA\_LoFlat_001.fits",\n~;
print qq~  "START": 1,\n~;
print qq~  "NRAMP": 30,\n~;
print qq~  "TSTART": 2\n~;
print qq~\},\n~;

# including darks as well -- a dark ramp won't pull the fit,
# but since it's just for subtraction and the read noise is tiny
# compared to the Poisson noise in the flat, it's OK to have a smaller
# number here.
print qq~\{\n~;
print qq~  "FORMAT": 6,\n~;
print qq~  "FILE": "$TARGET_DIR/99999999_SCA$SCA\_Noise_001.fits",\n~;
print qq~  "START": 1,\n~;
print qq~  "NRAMP": 25,\n~;
print qq~  "TSTART": 2\n~;
print qq~\}\n~;

print qq~],\n~;

# in the above, the last set of ramps is the dark (needed to subtract to get the P-flat)
print qq~"DARK": -1,\n~;

# 3.04 s is the default frame time, but you can change it by un-commenting this
print qq~"TFRAME": 3.04,\n~;

# order of the polynomial to fit
print qq~"P_ORDER": 10,\n~;

# where the output file goes
# note both a .asdf and a .fits will be generated
print qq~"OUTPUT": "$TARGET_DIR/roman_wfi_linearitylegendre_$TAG\_SCA$SCA.asdf",\n~;

# we want SIGN=1 for an increasing ramp
print qq~"SIGN": 1,\n~;

# how much the slope of the linearity curve is reduced for us to call a pixel "saturated"
# (needs to be <1)
# note that JSON is particular about the leading zero in a float, so 0.25 is OK but .25 is not!
print qq~"SLOPECUT": 0.5,\n~;

# where to get the bias value (so linearized signal is set to 0 and its derivative is set to 1 here)
# "PATH" is where to find the bias in the asdf tree
# note this is from .asdf so SLICE=1 takes the *2nd* frame as a bias
print qq~"BIAS":\n~;
print qq~  \{\n~;
print qq~    "FILE": "$TARGET_DIR/roman_wfi_dark_$TAG\_SCA$SCA.asdf",\n~;
print qq~    "PATH": ["roman", "data"],\n~;
print qq~    "SLICE": 1\n~;
print qq~  \},\n~;

# how many DN below the bias to set the polynomial range
print qq~"NEGATIVEPAD": 500\n~;

print qq~\}\n~;
