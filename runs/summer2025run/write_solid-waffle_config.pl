# Get:
# Target directory -- where we are building the files
# SCA number (1..18)
# start exposure number
# end expsoure number
($TARGET_DIR, $SCA, $ESTART, $EEND) = @ARGV;

# detector name (doesn't affect the calculation)
print(sprintf "DETECTOR: SCA%02d\n", $SCA);

# flat images
print("LIGHT:\n");
for $e ($ESTART..$EEND) {
    print($TARGET_DIR.'/');
    print(sprintf "99999999_SCA%02d_Flat_%03d.fits", $SCA, $e);
    print "\n";
}

# dark images
print("DARK:\n");
for $e ($ESTART..$EEND) {
    print($TARGET_DIR.'/');
    print(sprintf "99999999_SCA%02d_Noise_%03d.fits", $SCA, $e);
    print "\n";
}

# format code: here we have converted everything to 6
print "FORMAT: 6\n";

# characterization - advanced
print "CHAR: Advanced 1 3 3 bfe\n";

# this could be un-commented to speed things up for tests,
# but isn't going to be as accurate
#print "CHAR: Basic\n";

# timestamp corresponding to the reset (starting at 1)
print "TIMEREF: 1\n";

# number of superpixels, e.g., 32x32
print "NBIN: 32 32\n";

# for best results, want to turn on all the non-linearity fitting
print "FULLNL: True True True\n";

# polynomial order for gain solution
print "NLPOLY: 3 2 16\n";

# implement subtractions in the correlation function - True is recommended
print "IPCSUB: True\n";

# time steps for Method 1
print "TIME: 2 8 9 15\n";

# output file
print(sprintf "OUTPUT: $TARGET_DIR/sw-SCA%02d-E%03d\n", $SCA, $ESTART);

# hot pixel characterization (not used downstream)
print "HOTPIX: 1000 2000 0.1 0.1\n";

# write to list of summary files
# (note this is to a separate output, not to the solid-waffle configuration)
$fsummary = sprintf "summary_files_%02d.txt", $SCA;
open(OUT, ">>$fsummary");
print OUT (sprintf "$TARGET_DIR/sw-SCA%02d-E%03d_summary.txt\n", $SCA, $ESTART);
close OUT;
