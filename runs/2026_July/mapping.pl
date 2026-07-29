# This script converts the desired files from ASDF --> FITS
# and moves them to the target directory.

# It requires the environment variables:
# USE_SCA -> in "01" .. "18"
# MAST_DL_DIR -> MAST direct download directory
# TMPDIR -> temporary storage for this file

use IO::Handle;

($targetdir) = @ARGV;

# Get the list of files
@files = ();
$i=0;
$sca = $ENV{'USE_SCA'};
$c = 'WFI'.$ENV{'USE_SCA'};
for $line (split "\n", `cat download_wfi$sca.txt`) {
    $l2 = $line;
    $d = (split ' ', $l2)[-1];
    $files[$i] = $d;
    $i++;
}

# Convert all files
$tmpfile = $ENV{'TMPDIR'}."/conv.py";
open(OUT, ">$tmpfile");
print OUT qq`from solid_waffle import asdf_to_fits\n`;
print OUT qq`asdf_to_fits.main(\n`;
print OUT qq`  input_dir="`;
print OUT $ENV{'MAST_DL_DIR'};
print OUT qq`",\n`;
print OUT qq`  output_dir="`;
print OUT $ENV{'TMPDIR'};
print OUT qq`",\n`;
print OUT qq`  fmatch="*$c\_uncal.asdf",\n`;
print OUT qq`  format="wfi_tvac_rst"\n`;
print OUT qq`)\n`;
close OUT;
print "\nConverting files ...\n";
system "cat $tmpfile";
print "\n";
system "python $tmpfile";
print "\n";
STDOUT->flush();

# Move the files to the indicated directories.
print "# $$$ MOVE FILES $$$\n";
open(OUT, ">$targetdir/mapping.txt");
@files = sort {$a<=>$b} @files;
$i = 0;
for $i (0..199) {
    if ($i>=100) {
        $target = $targetdir.(sprintf "/99999999_SCA%02d_Noise_%03d.fits", $sca, $i-99);
    }
    elsif ($i>=50) {
        $target = $targetdir.(sprintf "/99999999_SCA%02d_LoFlat_%03d.fits", $sca, $i-49);
    }
    else {
        $target = $targetdir.(sprintf "/99999999_SCA%02d_Flat_%03d.fits", $sca, $i+1);
    }
    $ipt = $ENV{'TMPDIR'}."/$files[$i]";
    $ipt =~ s/\.asdf$/_asdf_to\.fits/;
    system "mv $ipt $target\n";
    print "$files[$i] $target\n";
    $tfile = (split "/", $target)[-1];
    print OUT "$files[$i] $tfile\n";  # save the mapping information
}
close OUT;
print "# $$$ END MOVE FILES $$$\n";
STDOUT->flush();
