# Usage:
# perl write_download_script.pl <sca number>
#
# This prints a download script to the screen. You can run it from the directory where you want to download the files.

($sca) = @ARGV;

$c = sprintf "WFI%02d", $sca;

for $line (split "\n", `cat download_wfi06.txt`) {
    $l2 = $line;
    $l2 =~ s/WFI06/$c/g;
    print "$l2\n";
}
