"""Sim -> L1 -> L2 workflow test."""


import contextlib
import gzip
import os

import asdf
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from astropy.wcs import WCS
from roman_datamodels.dqflags import pixel
from romanimpreprocess.from_sim import sim_to_isim
from romanimpreprocess.L1_to_L2 import gen_cal_image, gen_noise_image
from romanimpreprocess.utils import fpaplot, ipc_linearity, maskhandling, orientation, visualize

# Settings for this test
id = 163
sca = 4
tag = "TESTONLY"
dt = 3.04

# This is a shortened 14-read sequence that we don't plan on using in flight.
# I chose it here because it is short and so good for a unit test
# that we want to complete quickly.
#   -- C.H.
#
read_pattern = [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9, 10], [11, 12], [13]]


def genfile(out, v=1):
    """
    Generates a simple FITS image to test calibration.

    Parameters
    ----------
    out : str
        Output FITS file.
    v : int, optional
        Which version of the file to make.

    Returns
    -------
    None

    """

    # make the image
    N = 4088
    img = np.zeros((N, N))
    x_, y_ = np.meshgrid(np.arange(N), np.arange(N))
    for j in range(25):
        x = 10 + (N - 20) * j / 25.0
        y = 10 + (N - 20) * ((13 * j) % 25) / 25.0
        img += 10000 * j * np.exp(-0.5 * ((x_ - x) ** 2 + (y_ - y) ** 2) / 2**2)

    # make FITS header
    phdu = fits.PrimaryHDU(img)
    phdu.header["EXPTIME"] = 139.8
    phdu.header["FILTER"] = "F184"
    phdu.header["CRPIX1"] = (N + 1) / 2.0
    phdu.header["CRPIX2"] = (N + 1) / 2.0
    phdu.header["CD1_1"] = 3.0555555555555554e-05
    phdu.header["CD1_2"] = 0.0
    phdu.header["CD2_1"] = 0.0
    phdu.header["CD2_2"] = 3.0555555555555554e-05
    phdu.header["CTYPE1"] = "RA---TAN-SIP"
    phdu.header["CTYPE2"] = "DEC--TAN-SIP"
    phdu.header["CRVAL1"] = 37.0
    phdu.header["CRVAL2"] = -20.0
    phdu.header["LONPOLE"] = 215.0

    # SIP coefficients
    phdu.header["A_ORDER"] = 2
    phdu.header["A_0_2"] = 2.0e-6
    phdu.header["A_1_1"] = -1.0e-6
    phdu.header["A_2_0"] = 3.0e-6
    phdu.header["B_ORDER"] = 2
    phdu.header["B_0_2"] = 1.4e-5
    phdu.header["B_1_1"] = -1.0e-5
    phdu.header["B_2_0"] = 3.0e-7

    # these coordinates aren't consistent with the SCA location, but shoudn't be a problem for this test.
    phdu.header["RA_TARG"] = 37.0
    phdu.header["DEC_TARG"] = -20.0
    phdu.header["PA_OBSY"] = 185.0

    phdu.writeto(out, overwrite=True)


def _trim(arr, d):
    """
    Trims a 2D array by setting the outer regions to zero.

    Parameters
    ----------
    arr : np.array
        The array (trimmed in place).
    d : int
        How much to trim.

    Returns
    -------
    None

    """

    if d == 0:
        return
    arr[:, :d] = 0
    arr[:, -d:] = 0
    arr[:d, :] = 0
    arr[-d:, :] = 0


def _write_wcs(fn):
    """Writes the test WCS file."""

    with open(fn, "wb") as f:
        _x = gzip.decompress(
            b"\x1f\x8b\x08\x08t\x95\xf4j\x00\x03r0106202001001001001_0001_wfi05_f062_wcs.asdf\x00\xed="
            b"\x0b\x98\x1cE\x99\x13\x0e1, /\x91\x03E\x9b\x00\xd9\\\xdc\x99\xed\xf7c\xd9l\\\xf7\x01\x0b"
            b"\x9b\x87\xd9\r\x01Clzgzw\x87\xccLO\xba{6;\t\x08~\xe2\xc7\xe1\x89\n~\x82 9N\x05\xb9SQO\xcf"
            b"\x17a\x13\xf0\xd0\x83SQ\x02\x12\xe5\xc4\xe3\x13\xef\x14ADNH\xf0D\xef\xaf\xea\xc7\xf4\xcc"
            b"\xf4k\xf6\xbd\xc7\xf4\x97df\xaa\xea\xaf\xfa\xeb\xaf\xffYU\xfd\xe7\xac\xee\xa1\xde~\x82J"
            b"\x91)\xb2\xe5,\xf4]\x1e\x1a\xee^\xdf\xdb\xbd\xa9\x17\ny(<\xe7\x92\xeeu\x83\xf0\x9dj9g"
            b"\xb8\xfb<\xe2L\xc2T\xc6:\x0c\xd3HgSj\xa6\xd4\xa1\x18\x99\xd1\xf6\x96d2I\x9c\x99\xd6t\xb5"
            b"\x1d\xfdNBk\x80D_\xe5\\vDW\xf4r\x87]mh\xa3\xe6NEW\x93xDb\xb7R2\xc75\xbd\x83\x18\x1eW\t"
            b"\x8cI\xaf:\xa1\xe6\xb4\xa2\xaa\x1bm\xc4\xb8\x96W\x8b\xca\x98\xdaA\xb4\x8e\x9bf\xb1\xa3"
            b"\xbd},k\x8e\x97FRi-o\r4\xaa\xe9y\xc5\xc4\xdf[\xdbZ\x08\xa2\xa0\xe4\xa15\xfa\xd9FL@\x1fY"
            b"\xad\xd0Ap)6E^\xd92\x9e5L\r0\x81f\xea\xa4\xa9\x16P\xa5\x81~9\xa8\xbb\xa5r^5\x95\x8cb*"
            b'\x16\x96\xd0\xc4\x03"\xa7s\x8aaX\x83\xa4\xdc\xd2\x94\x9cW\n\xd9Q\xd50S\xeb\xec/}N]\r|I'
            b"\xcfZ\xd00\x1f\xcf$R\x9a>\xd6^\x8d\x87\x81\x7f'\xadu@\x9d8c\xc8\x0e\x19\x83\xc8Z\xa1"
            b"\x83l\x98J!\xa3\xe8\x19\x0fA\xa8\x14\x87\x08\x82z\x8c\xdfQ=A\x17\x15\xe5\x14\xc3\xd4"
            b"\xb5\x82\x96/\x03\xcd4=\x93-(\xa6jT\x93\xd2-\xf6 7\x1d\x8azz\x92\x8d\xf4\xb8\x9aW\x0c"
            b'\x0fqH .\xd5(q\x93\x18\xfdb\xb9\xaa\x1f\x8aZ\xc4T\x1e\xdb\x99\xae"/\xfa\r\xe8\xb03'
            b"\xa0+\xf4\x10@\xcf\x86\x98\x15aR\xc5\xecd\x8aY`2\xba\xea\xb2\x1d\r\x97\xd72j\xceh\xd75"
            b"\xe8\xcbK\xc2J]\x92\x9e.\x83\xeaJ\xa6f\xee\r\xd1\x0e\xe3$W\x10\xf1\xebkQ\xb2\xa3\x87"
            b"\x8cn!\xa0C/\x02\xc5\xb9xe\x1b\xa3U!d\xb1\x8a\x8c\xa5B\xd6D\x12\xcd\xd8H\xfc\x7f\x98p5"
            b"\xf7\x98\xbaR0\xd0o\xef\xb4\xddB@G\x98\x01\xf7\xb8\xfd\xf8\xaa4~V\xd9\x08K-\xf4\xd0\x19"
            b'\xadn\xc0y3\xdaw\x8ef\x91\xb2\xb5\xb4L\x17\xe0\x81\x96\xa0\x03\xe3\x93V\xb0\xc3f"\xda9'
            b"\xb8\xc8\x16\x12\xad\xad\xc1-\xaa\xb4\x04m\xb5\xabX\xc9\x0eb\xb7\xae\x8e\xaa\xbaZH\xab"
            b"\xf2\xa8\x8e{\x1b\xe8\xd94d\x91@-\x02}T=\x8b\x9a\xb9\xdf\xe5:\x80\xbeu}4I\x92m\x84Q\x84"
            b"\xc1\x95\x9c<\t\xa3\xb1\x0c-\xd04\xc7\xa5h\x92\x97\x04\x91\xa7\xdbp\x97\x84\xdb\x08"
            b"\x1cO\x96\xe2HA\xe4\x98\x14\xc3\n\xb4\xc4\xd2\x0c[\xe9c\x17\xf4!\x88\x0c'RLJ\xa2$\x96"
            b"\xe2\x19\x8ao#\xcc,\x1a\x91\xa7\x18\x9aM\t\x94(pN\xb7f\xb9\x08\x15\x1b7\xf5\xf5\x0e\xf4"
            b"\x0c\xf7\xf5\xa2\xb5\xc8i\xe9\xacYF\xd8$\xa5\x14\x02\xa1H\x91\xe4\x19A`h\xceS\x0f\x88"
            b"\xd0\xb0\xea4\r\xe33\xb4\xc8\xc0C\xb9\xdd\xba\xad\x10> r\x8c\xc8R\x12)p\x14'I\x9ch\x93i"
            b"\xb2\xa8\x19%\xe0\x14\x1b\x06-\xad\\\xd4\xb5\x91\x9c\x9a\x87\xc5Y\xdbj\x97\xab\xa3\xa3j"
            b"\xda\xccN\xa8\xb2\x03![\xd3aH>%p\x12O;\r\x0b\x19\xbb\xe6L\xf4\xd1\x8e\xfe\xb1,8A\x934"
            b"\x9f\xa4\xc8$E\rSB\x07\xfc\xe1$`5\xca\x05\x1c\xcb\x16TX\xa5\xc2\x98\xbc\xa3\x04\xec`\xc2"
            b'\xe46\\\xe8\xd4\x86\x8f\x8aW\xd3\xa9KQ<\xcd\nv\xcd\xf8\x98"\xe7\xb5\t(\x1fUr\x86j\x97'
            b"\xe6\x15\xd9T`\x8er6\xd3A\x0c\xf5\x0cP$%\xd5VY\xfc9\xb0NfHA\xa6\xf8\xba\xeaR~D\x85\x18"
            b"\xc3\x03Z\xd0U\xa3\x94\x03\x05o\x02\xd7\xb9\x10\xba\xaadd\xe0\x0cS\xd5\x0b\x0e\x99\x93"
            b"\xc4Vj[\xe5;\xed\xf9\xce\xb4\x11\xac\xe7'\xac6\xb0\x8e\xd0F\x88\x9eB\xa9\r\x86\x85\xbf"
            b"\x14\xfc\xa5\xe1/\xc0P^ \xe0\x01\x02q\x1c\x05p\x94\x08\x7f\x01\x80\x06\x00\x1a\x00h\x00"
            b"\xa0\x01\x80\xf6\x02 \xa6\xa2\x01\x80\x06\x00\x1a\x00h\x00`\x00\x80\x01\x00\x06\x00\x18"
            b"\x00@\x1c\xce@;\x86\xf7b\x0b\x00\x0c\x000\x00\xc0\x02\x00\x0b\x00,\x00\xb0h\x1a\x00\xc0"
            b"\x02\x00\x0b\x1d\xb3\xd0\x8e\xf5N\x81\x05\x00\x0e\x008\x00\xe0\x00\x80\x03\x00\x0e\x008"
            b"\x00\xe0\x00\x80\x03\x00\x0e:\xe6\xa0\x1dOz\x00y\x00\x00\x89$x\x00\xe0\x01\x80G\x14\x02"
            b"\x00\x1e\x00x\x00\xe0\x01@\x80\x8e\x05h'x\t+\x00\x80\x00\x00 w\x84\x80h\n\x00\x02\x00"
            b'\x08\x00 \x02\x80\x08\x00\xa2\x17@\x04\x00\x11\x00D\x00\x10\x01@D\xab\x00\x00"\x00H^'
            b"\x94$\x80\x94\x00%\t\x00$/Y%\x80\x94\xbc\xe4\x92\x04\xe7\x07\xf8\x01\xba\x19SR\xe8\x0e"
            b"\x16$E\x14\x1du\xa1\x97\ni\xd0\x80\x99j\x9e\xb6\xb4\xc8\x96\xfe\x01y`]\xf7y}\xb8t4\x0b"
            b"\x9c\n2\xed7\xc8\xee\x11\xc5\x00=\x88-W\x07\x12|\x15U\x82rQr%\xa4\x98]$\xe8a\x92\xee"
            b'\xe0\xf8\x0e\x909\x91e[\xaft{\xb6\xbd;\x92"yhL\xc2\xa7\xf3G&\xd1?`\x11HN\x1e\x85Zd'
            b"\x18R\xc8\x8e`\xd8l\x01\x8cN)\xaf\x16`\xdc\xddX\xe3\x80J\xd62H\xf2\xb5\xd1QC\x85b\x16+"
            b"\xe5\x8cj\x82\xd6A\x91<L\x8b\x04bZ#\xc2\x8f6B+\x9aY\xb0\x1a\xb2\n\x9a\n\xf7\xd4\x0f\xe3X"
            b"\xb8a\xdb$[\xf4\x18\xc8C\xa8\xbf\x0e\x15\xe0*m\xc4P\xf5\tlg\x90a\x98T\xd3%ls\x8a9d"
            b'\xeb`\r]\x85H\xc0\xa2V\xb5\xae\xfe\x8d5G\xab\xcf\xd4I\x86B\x9f$\xd5\xea\xe8\xe1"v6\x00'
            b"\x1c\xf4\xea\x18(*\xa43\x10\x07\x1b\xea\x98\x859TMd\x8d\xac\xe7\x9b\x8c\x17NA\x1a\x17"
            b"\xeb\xc1V\xd4\x9d\xb7jL\xd7JE\xd0u\xae\xae\xafT\x19\xea\x8e\x122q\x9e\xde\x82p\x85>\x91"
            b'\xd5.*\xba\x92\xcb\xa99\x9b\x9f,"jz\x16\x142h\xc7a\xd0\x8f\xedC\x1bzpiQ\xcb\x16LX(\xd7^'
            b"\xa8iy\x82\x02+\x859\x94\xa7H\x8e\x05s\x07\x16\x91r'/+EU7-\x92\xd2\x10\xe4\x03\x1a\"G1"
            b"\xa2H\xb3\xbcXi5\xc1\xa0z6\xc5\xb3$/\x08\x1c\xcf\xf3\x12\xe7jqgX\xd9k%\x0c\xad\xa4\xa3i"
            b"\xf6t\x0f\xf6l\x1e\xec\x06\xcbi\xb7\x06\xf3\x81\x94-^\xb5\xadI2%H\x1c\xcd\xb3\x92$\xd0"
            b"\xa4@!\x99\x842\x9a!EQdy\x86\x83\x01\x19\x06\xe9,\x88\ti\x89\xe1\x01IZ\xa0\x18I@\n\x0f"
            b"\x9c!\xe0y\x16\xcc\xb5\xc4\xc1CI4\xe3H\xb0\xae\xe0\x99\x83\x11c\x04F\xa08\x01\xe6C\x81"
            b"\xe3\xe0\x88\xa3\xa2\x8f\xa9\xa6g\xf2H2{\xfa\xd6WW\x03\xfd,\xe2\xb1\x80\x0e/J\xa2\xc4"
            b"\xf2\x12\xc3\xb1\xd5\xadt\x05\xc6\x11S<)\x80\x03\x00\xec\x05\xe6\x8d\xe2\xac\xf5\xd03"
            b"\x157i\xe3\xa6\xde\r\x1b\x876\xc1Z%I\x8a\xb1\xeb\xb5L)m\xda\xc2\x90\xa3\x9dB\x8b\x0fw#52"
            b"\x86\xb6\xaf\x88\x9e\r\xeb\xda@0'\xc0!\xcd\x8e) s\x8e\x7fv\xbeV*\x18j.\xd7FlRG\xd4\xed"
            b"\xca80\r8\x82\xd8\x89\xaa\x80\xaf\xd7\n\xaa\xeb%\x95F*\x15XV\xcd\xac\x99C}\xf5toJJ\x14"
            b"\x07n\x97\xa2\xe7\xcaD\xaf\x9a\xd7\x90*\xc0\x02Eh\xa3\xc4\xf9\x83\xc3\xbdC^)\xb3{D"
            b'\xeaRG]\x96\t\x88\x9e\x08 )V[\x80\x82\xad\x8dpdU\xe76"\r\x06\xc1\x9aE\t\x13\x14\x85'
            b'\x91\xd6\x10\x196mX\xd7m\xad\x83\xeb")`\xcbuG- \xa6v=Eky\xc0\x1b\xe4I\x8a\x83%\x96Hd'
            b"\x9e`\xf1=-`m\x18\x1e\xdc=\x89a\x81\x03X\xd1\x15K\x034\x14\xe8X\xc5Rb\xe0\xc6\xa2G\x04"
            b"\xff\x8b\x11\x91[\xc8[\xd8[\xc2\xefHT\xd6\x1c\x07\xbf\xc2\x9d\xb9\xa5\xa3\xd4\x8a\x0bAl"
            b"\xdd\xe6V\x16\xf5l^\xd1\xcb\xf6Z\xad\xdf\xb0\xbe\xafB\xb1\xd2H1;\tz\xb0\xae\x0eD\t\xfa"
            b"\x01\xd5iqW\xb5\x15)8\n\x10T\x96\xc3\xca\r\xd8*\x92\xe9 !@\xe6\\)\xb0]\xdb\x81u}\xf2p"
            b"\xf7\xa6\xf3\xfa@J\xe5\xfe\x81\x8bmYE\x9b3\xa3\xa0\xa0,\xc7\xc9pf\xdd\xa9kfW\x07\x12S"
            b"\x12d\x93\x96D\x16(\x0f\x8e\xab\xc0\n\x8e\xc6\xe8\xc4\xa4\xedr\x88\n\xe4\x04\x97]\xe0E"
            b"\xde\xd5\x19i\x15\xcd\x13i\x00\x8aNq\xbc\xc4\x03\xd5A\x03J<rh\x924H;G\xc3\x9a\xf1\x1c"
            b"\xc70\xa0j\\\xbb\r\xf8\x8c\xa9\x1aH\xc7\x98ZPA1\xba\x8e\x9f\x8ab#\xf0\xb5y\x8e\x02\xbd"
            b"\x01Z\x8ac\\\x05\x066T\xcfNz\xdd;\x8c\x16/\xd1\xacDJ\x0c\xa8FP&I\xd0~\xe0\x9a\xf2(b`"
            b"\x18P=\xa2\x9a$\xbd\xee\x01\x9a/IQ \xe6P\x0f\x9a\x9a\x11$\x16y\x1e\xb83\x8a\x84) E\xc6"
            b'\x92\x9c\xebF\x14``\x88\x04\r\xc4\x83\x8e\xae\xd4\xd1Fw\x07r\x13\xd4\xaa"\x19\xa8'
            b"\x1aNT=o\xd8\x93\x04='\xd0\xa2\x04\xcd\x04\x06i\x1d\xa7\x01\xeaa+\xee\x82gX\x1eX\x9dgE"
            b"\x8e\xe5y\xe4e\xe2b0\x7f\x10\x0e\x81\xd2\x84\n\x88GDn\x9bW\x14:<\x84\x01\xf2\xd3\xa2 a"
            b"\xff\xd43?\x98:\x043.\xd4xv\xd4\x19\x12\xd9\x04\na\x0eJ\x1c\xa4\rkl\x92\x85\xa0\x86fD"
            b"\x06\xaaX\xa0\\\x05p\xbb\xba\x13O\x85\xe4a\xb6\x10;\x89`\xfa\x00Hd+Lm\x96\x80nC\x9b{z"
            b"\xfa\x86\x86\x1cv\xcc\x16F5P\x01\x8e\xda\x96]\x8f\x03<\x99\xfe\xcd\x83\x83m\x84\xad\x1d"
            b'\x1c\xbd \xb1\x12\xd2\xfe"\x18\x07\xd1\xd1\x0b\x96F\x80*\x9a\xe3)\x81\xa2\x81\xc9'
            b"\xda\\\n\xe6rV\x93zc\x08\x8a\x15\xaa\xc6,]\xbea\xf0\x92\xf36\xac\xc7\x11-\x81\xbb\xa3@"
            b"\xe9\xc3TY<.(\x1a\x16\xe0H\xd6\x95wh\xc2\x8a\xc0\x9c\xd0\x93h\xa1\x06\x9a\x8a\x82_,"
            b'\xae"YA\xa2y\x81\xb4,\xb5 \x90\x807I\xa3*D=I\x82\xde-(\x1a\xd6S"\xc1\xcbv<\x0b\xdaF'
            b"\x17\xa6\x98B\x1e\x05c\xcf\x9e\x12\x91\xa4#gx\x82)+\x851\xb4\xbaI\xb0\x86\x12,\x10\xac"
            b"\xcd\x04x\x15\xd8wI\xe2}k$\xe99\x1am'T\x9f\xf4\xe0\x1d_w\x97\xb7\x0b\x0fj\xd1\xbc"
            b"\x7f`89x\xd1 \x9d<\xaf{\xa0\xbbw\x13#\x0f\xd9&\rk5c\\A\x9a\xa5P\xcaY\x02j\x98j\xd1\xd6"
            b"\x1fI\xffAP\x0bk\xe7\xa9\xcb\x1b5\x06\xa0\x84\xeb\xe8\x8c\xb5\xdb\xd7\xe5RY\x99T\r\xcc"
            b"\x13\xc09['\xdb\x88\xf2\xb6\xea*M\xcf`\x9d\x83\xa23oU\xd6\x90\x8b\xe3e\x03{\xacH'\"\xf0"
            b"\xd6t\xc90A\xd1L\x82Eu\xbe\x97[+P\x16\x19\x1c\x1f\xd8-F[h\x00}&\xfa\xc4\xfbi\xf6\xae\x0e"
            b"\xa6K\x1b\xe1_\xee\xf4\xean !M\xee\xeeT\xa5\xb5<(}\xd5\xb3\xd3\x8e\x9e\x11\xb0\xfc\xd8)"
            b"\x1f\xd1&\xab\x9a[\xea\xc4,\xb7{[x\xf6E\xad';V\xd0\x90\xdf\xe31W\xb6\xed\x81\x08\xc3\xe8"
            b"\xf0\x14\x12\xc4$i\xc98\x8a\x16IQHq\xdb\xaa\xab\xa9\x90j\x9b\xe4\xfdn\x11\xe0\x08~@"
            b"\xa62B2z\xae>P1\xe1|!\xeb`q\xc0\x86\xb6\xad|\xe0\x03z@}\xac\xccf@1z\xbb\xc2\n\xd1\xb7"
            b"\x13D\xdcb\xc9\xc4\x8c\xb9\xad\xae\xce\t\xac(\x1f8\xadd\xda\x80\xe5Z\xc0$\xb1\x1a"
            b"\xa3PS\\\x19\t\x18}\x92\xaa\x85\xf2t\x08\xf5\xe5\x9a\xfa\xd9!\xcd,\xd2\x04\x9c\x01"
            b"\x96Mq\x8d\xd1ea\xc7\x9f\xd1\x02\x84\x02\x87\x81\xce\xa2@\x04\xc1\xc6[q]\xcd+E\x19i"
            b"\xdc$>=\t!\xbb/}\x90\xc3V,\xa28\xd6R\xd5\xe0JT)\xeczbX\xfd\xc0_0m\x93L(;D\xb1t\xe0\x14k"
            b"\xfa)j\xb9rA\xcbg\x95\x9c\xe7\xc0\xa4\xfaIk\xea\xe8h\x16Ec\x08Ik\xb3\x1f\x1d/\xe9J\xd9"
            b"\xbe.Q\x0f\x83\x8e\x07\xac\xc0\xd9\xbf\x16m\xc8Xn\xfbhNSL\x9e\xf5m5R6U[\xf3\xe6\xb2&"
            b"\x84w\xfe#YVz+\xda\x9f\xab\xa7.\x8c\xa5\xe5\x95l\xa1\x9e\x0e\x8e\xee\xa3\x18\xe4\xbe"
            b"\xfb.\x8d\xdb\x84\riRa\x83*S]y*\x0b\xbc\xcb\xafzg\xb6\x90\xd1v\x86\xe0\xc7E\xe3\xc7"
            b"\x076\x99\xdf\xd5\xae\xd5\xe1\xd6\xb3\xa8V[\x88\xa6\xa6\xb8\x90\xab-E\xe2G\x93\x81M"
            b"\xaaTR\xd9V'\xb5\xaa\xb9\x06Kh\xb5\xab\xaeE\x84nC\x9b8\xba\xa1\xc6p\xf0\x9c'\x9e:\x8a"
            b"\xd4\xb91\xb4n\\\xbd\x1b_\xf36\xac{C\xa6\xdb\xb8DNW&+RI\x07\xd4\xc7\x93\xcb\xb8\x92"
            b"\x19-\x9ba\xd2\xe927\x15\xc2\xffn#:\xb4Q\xa4\x8cFJi\x98\x9c\xbaH\x84\xd9\r\xb7Q\x98\xe5"
            b"\x98\x7fN`\x96\x12'\x84\xd9=\xb7Q\xb0\xe5C\xcf<qB\x98Mq\x1b\xc5\xb3*\x11z;Zs\xc7\xd0"
            b"\x90Q]\x84\xd77\x9d\xec\xc8)\xce\xb7\xdb\xe5/\xaes\xe1v\xa1\x83\xfd\xe9\xb8]t\xb4[\xc3"
            b"\x04\xbb5s\xefv1af\xc7n\x12lt\xe6w\xb5\xeb\x83x\xf4,\xa6\xd5f\xa2C*f!C*&:\xa4b\x82\rK"
            b"\xd3\xc9FO\xd3\xc9\xb6\x1eG*\xf9\x80\xfa\xb9r\xad\x82d3\x8ek\xc5\xc4qX\x980\x87e\xbe\\+&"
            b"\xccn8\x8d\xd80\xcb1\xff\x9c ,!N`\xe3\x84[\xecb\x08\xb7\xd88\xe1\x16\x1b\xcf\xaa,~';"
            b"\xe6\xf6y=h\x08`0Xl\xcd\x1bu\x80\x15tJ\x11tB\xe1\x9e\x18\xe1\x83\xdf\x80IV\xf1\xd4\xac"
            b"\x0c\xe7\x9e+\xc7\x191\x16Ek\xcf1\x02\x81\xfcA\xe6\xed8y\x02\x9d\x9a{l\xef\xb4\x0f\x94'h"
            b"\xcf\x89\xf2\x04Sw\xa4<AOT6\x1b\x82\x8e\x93\x15=m\xa8i\x9f\xf3d\xabb:\x07\xcaQ\x07\xb33"
            b"\xe0mt\xb9\xc4\xef$*\xf0\xe6U,\x9e\xb4O\xe0'\x14\xd9\xba\xc85A7.\x06\xf3\x84\x193g\xe2"
            b"\xb2`\n\xa82\xc7\tZ\xc6\xc0\xfe\n\x03\x98\x92bx\x86\x14%\x9a\xe2YJ\x10XIl|\x9d\xa6\x8f"
            b"\x1c\x13\x82\x1c\xc2\x8dcEI\x129\x8e\xa7\x05t\xe3\x8c\x9b\xb3\x95\n\x04\xb2\x90\xed\xbd"
            b"\xa8[\xee\xd1t\x1d\xbd\xc8\xe1\xb9Z\xf9Z\xd2z\x13\nx\x8c\xd1Wi\x16@\xf7-\xc4\xa5\x94"
            b"\x99m\x0f\xceF\xb8\x1b\xdeG\xfc(2\xaa\x9f\xd0\x9e\x1a\xee+\xa27\xf7\xe2\x0e\xdbh\xd0"
            b"\x1c\xabs\xb7{:\x8e\x9d\xa9\xe9\xda\xb6:\xf8\xa6&-T=b\x08\\\xd8-\x96\xcac\xbfV\x88eD659"
            b'\xa3\x8e\xc9To\x08@\xc8\x05\x17\xefl\xf1\x15\xa4\xa0\x83\xb2X\x1b"\xc1\xf8\xd1\xc1\xf8'
            b'\x85_\xa2\xf1"\x88\x97\x83\x0b\xd0\x98\xc5qU\xc7\n+\xad\xe8\xa6jd\x95B\xb5\x02\r\x99PN+'
            b"\xb4\x119\xc5\x8c\x9a\x92A\xa7c\xcc\x02\x85\x80\x10[\x04wVy\xbb\xd4\n\x85+\xa8\x03"
            b"\xb1\\\xec\x03\xc1w\xeaJQ\x06\x94e\xf4>\x11x\xf1\x01\rc.WL\xbcm\xe2\xf3U\x9bh\x1a\x04"
            b"\xf3\x95\x17ad&Sw9\xd1\xfb\xe0\xbb\xabh 2\xc5\xb2$Cr\x95\x87\xc7o\x85\x90$\xc3\x89\xa4"
            b"\xa7\xac\xfe\xean\xd0,\xbc\xd6lWy2\x92$\xe1K\xe4\xde\x08E\x0b\x82_\x80\x92\x91M\x9c\x19"
            b"\x05\xf1\xadn\xeb\x9d&k\xd9\xc3W:\xd6\xf2\xc5\x1a\xda^:a\xae\x14y&;\x91\xcd\xccP\x8f7"
            b"\xb65k=5\x14B\x9b\xaa\xc12W\xbbU\x1b\xd2\xd4Z\xfd\xc9\xf2\xae8\xb2\x1e9\xf4|\xcf\r"
            b"\xfd\t\x99\x9b\xec\xf4\x1ct\x0c\xee\xce\x7f2H\x8a\x1a\x9b\x7f\x033\x89\xdf\xedt\x88\xda"
            b"\x00\"\x15\x82\x86q\x8aM'\xb3l\xc6\x9cO\xd4\xd6a\x04^\xd6x=\x8e\xd6 \x98^\xc2\xd4\x88"
            b"\xe1\xee\xf5Q\xaa!p\xf0\x18:&\xba\x9b\xaa\xb5PFG\xb3\x055\xc6\x99J\xc06\xa9\xfd~O\xc3"
            b"\xe7\x81\xfeN\xd5|\x9e\x07Z\xabc\x02#b\x12\x84\x932`\xf6\x98\x8e9\xfb\xf5\xb8FI \xcd"
            b"\x1f\tf\xb4\x15\xec7y\xdbD\x89K\xe0\x98/\xe8\x8c\xaf\xa2\rh\x7f\xd3\x11O\xbb\xcd\xd9"
            b"\x01`\x1a\xbd\x02\xaa\x14LOB\xb2\xda'\x93\xcd;\x89\xd5\x02n\xb1F\xc5'\x16\x114_\xf6\x8f"
            b"\x11\x86\xd8\xaf\xc4\xfbsz\xd5l\xc0\xe1(\x98Y\xb3\x1c\xe2u\xc4r\xa0\n2L\xda\x08\xba\x1eh"
            b"\xa7\xa6XE\xf7\xfe\xcd,\xe8Xh\x11C\xcd\x96\x83\x9bE\xce\xc8\xc2\x17,\x022\x0ci\x8f\xa1"
            b'\x08\x11\xc7\xc0\xf1\xe2\xbf\xfa\xe0\xdf\x83-\xd3\xd24"\x067RH\xfa\xbd\xbc\x97\xf4\x0b'
            b"\x15\x92>AE-\xca\xde \xa1^F\xa3\x82\x03\x9b\xb7\xddh\xc0\x8alkm\x7f\xa4G\x1e+\x08\x88y"
            b"\xf4\xe53\x84}\x83\x9a\x9cQ\x90\x1cN\n\x8b\x10i\xda\x08\xc0\xca?\x9a\xae\rz]\x04\x10"
            b"\x1d]\xbc\xaa@B\x02\xddX\xfb\xb6\xf5\x88\xd8\xd4\xa1f\xb4\xd5\x8e\xd9\x9ai\xe4\xf0\x81"
            b"\xe1I\xb2\xb1\xfdnu\x0c\x91\xc5\xdeB\xa9\xd9\xdb\t\xdcf\xc7\xfb7L<*\xf9\x8dS\xb5G3\x8d"
            b"\xed\xefF.\xeb4w\\\x97\xe8\x8ek\x83\xfd\xc5\xdaf\xc5\x8c\x1bt\x15\xc4m\xe0\x7f\xe7\x0f="
            b"\xb1w&co\xd5\xd8C\x06]h\x9a\xed\xbd5<\x98\xff\x9d\x99F\xf7\x81\x02[5\x12\xa7\xc5\xba\xd0"
            b"\xd2h\xac\xe6yE*\xa8\xc1\xfc_\x0f\xaa\t\xdbd\xd0aQ\x14\x0e$\xc8\xb4\xc27\x0fU\xfc}\xee9"
            b"\xa4\xca\xcc\xf7\x04|ia\xf3s}L\xde@\x94\x18\xe8\x96\xda\x9d\xd7F\xbb\xb1\xbdU_I\xb4:\xa5"
            b"\x82Lt\xa8\x13\xe6\xe7\xee\xd8\xfdy\x174\xd2\x16\x0fX\xd6\x93\xb8`\xcb\xd00\x01\x91\xda"
            b"\x18D8I\x94/K\x85\xd5,\xa8\x8aN\xa4\xddC\xe5\x141\xe1\xed;\xd8V[]7\xdc\xe5k\xe9\xac"
            b"\xbayR=\xab9\x02B\xceG\x1b;\x15\r}\xa3?n\xc6\x80\x85\xc6f\xa6\t\x1cft\xd2\x19y\xbe\x19"
            b"\xe9\x9fL\xeb\xa42\xf4|r&\x91m\xa3\x1b\t\xb3q\xe0\xd8V\xcb^>\xc9\x88\x92uY\x88\xaa"
            b"\xa7Us8Y\xb3\xf3\x10\x1elG,Q\x8c\xdd\x84X\xa1r]\xe73\xe1\xbd\xb0\x19\x85\xdb\xcd\x867\t"
            b"\xaax\r\xc2\xec\xe8 \xd5Q\xfc\x8c\xa9\x19\xdb\xcb>\x16\xaf\x1a\xafY\xb1yi\x94\x13\x0f"
            b"\xa7,\xc6\xadBm_=Y\x1a\xb6~`\x00R\xea\x8e\x94\xae\xb4\x11\xf6\xd7L\xc5D9\x14\xd8\xa9"
            b"\xe9\xb9\x8c[V\x97\xb0\xd9\x9a\x87'\xcb\xb8\xf3\xbd\xea\x7fh\xc0\x8d\x8d\xf6lZ7,\x17\xd7"
            b"\xcb\x06V\xb2`\xc5\x84(a\xa4d%\x92\xbe2\xd2\xd0f\xd41\x1f+\x0b\xa5>&\xd6NJ\x85S^Q\xcd"
            b"\x94W\xcd\x94W\x91)\xaf$.4\xe5Umu\xc3)\xaf\xe6\xff\xad\x81d-\xc0\x9c\xbe3\x10s\xb0\xd9"
            b"\xb8\xff\xec\xef\xc6\xce\x90q\x02X'\x90y\"3\xa6E\xe4L\xf3a\xa1\xc5\xbe\xf59\x83\xd7\xbd"
            b"\xf1\xde:\x1d\x87\xd1\xac'\xfc41,\x93Z\xc4Y\xa2\x1d~\xd7\x1f\xea5\xb2\x01\xe1{\x17qN\xde"
            b"\x8d\x9f\r*\x05\xe56\x8b\xa4\xd4\xe2\xc0c\xc6\x0b3\x9b\x99\xf2\x16\xfe\xf4\xa0\xf9zr}"
            b"\xfb\x99\xbf\x94\xea\xa3\x10\xacg\xb1\xa5~\xc1\xda\xcb\xff\x82\x9e]\xe9\x8f\xe1\xdc\xbf"
            b"\x85\x8a\x07\xf7?\x88\xb1+\xfd\x8eL\x16`\xa9\x97J\x96\x1fL3\xff\x93\x1f\xbb\xd2\xffj"
            b"\xdb<-\xb5\xff\xa52\xab\x92\x8eTKs\xfe\x82q\xe3\xe9!\xe2\xab\xa4X\xd7Jc\x9d\xce\xc5\xd5"
            b"\xc3\x8dh\xe2i\xe8\xe2F\x8e\x8ac\t\xe9\xf4\xc5\xd4#\xa8\xc1\xc7\xbeqE5\xbe\xb0\xc6\x11"
            b"\xd7p\x81uy?\xe8\xac\xce\xae\x0e\xb241\xc46\x86\xe0\x86\x8b\xae\x8bD\x90\x0e\xb4\xab"
            b"\xfd\xe9\xb9@l\x10|\xb8\xbf\xa8\xd9 \x08m\xbb:\xfa\xfe\xc0<\xb0AP.\x0e\xbb:\xca\xbeD"
            b"\xea\xf18\x9a\xbc\xc1s\xe5\xa0n\xa2Z4\x1d\xf1\xa5\xe1\x88/\x95DAX@\xc2| f\xda\xb7gf"
            b"\xc1;c\xfc\x8d\x90]\xe9g\x82\x16`\xa9\x97J& L\xb3\xb0\x98\x8bY\xc8\x98\x8b\t\x8b\xb9"
            b"\x18?qj:\xe2\xf5O\xd3\x11\xaf\x7f\\A\r~/}.=\xb0`q\x8d\xe7\x811\xe1\xde\r\x134\xab\xf9"
            b"\xf4\xc0\x18\x7f\x0b\xe2T\xb3\xfe\x8b\xb3@l\x10\x84\xeb\xe2f\x036<\x1ec\x17C<\xc6\x86"
            b"\xc7cl\x94}YJ\x8ex\xec\xed\xf8YM\xfc\xd6\x80&\x8e>J\x0b>\r\t>\t\t\xc9\xe5\x16r\x002k\xc3"
            b"\xfa\xe7t\x0b\x19y\xda\xffCM3!\\3!\\3!\\3!\\3!\\(P3!\\3!\\\x03\xb0\xcd\xd7\x13g%!\x1c+4"
            b"\x1e\x91\xc7\xe8\xdc\xed\x9e{-%\x84cg\xe1\xb5\xcb9M\x08\xc7\x8a\xcd\x84p\x0b\x96\x10\x8e"
            b"\x9dNz\x07\xe7i&\x84[\xc8\x84p\x1c\xd9L\x08\xd7L\x08\xd7L\x08\xd7L\x08\x178t\x9c3\xea"
            b"\x88\xb3\xe9\x85J\x08\xe7{9\xed5\x96\x11.\xe0\x92\xd2RJ\t\xc7Q\xb3\x1d\xf94S\xc25S\xc25S"
            b"\xc2-dJ8\x8e\x9eF\xcc\xd0L\t7\xc3\x94p\x1c\xd3L\t\x17\x9c\x12\x8e\x8b\xfd\x7f\x08\xf8"
            b"\xee\x0c\xe2P\x98o\xe4\xf8a\x1eS\xc2\xb1|<*5S\xc2\x11\xd3\x82\x7f\xcd\xef\xb9\xceUJ86"
            b'\xe8n\x8f\xdb x\xebt\xaeR\xc2\xb1A\xf7c\xe6"%\x1c7\x037f\x89\xa5\x84\x0b|c\xe25\x9d'
            b"\x12.\xf0\x15\x8e\xa5\x99\x12\x8e\xab\x0f!f/%\x1cW\xcbA\xb3\x91\x12\x8e\xab^\x80\x19"
            b"\xa7\x84\xe3\xbc\x0b\xd5L\t\xb7HO\xab\x9bg\xd5\xcd\x94p\xcd\x94p\xcd\x94p\xcd\x94p\xcd"
            b"\x94p\xf5x5S\xc2-\xf2\x94p\xa9T\xaa\xe5\xd1w\x0e^\x98 \x13\xf0\xe4v\xb1\t\xfby\xa4\xfas"
            b"\x19\xf1\xc0g\xfe\xe9\xc0\xceg\xbe\xf1\xfd[?\xf7\xef\xbb\xd9_\\w)\x14\xfe\x80X\x96H\x9c"
            b"\x9cX\x96\xf8\x9f\xee/\x1d|\xdb\xb77n\x1eZ{\xf0\xcf\xa9\xb3\x07\xee\xbed\xdf\x96G>\xfe"
            b"\x95\xe7\xa5\xf4\xd4m\xfa\xd4\x07~z\xee\xd4\xbd\xfd\xa5_\x1d.\xfc\xe4\r\x9d\xef\xf9\xf0M"
            b"\xd7L=\xb3wm\xfb\x9b\x0f\xbd\xe7\xe17^\xdb%\xbeu\xdb\xde\x1f\x1e\xf8\xe8\xd4\xd7\x1f"
            b"\xe8L\xdc:xs\xa7y\xce{G\x8f<\xf2W\xe7nI\xfc\xe1\xa4\xc4u\x1f\x7f\xfa\xd0\x8aS\xca\xfb"
            b"\xfe\xf3\x13S\xb7>\xba\xe2\xbdS\x853?\xcc\xe7\xdf\xb5\xe7\xde\xb7^\x9b\xa7F\xa7^\xd7"
            b"\xb9*q\xe4\x11\x89?\x1c\xfdb\xf9S\x97\xfe\xf0\xe5=SE\xfe\xf6\xe7\xb5\x87\xee\xed\xec"
            b"\xbd\xf3\xf6\x16\xe3\xee\xc7\xf7\x12\x89c\x8fH\xbc\x9ah{\xe3\xbf\x9c|\xd9i\xdf\xea\\"
            b"\xf9w\x07\x96\x7fI8y#*\xbd\xea\xc0\x9d\xdf2[\xdew\xf6\xaa\xc4\x1b\xaevf\x1c\x8f\x0c"
            b"\x1f`\xbfy\xce\xca\xb5\xab/\xd8\xd3\xff\xec\xf2\xde\x97w\xa5\xaa\xc9\xf0\xb5\x07\xbb"
            b"\xde\xd8\xbdr\xef\xda\x15\xdf\xfe\xec\xab\xa7~\xfe\xa3]\xf7\xfc\xc7\xe4\xaf\x8b\xc7"
            b"\xdc>\xd5\xfa\xbd\x1f^\xf0\x83C\x8fv\xfe\xe9C\x93\x89\xeb\xb7\x11\x9d\xb7=|\xdf\x9e\xc7"
            b"\x8f\x1cZ{\xcc\xb2\xc9\x8f\xb6\xf3\xdb\xf7\xfd\xf8\x1d\x07\x7f{\xe8\xfa\xeb\xa6\x1ee\x84"
            b"\x9bJ;\xcc{\xef\x1e\xb9\xe4X\xee\x81\xd7unI\xbctbB/\xae\x19\xb9\xe3\xe6\x89\xaeC\xcbZo"
            b"\x19\xa4\xee\x98zL\x7f\xf7\xdbN\xbd\xe2\x81\xceS\xbe\xde~\xe6\x11\x85VL\x85W\x97\x1f"
            b"\xdc\xba\xe7\x9e\x0f\x9e\xbey\xea\xd5\x9f~\x91\xb8\xf1]\xdf\xb9\xf7\xe7\xd4\xad{\x8f"
            b"\x7f\xfaT\\\xfb\x97\xc4\xe1o\xb6\xf5\xdd\xf5\xf9\x7f\xeb\xbch\xf3\x11\xc7\xed^\xfe\xe6U"
            b"\x89\xa3\xa0\xf8\x9a\xd3\x7f\xff\xd4K\xca\xb3\xc7\xdf{B\xe2\xf8#\x12\xcb7\x06\x93\xe1G"
            b"\xd5\x9f\xcb\x88;\xff\xd8\xbbk\xf3\xceK\xfe\xfe\x98\xaf|\xcf|\xee\xd8c\xee\x80\xc2\xefW"
            b"\xc8\xb0\xe5\xa5\xcf\xbc4I\xdc\xb3\xff\xa8\xeb\x9e~\xf2\xc5\xcc\xe1\xae]\x87\x9e\xd8\xfc"
            b"\xee7\x10]\x9f\xbba#\xf3\xe3\x0f\xdc\xbfF\x98\xfc\xe4\xc9W\xb7\xf4N=\xf4\\\xdf\xa9\xe7_"
            b"\xbc\xe2\x1d\xeauW\x9c\xb2J\\\xb1\xff\xf0e\xdfX}\xe9\xd9#]\xcb~\xf9\xfc\t\x1f\xfa\xc2g"
            b"\xa7^:\xfd\xe2\xff\xbd\x9bi\x99\xc2d\xb8\xe6\xd6U7\x0c\xee:y\xed\xcak\xcf\xd0\xee\xbc"
            b'\x8d\xeb\xbas\xdb\x8a\xcb&v\xdd\xb5\xe6\xd3\xbf\xfb\xd9\xde\x93~"\xd9\xcc\xf0\xec\xd5'
            b"\x0f\x7f\xe7cGo\xef\xda\xb3\xe3\xc3\xab\x7f\xfe\xfa\xfb\xa6\x1eY\xb6\xfd\x8c\x8e_\xbeu"
            b"\x8d\xcd\x0c\xa3\x17\xc8/^\xb1\xfa\x1bSw\xff\xf9\xb9\xcf]S\xea\xb1\x98\xe1e\xed\xf0-\xdb"
            b"\xd2[\x1ag\x86\xa7\x9ey\xb1\xffM-\xfc\xcdo\x1e_\xbe\xe6\xb7\x97w\xe8\xd5\xcc0\xf8\xa9"
            b"\x9dC\xf2\xcfV\xbcc\xcdq]\xc6)\xe3o\xdf\x7f\xc7\xd6\x1bwo\xbc|G\xd7\x0f\x8e=w\xdd\xbe["
            b"\x0eN=\xf9\xdd\xd7\xff\xa8\xe7\x17\xc3S\x87V\xd2\xff\xf5\xf8W\xbf\xb5\xff\xbb\xb7\x9f"
            b"\xf8\xae\xdf}\xec\xe8\xb5\x97m+\x1f~\xec\xdd\x1b\xba.\xff\xab\x03/o\xbe\xff\x93k\xee9"
            b"\xf7o\xff\xe1\xb6\xed\xbcE\x85\x87O<\xed\xe2\x8b\x7fz\xe2~C\xef\x1c~\xfe{F\xd7I\xc5\x03"
            b"\xef\xbfy\xfbCS\x89Uw}\xea\xcb=\x97\xda\xcc\xf0\x9b\xa7\xef{\xec\xd5\xc7\x96\xed\xcb="
            b"\xb9\x95\xbc\xe3\xfa\x07\xd7\xbc\xa5kx\xf7\xdem\x17\xda\xccpn\xf7'>{-\xf7\xe0\xd4[\xa6"
            b"\x0e\x9d}\xe8\x86\r63\xac&\xc8\xb6\x936\xf7\xac\x89d\x86\x955\x9f\xc4\x81\xe3{\x96\xaf"
            b'\xbe\xef\xa8\xd7}\xf5\xd9\xff\x1e:\xfa\xfa/\xb6B\xd9\n"a\x91\xe1\x85\xa3\x9f:\xff\xb8-'
            b"\xc6??\xbb\xff\x81\xfb\x88?~e\xf9S\xfbCI\\\xd7\xf7\xe1\xcb\x8f\xbb}k\xdf\x86\x876\x1f"
            b"\xfc\xe3\xf7\x9f\xd8w\xb0\xaf\xbao\xbb\xcf\xb5\xce\x18\x8d\xf5\xbdD\xf0^]\xf3I\xf4"
            b"\xa7_yry\x7f\xc7\xa9_\xbf\xf6\x0bW%\xce\xbe\n\x95\xafD}\xbfp\xe22\xe3\x99\xdb\xceO\xbc"
            b"\xb06q\xd4\xed?\xef\x7f\xe6\xf1}\xaft|\xa7\xd0\xdfr\xc6\xda\xado?\xed(\x03\xca}\xfb?\xbd"
            b"\xe6\xf3\x84\xde\xfc\xea\xb3F\xbaN\x18\xe4\xdf\xbf\xe7\xfd\xad\x9f\xfe\xc2\tP\xf6&\xf4"
            b"\xcf\x0b\xcb\x0e_t\xc2\xd7\xb6h7\xee\xeb\xbc\xeb\xfe\xfbo\xb8\xfa#]\xf1\xf0\x1dN\xfeX"
            b"\xf9\xc3\x9a\x7f}\xe2\xaf\x1f\xf9r\xe2\xbd\xb9\xcc\x8a\n\xbe\xaf\xfc)\xf7`\xe1/\xbf["
            b"\xfbk\xe2y\xa1\xe3\xd7\x8fw=\xc4m\xff\xcd\xef\x97\x9f\xb1\xff\x82s^x\xe54(\x8f\x87\xef"
            b"\xca\xe7\xfe\xf1\xe3\xaf\xdct\xc5\x83O|z{\xf6\x83/\xbfxb\x05\xdfcv\xf4g\xff\\\xb8\xb1"
            b"\xeb\xca\x81=/\xdcs\xf5G\xf65M]0\x9b5M\x1d~\x9a\xa6\xce\x97\x19\x9a\xa6\x0e=K\xc5d,U"
            b"\xbc\x9b\xa6.\x02\xdf\xf8\xa6\xee\xac\xee\xa1\xde~\xe2\x9d\x83\x1bz.$\x06\xd6\xf7\xf6]"
            b"\xdcr\xce%\xdd\xeb\x06\t\x084[\x92\xc9dK\x92`9\x91#\xd1'OQ\x0c\xfed\x04\x1e\x7f\xf2\x8c"
            b"\x88?%\x92\xb2>%\\/\x90\"\x87?)A\xc0\x9f4/\xe1O\x86\xb7\xcaY\xd6\xfa\xe4X\x0c'\xf0\xb4"
            b'\xf5)\x8a,\xfa\x14)\x16\xc3\x89,\x89\xe1D^\xa0\xf1\xa7\xc0[\xf5"\x87\xc7\x17%\x16\x8f/'
            b"\xa1\xf7\xfd\xd1'\xc5\xe0r\xf4\xee\x1c\xfed(\x1aG\xae\xff\x07\xe4\x88\xd7\xd5\xd4\xc1\x00\x00"
        )
        f.write(_x)
        del _x


def gencal(cstem, rng):
    """
    Writes a bunch of dummy calibration files.

    Parameters
    ----------
    cstem : str
        Stem for the calibration file names.
    rng : np.random.RandomState
        Random number generator.

    Returns
    -------
    dict
        Dictionary of the calibration files.

    """

    N = 4096  # full size of SCA
    N_ = 4088  # excluding reference pixels
    dtrim = (N - N_) // 2
    x, y = np.meshgrid(np.arange(N), np.arange(N))  # pixel positions, for making maps
    ngrp = len(read_pattern)
    t = np.zeros((ngrp,))
    for j in range(ngrp):
        t[j] = dt * np.mean(np.array(read_pattern[j]))
    print("t =", t)

    ### BIASCORR

    # this is trivial right now, just to check schemas
    asdf.AsdfFile(
        {
            "roman": {
                "data": np.zeros((ngrp, N_, N_), dtype=np.float32),
                "t0": np.mean(np.array(t[1])),  # in seconds
            }
        }
    ).write_to(cstem + f"_biascorr_{tag:s}_SCA{sca:02d}.asdf")

    ### DARK

    dark_slope = 0.005 * 10.0 ** rng.normal(loc=0.0, scale=1.0, size=(N, N))
    _trim(dark_slope, dtrim)
    bias = np.zeros((N, N))
    bias += 13000 + 200 * np.cos(2.0 * np.pi * x / 256.0) + 100 * np.sin(2.0 * np.pi * y / 256) ** 3
    asdf.AsdfFile(
        {
            "roman": {
                "data": np.clip(
                    (bias[None, :, :] + dark_slope[None, :, :] * t[:, None, None]).astype(np.float32),
                    0.0,
                    65535.0,
                ),
                "dq": np.zeros((N, N), dtype=np.uint32),
                "dark_slope": dark_slope.astype(np.float32),
                "dark_slope_err": np.zeros((N, N), dtype=np.float32),
            }
        }
    ).write_to(cstem + f"_dark_{tag:s}_SCA{sca:02d}.asdf")

    ### GAIN

    asdf.AsdfFile(
        {
            "roman": {
                "data": np.clip(1.5 + 0.03 * rng.normal(loc=0.0, scale=1.0, size=(N, N)), 1.4, 1.6),
                "dq": np.zeros((N, N), dtype=np.uint32),
            }
        }
    ).write_to(cstem + f"_gain_{tag:s}_SCA{sca:02d}.asdf")

    ### IPC4D

    K = np.zeros((3, 3, N_, N_), dtype=np.float32)

    # simple IPC kernel
    K[0, 1, :, :] = K[2, 1, :, :] = 0.015  # horizontal
    K[1, 0, :, :] = K[1, 2, :, :] = 0.013  # vertical
    K[0, 0, :, :] = K[2, 2, :, :] = K[0, 2, :, :] = K[2, 0, :, :] = 0.002  # diagonal

    # eliminate dependence on the edges
    K[0, :, 0, :] = 0.0
    K[:, 0, :, 0] = 0.0
    K[-1, :, -1, :] = 0.0
    K[:, -1, :, -1] = 0.0
    K[1, 1, :, :] = 1.0 - np.sum(K, axis=(0, 1))  # normalize to 1

    # write
    asdf.AsdfFile({"roman": {"data": K, "dq": np.zeros((N, N), dtype=np.uint32)}}).write_to(
        cstem + f"_ipc4d_{tag:s}_SCA{sca:02d}.asdf"
    )

    ### LINEARITYLEGENDRE

    # general information
    Smin = 5000 + 500 * np.cos((x + 3 * y) / 100.0)
    Smax = 56000 + 10000 * rng.uniform(size=(N, N))
    Smin = np.clip(Smin, 0.5, 65534.5).astype(np.float32)
    Smax = np.clip(Smax, 0.5, 65534.5).astype(np.float32)
    Sref = (Smin + 300 + 100 * (x % 2)).astype(np.float32)
    pflat = (0.95 + 0.1 * (x / N - 1) - 0.2 * (y / N * (1 - y / N))).astype(np.float32)
    pflat[:dtrim, :] = 0.0
    pflat[-dtrim:, :] = 0.0
    pflat[:, :dtrim] = 0.0
    pflat[:, -dtrim:] = 0.0

    # now build a non-linearity table
    data = np.zeros((4, N, N), dtype=np.float32)
    data[2, :, :] = 20 + 180 * rng.uniform(size=(N, N))

    # now the derivative at Sref is given by:
    # z = 2*(Sref-Smin)/(Smax-Smin)-1
    # deriv = (data[1,:,:] + 3 * data[2,:,:] * z ) * 2/(Smax-Smin)
    # so to make this equal to 1:
    z = 2 * (Sref - Smin) / (Smax - Smin) - 1
    data[1, :, :] = (Smax - Smin) / 2.0 - 3 * data[2, :, :] * z
    # and now the value at reference is zero
    data[0, :, :] = -data[1, :, :] * z - data[2, :, :] * (1.5 * z**2 - 0.5)

    # write
    asdf.AsdfFile(
        {
            "roman": {
                "data": data,
                "dq": np.zeros((N, N), dtype=np.uint32),
                "Smin": Smin,
                "Smax": Smax,
                "Sref": Sref,
                "dark": dark_slope.astype(np.float32),
                "pflat": pflat,
                "ramperr": np.ones((2, N, N), dtype=np.uint16),
            }
        }
    ).write_to(cstem + f"_linearitylegendre_{tag:s}_SCA{sca:02d}.asdf")

    # check the mapping
    Slin0, _ = ipc_linearity.linearity(Sref, cstem + f"_linearitylegendre_{tag:s}_SCA{sca:02d}.asdf")
    Slinp, _ = ipc_linearity.linearity(Sref + 5, cstem + f"_linearitylegendre_{tag:s}_SCA{sca:02d}.asdf")
    Slinm, _ = ipc_linearity.linearity(Sref - 5, cstem + f"_linearitylegendre_{tag:s}_SCA{sca:02d}.asdf")
    Slin_der = (Slinp - Slinm) / 10.0
    del Slinp, Slinm
    print(np.amin(Slin0), np.median(Slin0), np.amax(Slin0))
    print(np.amin(Slin_der), np.median(Slin_der), np.amax(Slin_der))

    assert np.amin(Slin0[dtrim:-dtrim, dtrim:-dtrim]) > -1.5
    assert np.amax(Slin0[dtrim:-dtrim, dtrim:-dtrim]) < 1.5
    assert np.amin(Slin_der[dtrim:-dtrim, dtrim:-dtrim]) > 0.99
    assert np.amax(Slin_der[dtrim:-dtrim, dtrim:-dtrim]) < 1.01

    ### MASK

    mask = np.zeros((N, N), dtype=np.uint32)

    # reference pixels
    mask[:dtrim, :] |= 2**31
    mask[-dtrim:, :] |= 2**31
    mask[:, :dtrim] |= 2**31
    mask[:, -dtrim:] |= 2**31
    # dark map
    mask |= np.where(dark_slope > 0.25, np.where(dark_slope > 12.5, 2**11, 2**12), 0).astype(np.uint32)

    asdf.AsdfFile({"roman": {"dq": mask}}).write_to(cstem + f"_mask_{tag:s}_SCA{sca:02d}.asdf")

    ### PFLAT

    asdf.AsdfFile({"roman": {"data": pflat, "dq": np.zeros((N, N), np.uint32)}}).write_to(
        cstem + f"_pflat_{tag:s}_SCA{sca:02d}.asdf"
    )

    ### READ

    # this is just a toy case to test out the code
    medband = np.zeros((4096, 128), dtype=np.float32)
    stdband = np.zeros((4096, 128), dtype=np.float32)
    medband[:, :] += 29000.0
    stdband[:, :] += 4.0
    for i in range(16):
        stdband[256 * i, :] = 5
        medband[256 * i, :] += 30
        medband[256 * i + 1, :] += 15

    amp33info = {"valid": True, "med": medband, "std": stdband, "M_PINK": 0.8, "RU_PINK": 1.0}

    asdf.AsdfFile(
        {
            "roman": {
                "anc": {"U_PINK": 0.4, "C_PINK": 0.8},
                "data": (6.0 + 5.0 * rng.uniform(size=(N, N))).astype(np.float32),
                "resetnoise": (25.0 + 5.0 * rng.uniform(size=(N, N))).astype(np.float32),
                "amp33": amp33info,
            }
        }
    ).write_to(cstem + f"_read_{tag:s}_SCA{sca:02d}.asdf")

    ### SATURATION

    asdf.AsdfFile(
        {
            "roman": {
                "data": np.clip(Smax - 50, 1.5, None).astype(np.float32),
                "dq": np.zeros((N, N), np.uint32),
            }
        }
    ).write_to(cstem + f"_saturation_{tag:s}_SCA{sca:02d}.asdf")

    ### DARKDECAY

    dectab = {}
    for k in range(1, 19):
        dectab[f"WFI{k:02d}"] = {"amplitude": 0.3 + 0.1 * np.cos(k), "time_constant": 20.0 + k}
    asdf.AsdfFile({"roman": {"decay_table": dectab}}).write_to(
        cstem + f"_darkdecay_{tag:s}_SCA{sca:02d}.asdf"
    )

    return {}


def forward_backward_lin_ilin(linearity_file):
    """
    Forward-backward test for linearity routines.

    Parameters
    ----------
    linearity_file : str
        ASDF linearity calibration reference file.

    Returns
    -------
    None

    """

    ymin = 260
    ymax = 262
    xmin = 140
    xmax = 143
    dy = ymax - ymin
    dx = xmax - xmin
    with asdf.open(linearity_file) as F:
        print("Smin", F["roman"]["Smin"][ymin:ymax, xmin:xmax])
        print("Smax", F["roman"]["Smax"][ymin:ymax, xmin:xmax])
        S = F["roman"]["Sref"][ymin:ymax, xmin:xmax]
        S[:, :] += 5000.0 * np.linspace(0, dx * dy - 1, dx * dy).reshape((dy, dx))
    Slin, dq = ipc_linearity.linearity(S, linearity_file, origin=(xmin, ymin))
    Sfwd, exflag = ipc_linearity.invlinearity(Slin, linearity_file, origin=(xmin, ymin))

    print("coefs:")
    with asdf.open(linearity_file) as F:
        print(F["roman"]["data"][:, ymin:ymax, xmin:xmax])
    print("signal [DN_raw]:")
    print(S)
    print("inverted signal [DN_lin]:")
    print(Slin)
    print("recovered signal [DN_raw]:")
    print(Sfwd)
    print("flags")
    print(dq, exflag)

    # was the recovery within bounds?
    assert not np.any(exflag)
    print(Sfwd - S)
    assert np.amax(np.abs(Sfwd - S)) < 0.002


def il_example(linearity_file, gain_file, ipc_file):
    """
    Some tests for the inverse linearity class.

    Parameters
    ----------
    linearity_file : str
        ASDF linearity calibration reference file.
    gain_file : str
        ASDF gain calibration reference file.
    ipc_file : str
        ASDF ipc4d calibration reference file.

    Returns
    -------
    None

    """

    # expected results
    target1 = np.array(
        [[4801.0491668, 4900.74928657, 4800.50198393], [4800.30217909, 4900.15392476, 4800.05504147]]
    )
    target2 = np.array(
        [[4803.76066256, 4920.3284374, 4803.19832938], [4817.8237426, 6177.69747299, 4817.69985963]]
    )
    target3 = np.array([[3.99985202, 30.00056007, 4.00053699], [25.99872292, 1871.82650119, 26.00227536]])

    ILTEST = ipc_linearity.IL(linearity_file, gain_file, ipc_file)
    n = 4088
    NE = np.zeros((n, n), dtype=np.float32)
    ymin = 260
    ymax = 262
    xmin = 140
    xmax = 143
    val1 = ILTEST.apply(NE, electrons=True, electrons_out=False)[ymin:ymax, xmin:xmax]
    print(val1)
    assert np.all(np.abs(target1 - val1) < 0.002)
    NE[::3, ::3] = 2.0e3
    val2 = ILTEST.apply(NE, electrons=True, electrons_out=False)[ymin:ymax, xmin:xmax]
    print(val2)
    assert np.all(np.abs(target2 - val2) < 0.002)
    val3 = ILTEST.apply(NE, electrons=True, electrons_out=True)[ymin:ymax, xmin:xmax]
    print(val3)
    assert np.all(np.abs(target3 - val3) < 0.002)

    assert np.shape(ILTEST.apply(NE, electrons=True, electrons_out=False)) == (4088, 4088)
    assert np.shape(ILTEST.apply(NE, electrons=True, electrons_out=True)) == (4088, 4088)


def run_all(tmp_path, subtr, checkdata, cleanup=True):
    """
    Test function for a small pyimcom run.

    Parameters
    ----------
    tmp_path : str or pathlib.Path
        Directory in which to run the test.
    subtr : bool
        Test whether to package the reference array?
    checkdata : dict
        Dictionary of properties to check. Includes the keys:

        - ``wfi18p1``: 1st percentile of each time slice of the L1 with WFI18 transient
        - ``wfi18p99``: 99th percentile of each time slice of the L1 with WFI18 transient

    cleanup : bool, optional
        Whether to clean up files when done? (Only turn off if you are running on a local
        machine to look at artifacts.)

    Returns
    -------
    image : np.ndarray
        The slope image from the default run.
    mask : np.ndarray
        The mask from the default run.

    """

    tmp_dir = str(tmp_path)  # get string version of directory
    rng = np.random.RandomState(seed=1000)  # old RNG, for compatibility
    band = "F184"

    # subdirectories
    with contextlib.suppress(FileExistsError):
        os.mkdir(tmp_dir + "/IN")
    with contextlib.suppress(FileExistsError):
        os.mkdir(tmp_dir + "/OUT-L1")
    with contextlib.suppress(FileExistsError):
        os.mkdir(tmp_dir + "/OUT-L2")

    genfile(tmp_dir + f"/IN/Roman_Test_truth_{band:s}_{id}_{sca}.fits", v=1)
    gencal(tmp_dir + "/roman_wfi", rng)

    ctypes = ["linearitylegendre", "gain", "dark", "read", "ipc4d", "flat", "biascorr", "saturation"]
    caldir = {}
    for ctype in ctypes:
        ctype2 = ctype
        if ctype == "flat":
            ctype2 = "pflat"
        caldir[ctype] = tmp_dir + f"/roman_wfi_{ctype2:s}_{tag:s}_SCA{sca:02d}.asdf"
    print(caldir)

    these_reads = []
    for i in range(len(read_pattern)):
        these_reads.append(read_pattern[i][0])
        these_reads.append(read_pattern[i][-1] + 1)
    print("these_reads -->", these_reads)

    # forward-backward test (contains its own asserts)
    forward_backward_lin_ilin(caldir["linearitylegendre"])
    il_example(caldir["linearitylegendre"], caldir["gain"], caldir["ipc4d"])

    # make sample images
    arr = fpaplot.multi_image(
        tmp_dir + "/roman_wfi_{:s}_" + tag + "_SCA{:02d}.asdf", 128, maskhandling.PixelMask1
    )
    # check a few pixels in the output image
    checkpix = np.array(
        [
            [2057, 672, 60, 78, 138],
            [2057, 1527, 175, 220, 46],
            [709, 1624, 68, 1, 84],
            [0, -1, 255, 255, 255],
            [455, 422, 255, 255, 255],
            [455, 424, 0, 0, 0],
        ],
        dtype=np.int16,
    )
    for i in range(np.shape(checkpix)[0]):
        diff = arr[checkpix[i, 0], checkpix[i, 1], :].astype(np.int16) - checkpix[i, -3:]
        assert np.all(np.abs(diff) < 16)
    # Image.fromarray(arr[::-1, :, :]).save("panel_image.png")

    supp = {"EXTRACT_REF": {"data_encoding_offset": 4000}} if subtr else {}
    sim_to_isim.run_config(
        {
            "IN": tmp_dir + f"/IN/Roman_Test_truth_{band:s}_{id}_{sca}.fits",
            "OUT": tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_{sca:d}.asdf",
            "READS": these_reads,
            "FITSOUT": True,
            "CALDIR": caldir,
            "CNORM": 1.0,
            "SEED": 200,
        }
        | supp
    )

    # this will have darkdecay
    sim_to_isim.run_config(
        {
            "IN": tmp_dir + f"/IN/Roman_Test_truth_{band:s}_{id}_{sca}.fits",
            "OUT": tmp_dir + f"/OUT-L1/sim_L1withdd_{band:s}_{id:d}_{sca:d}.asdf",
            "READS": these_reads,
            "FITSOUT": True,
            "CALDIR": caldir,
            "CNORM": 1.0,
            "SEED": 200,
        }
        | supp
    )

    # test extraction of the orientation
    ordict = orientation.get_orientation(tmp_dir + f"/OUT-L1/sim_L1withdd_{band:s}_{id:d}_{sca:d}.asdf")
    print(ordict)
    assert 37.045 < ordict["ra"] < 37.047
    assert -19.507 < ordict["dec"] < -19.505
    assert 4.999 < ordict["pa"] < 5.001
    assert len(ordict["ra_sca"]) == 18
    assert len(ordict["dec_sca"]) == 18

    # copy this file into a place for WFI18 (so that we can test transient function)
    if not subtr:
        with asdf.open(tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_{sca:d}.asdf") as a:
            a["roman"]["meta"]["instrument"]["detector"] = "WFI18"
            # this is just a toy example to exercise that the WFI18 correction is "on"
            newdata = a["roman"]["data"][0, 4:-4, 4:-4].astype(np.float32)
            rows = np.linspace(4, 4091, 4088)
            rows += rows // 256 * 4  # insert a timing gap
            newdata += (-80.0 * np.exp(-rows / 150.0) + 5.0 * np.exp(-rows / 1300.0)).astype(np.float32)[
                :, None
            ]
            a["roman"]["data"][0, 4:-4, 4:-4] = np.clip(newdata, 0, 65535)
            a.write_to(tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_18.asdf")

    # make a GWCS file
    if not subtr:
        _write_wcs(tmp_dir + "/test_gwcs.asdf")

    # copy this file with guide window
    with asdf.open(tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_{sca:d}.asdf") as a:
        a["roman"]["meta"]["instrument"]["detector"] = "WFI16"
        p = a["roman"]["meta"]["guide_star"]
        p["hv_position"] = [1002.0, 100.0]
        p["window_xstart"] = 1736
        p["window_xstop"] = 1751
        p["window_ystart"] = 2940
        p["window_ystop"] = 2955
        a.write_to(tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_16.asdf")

    # Below here is stuff for Level 1-->2

    c2 = {
        "IN": tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_{sca:d}.asdf",
        "OUT": tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}.asdf",
        "FITSWCS": tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_{sca:d}_asdf_wcshead.txt",
        "CALDIR": caldir,
        "RAMP_OPT_PARS": {"slope": 0.4, "gain": 1.8, "sigma_read": 7.0},
        "JUMP_DETECT_PARS": {"SthreshA": 10.0, "SthreshB": 4.5, "IthreshA": 0.6, "IthreshB": 600.0},
        "SKYORDER": 2,
        "FITSOUT": True,
        "NOISE": {
            "LAYER": ["Rz4S2C1", "O", "Prb2"],
            "TEMP": tmp_dir + f"/temp_{band:s}_{id:d}_{sca:d}.asdf",
            "SEED": 10000,
            "OUT": tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}_noise.asdf",
        },
    }
    if subtr:
        c2["EXCLUDE_FIRST"] = False
    gen_cal_image.calibrateimage(c2 | {"SLICEOUT": True})
    gen_noise_image.generate_all_noise(c2)
    print("\nwrite mask")
    maskhandling.PixelMask1.convert_file(c2["OUT"], c2["OUT"][:-5] + "_mask.fits")

    # Also exercise the romancal likelihood ramp-fit path
    # Turn on the WFI18 transient step, which is a no-op here because this is WFI04.
    c_rc = c2 | {
        "OUT": tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}_romancalfit.asdf",
        "romancal_ramp_fit": True,
        "correct_wfi18_transient": True,
    }
    gen_cal_image.calibrateimage(c_rc)

    # now do a test as if this were WFI18 to exercise the post-reset transient functionality
    if not subtr:
        c2x = c2 | {"correct_wfi18_transient": True, "EXCLUDE_FIRST": False}
        c2x["IN"] = tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_18.asdf"
        c2x["OUT"] = tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_18_noexcludefirst.asdf"
        gen_cal_image.calibrateimage(c2x)
        c4 = c2 | {"correct_wfi18_transient": True, "EXCLUDE_FIRST": False}
        c4["OUT"] = tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}_noexcludefirst.asdf"
        gen_cal_image.calibrateimage(c4)

    # this one tests dark decay
    # right now not actually implemented -- this shows up as a change in the "sky"
    c2d = c2.copy()
    c2d["CALDIR"] = c2["CALDIR"] | {"dark_decay": tmp_dir + f"/roman_wfi_darkdecay_{tag:s}_SCA{sca:02d}.asdf"}
    c2d["IN"] = tmp_dir + f"/OUT-L1/sim_L1withdd_{band:s}_{id:d}_{sca:d}.asdf"
    c2d["OUT"] = tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}_withdecay.asdf"
    gen_cal_image.calibrateimage(c2d)

    if not subtr:
        c2e = {
            "IN": tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_{sca:d}.asdf",
            "OUT": tmp_dir + f"/OUT-L2/sim_L2gwcs_{band:s}_{id:d}_{sca:d}.asdf",
            "SOCWCS": tmp_dir + "/test_gwcs.asdf",
            "CALDIR": caldir,
            "RAMP_OPT_PARS": {"slope": 0.4, "gain": 1.8, "sigma_read": 7.0},
            "JUMP_DETECT_PARS": {"SthreshA": 10.0, "SthreshB": 4.5, "IthreshA": 0.6, "IthreshB": 600.0},
            "SKYORDER": 2,
        }
        gen_cal_image.calibrateimage(c2e)
    c5 = {
        "IN": tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_16.asdf",
        "OUT": tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_16.asdf",
        "FITSWCS": tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_{sca:d}_asdf_wcshead.txt",
        "CALDIR": caldir,
        "RAMP_OPT_PARS": {"slope": 0.4, "gain": 1.8, "sigma_read": 7.0},
        "JUMP_DETECT_PARS": {"SthreshA": 10.0, "SthreshB": 4.5, "IthreshA": 0.6, "IthreshB": 600.0},
        "SKYORDER": 2,
        "FITSOUT": True,
    }
    gen_cal_image.calibrateimage(c5)

    ### TESTS ON THE OUTPUTS

    dtrim = 4

    # gain map
    with asdf.open(caldir["gain"]) as g_:
        g = np.copy(g_["roman"]["data"])
    print("gain =", np.shape(g))

    # expected signal, in DN/s
    with fits.open(tmp_dir + f"/IN/Roman_Test_truth_{band:s}_{id}_{sca}.fits") as f:
        expected_signal = f[0].data[::-1, :] / g[dtrim:-dtrim, dtrim:-dtrim]
        expected_signal /= f[0].header["EXPTIME"]

    with asdf.open(tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}.asdf") as a:
        cry, crx = np.where(np.bitwise_and(a["roman"]["dq"] >> 2, 1))
        print("FIRST CRs:")
        print(cry[:10])
        print(crx[:10])
        v = max(cry[0], 3)
        u = max(crx[0], 3)
        print("2D image -->")
        print(a["roman"]["data"][v - 3 : v + 3, u - 3 : u + 3])

        for i in range(32):
            count = np.count_nonzero(np.bitwise_and(a["roman"]["dq"] >> i, 1))
            print(f"BIT {i:2d} {count:7d}")
            if i == 2:
                assert count > 10000 and count < 30000
        isGood = np.where(a["roman"]["dq"] == 0, 1, 0)
        dq_local = np.asarray(a["roman"]["dq"])
        err_local = np.asarray(a["roman"]["err"], dtype=np.float64)
        vp_local = np.asarray(a["roman"]["var_poisson"], dtype=np.float64)

        # now pull out the data, in DN/s
        data_out = np.copy(a["roman"]["data"])

        # sky checks
        print(np.array(a["processinfo"]["skycoefs"]))
        assert len(a["processinfo"]["skycoefs"]) == 6
        assert -0.3 <= a["processinfo"]["skycoefs"][0] <= 1.7
        for i in range(1, 6):
            assert np.abs(a["processinfo"]["skycoefs"][i]) < 1.0
        skycoefs = np.array(a["processinfo"]["skycoefs"])
        skyresid = np.array(a["roman"]["data_withsky"]) - np.array(a["roman"]["data"])
        # now build residuals from sky model
        N_ = np.shape(skyresid)[-1]
        print("shape", N_)
        u_ = np.linspace(-1.0 + 1.0 / N_, 1.0 - 1.0 / N_, N_)
        u, v = np.meshgrid(u_, u_)
        skyresid -= (
            skycoefs[0]
            + skycoefs[1] * v
            + skycoefs[2] * (1.5 * v**2 - 0.5)
            + skycoefs[3] * u
            + skycoefs[4] * u * v
            + skycoefs[5] * (1.5 * u**2 - 0.5)
        )
        print("MAX", np.amax(np.abs(skyresid)))
        assert np.amax(np.abs(skyresid)) < 1e-3

        print(a["processinfo"]["romanimpreprocess_env"])
        assert "." in a["processinfo"]["romanimpreprocess_env"]["romanimpreprocess"]

    hisignal = np.logical_and(isGood, expected_signal > 5.0)
    print(isGood, np.mean(isGood.astype(np.float32)), "hisig =", np.count_nonzero(hisignal))

    x = np.where(isGood, data_out - expected_signal, 0.0)
    # fits.PrimaryHDU(x).writeto(tmp_dir + "/out_diff.fits", overwrite=True)

    # some quality checks on unmasked pixels
    assert np.count_nonzero(np.abs(x) > 100) < 50
    assert np.count_nonzero(np.logical_and(np.abs(x) > 20, expected_signal < 1)) < 50

    # checks on the romancal likelihood ramp-fit output
    with asdf.open(c_rc["OUT"]) as a_rc:
        assert np.shape(a_rc["roman"]["data"]) == (4088, 4088)
        assert "dumo" in a_rc["roman"]
        assert "chisq" in a_rc["roman"]
        isGood_rc = a_rc["roman"]["dq"] == 0
        data_rc = np.asarray(a_rc["roman"]["data"])
        assert np.all(np.isfinite(data_rc[isGood_rc]))
        x_rc = np.where(isGood_rc, data_rc - expected_signal, 0.0)
        assert np.count_nonzero(np.abs(x_rc) > 100) < 50

        # The local and romancal-likelihood fitters use the same data, so on
        # common good pixels they should report similar uncertainties, agree well
        # within those uncertainties, and flag a similar number of cosmic rays.
        dq_rc = np.asarray(a_rc["roman"]["dq"])
        err_rc = np.asarray(a_rc["roman"]["err"], dtype=np.float64)
        vp_rc = np.asarray(a_rc["roman"]["var_poisson"], dtype=np.float64)
        common = isGood.astype(bool) & isGood_rc

        # median uncertainties shouldn't change by more than ~5%
        for name, m_loc, m_rc in [
            ("err", err_local, err_rc),
            ("var_poisson", vp_local, vp_rc),
        ]:
            ratio = np.median(m_rc[common]) / np.median(m_loc[common])
            print(f"uncertainty ratio rc/local {name:>11} = {ratio:.3f}")
            assert 0.95 < ratio < 1.05

        # the two fits should agree well within their reported uncertainty
        z = mad_std(((data_out - data_rc) / err_rc)[common])
        # cosmic-ray flagging should be similar
        jump_local = np.count_nonzero((dq_local & pixel.JUMP_DET) != 0)
        jump_rc = np.count_nonzero((dq_rc & pixel.JUMP_DET) != 0)
        print(f"mad_std((local-rc)/err)={z:.4f}, jump_local={jump_local}, jump_rc={jump_rc}")
        assert z < 0.05
        assert 100 < jump_rc < 2 * jump_local

    # checks on the WFI18 version
    if not subtr:
        with asdf.open(c2x["IN"]) as _a:
            assert np.allclose(
                np.percentile(_a["roman"]["data"], 99, axis=(-2, -1)),
                checkdata["wfi18p99"],
                atol=2.2,
                rtol=0.0,
            )
            assert np.allclose(
                np.percentile(_a["roman"]["data"], 1, axis=(-2, -1)), checkdata["wfi18p1"], atol=2.2, rtol=0.0
            )

        with asdf.open(c4["OUT"]) as a_notr, asdf.open(c2x["OUT"]) as a_withtr:
            diff = a_withtr["roman"]["data"] - a_notr["roman"]["data"]
            # the 10/20/80/90th percentile wthiout correction are -0.078/-0.049/+0.029/+0.031
            assert np.percentile(diff, 10) > -0.014
            assert np.percentile(diff, 20) > -0.007
            assert np.percentile(diff, 80) < 0.007
            assert np.percentile(diff, 90) < 0.014
            # make sure they are different!
            assert np.percentile(diff, 80) - np.percentile(diff, 20) > 1e-4

    # checks on the darkdecay version
    with asdf.open(c2["OUT"]) as a_orig, asdf.open(c2d["OUT"]) as a_new:
        diff = a_new["roman"]["data"] - a_orig["roman"]["data"]
        diff1d = np.median(diff, axis=1)
        assert np.all(np.abs(diff1d) < 1.0e-4)

        # sky goes "up" because we are correcting a downward slope
        skydiff = a_new["processinfo"]["skycoefs"] - a_orig["processinfo"]["skycoefs"]
        assert 0.004 < skydiff[0] < 0.007
        assert np.all(np.abs(skydiff[1:]) < 0.0015)

    # check that we can convert the output to PDF
    visualize.visualize(
        [
            None,
            tmp_dir + f"/OUT-L1/sim_L1_{band:s}_{id:d}_{sca:d}.asdf",
            "128,256,512,640",
            tmp_dir + "/out_im1.pdf",
            0.5,
        ]
    )
    assert os.path.exists(tmp_dir + "/out_im1.pdf")

    # GWCS test
    if not subtr:
        with asdf.open(tmp_dir + f"/OUT-L2/sim_L2gwcs_{band:s}_{id:d}_{sca:d}.asdf") as a:
            agwcs = a["roman"]["meta"]["wcs"]
            pos = np.fft.rfft(np.array(agwcs(0, 0) + agwcs(4087, 0) + agwcs(0, 4087) + agwcs(4087, 4087)))
            target = np.array(
                [
                    36.40255634261365,
                    -0.16714760571096945 + 0.23873396859962925j,
                    0.15375005488544957 + 0.2262702486514172j,
                    -0.16788418861885276 - 0.09397200154994632j,
                    430.47778368146265,
                ]
            )
            assert np.all(np.abs(pos - target) < 0.1 * 0.11 / 3600.0)  # within 0.1 pix

    # noise tests
    with asdf.open(tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}_noise.asdf") as a:
        print(a.info(max_rows=None))
        adata = a["noise"]
        nlayer = len(c2["NOISE"]["LAYER"])
        assert np.shape(adata) == (nlayer, 4088, 4088)
        for j in range(nlayer):
            print("layer", j, c2["NOISE"]["LAYER"][j])
            x = np.where(isGood, adata[j, :, :], 0.0)
            print(np.percentile(x, 0.1), np.percentile(x, 5), np.percentile(x, 95), np.percentile(x, 99.9))
            x2 = adata[j, :, :][hisignal]
            print(np.percentile(x2, 25), np.percentile(x2, 75))

            # assertions
            if j == 0:
                assert 0.7 < np.percentile(x, 95) - np.percentile(x, 5) < 1.1
                assert 0.3 < np.percentile(x2, 75) - np.percentile(x2, 25) < 0.4
            if j == 1:
                assert 0.14 < np.percentile(x, 95) - np.percentile(x, 5) < 0.40
                assert 1.1 < np.percentile(x2, 75) - np.percentile(x2, 25) < 1.4
            if j == 2:
                assert 0.14 < np.percentile(x, 95) - np.percentile(x, 5) < 0.40

    # test truncation to float16 on a clipped layer
    c3 = c2 | {"NOISE_PRECISION": 16}
    c3["NOISE"]["OUT"] = tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}_noise16.asdf"
    gen_noise_image.generate_all_noise(c3)
    p = tmp_dir + f"/OUT-L2/sim_L2_{band:s}_{id:d}_{sca:d}"
    with asdf.open(p + "_noise.asdf") as a_orig, asdf.open(p + "_noise16.asdf") as a16:
        diff = (a16["noise"][0, :, :] - a_orig["noise"][0, :, :]) / (1.0 + np.abs(a_orig["noise"][0, :, :]))
    assert np.all(np.abs(diff) < 0.005)

    # this should raise an exception, we will check that
    c3["NOISE_PRECISION"] = -1
    try:
        gen_noise_image.generate_all_noise(c3)
        assert c3["NOISE_PRECISION"] == 0  # shouldn't get here
    except ValueError as ve:
        assert str(ve) == "Unsupported noise precision."

    with fits.open(str(tmp_path) + "/OUT-L2/sim_L2_F184_163_4_asdf_to.fits") as f:
        image = np.copy(f[0].data)
        mask = np.copy(f[1].data)

    # test the guide window
    with asdf.open(c5["OUT"]) as af:
        v = np.where(af["roman"]["dq"] & pixel.GW_AFFECTED_DATA, 1, 0).astype(np.uint8)
        imtest = np.zeros((4088, 4088), dtype=np.uint8)
        imtest[:, 1732:1748] = 1
        imtest[2936:2952, :] = 1
        imtest[2935:2953, 1731:1749] = 1
        # fits.PrimaryHDU(v).writeto("v.fits", overwrite=True)  # for diagnostic
        assert np.all(v == imtest)

    if cleanup:
        if not subtr:
            os.remove(tmp_path / "test_gwcs.asdf")
            os.remove(tmp_path / "OUT-L2/sim_L2gwcs_F184_163_4.asdf")
            for rf in ["sim_L1_F184_163_18.asdf"]:
                os.remove(tmp_path / f"OUT-L1/{rf}")
            for rf in [
                "sim_L2_F184_163_18_noexcludefirst_asdf_to.fits",
                "sim_L2_F184_163_4_noexcludefirst.asdf",
                "sim_L2_F184_163_18_noexcludefirst.asdf",
                "sim_L2_F184_163_4_noexcludefirst_asdf_to.fits",
            ]:
                os.remove(tmp_path / f"OUT-L2/{rf}")
        for rf in [
            "out_im1.pdf",
            "roman_wfi_linearitylegendre_TESTONLY_SCA04.asdf",
            "temp_F184_163_4_L2.asdf",
            "roman_wfi_biascorr_TESTONLY_SCA04.asdf",
            "roman_wfi_mask_TESTONLY_SCA04.asdf",
            "temp_F184_163_4_refL2_asdf_to.fits",
            "roman_wfi_dark_TESTONLY_SCA04.asdf",
            "roman_wfi_pflat_TESTONLY_SCA04.asdf",
            "temp_F184_163_4_refL2.asdf",
            "roman_wfi_darkdecay_TESTONLY_SCA04.asdf",
            "roman_wfi_read_TESTONLY_SCA04.asdf",
            "temp_F184_163_4.asdf",
            "roman_wfi_gain_TESTONLY_SCA04.asdf",
            "roman_wfi_saturation_TESTONLY_SCA04.asdf",
            "roman_wfi_ipc4d_TESTONLY_SCA04.asdf",
            "temp_F184_163_4_L2_asdf_to.fits",
        ]:
            os.remove(tmp_path / rf)
        for rf in ["Roman_Test_truth_F184_163_4.fits"]:
            os.remove(tmp_path / f"IN/{rf}")
        for rf in [
            "sim_L1_F184_163_4.asdf",
            "sim_L1withdd_F184_163_4.asdf",
            "sim_L1_F184_163_4_asdf_to.fits",
            "sim_L1withdd_F184_163_4_asdf_to.fits",
            "sim_L1_F184_163_4_asdf_wcshead.txt",
            "sim_L1withdd_F184_163_4_asdf_wcshead.txt",
        ]:
            os.remove(tmp_path / f"OUT-L1/{rf}")
        for rf in [
            "sim_L2_F184_163_4_romancalfit_asdf_to.fits",
            "sim_L2_F184_163_4_noise_asdf_to.fits",
            "sim_L2_F184_163_4_romancalfit.asdf",
            "sim_L2_F184_163_4_asdf_to.fits",
            "sim_L2_F184_163_4_noise.asdf",
            "sim_L2_F184_163_4_withdecay_asdf_to.fits",
            "sim_L2_F184_163_4_mask.fits",
            "sim_L2_F184_163_4_noise16_asdf_to.fits",
            "sim_L2_F184_163_4_withdecay.asdf",
            "sim_L2_F184_163_4_noise16.asdf",
            "sim_L2_F184_163_4.asdf",
        ]:
            os.remove(tmp_path / f"OUT-L2/{rf}")

    return image, mask


def test_run_all(tmp_path):
    """Main workflow test."""

    # Test with moving the reference.
    im0, mask0 = run_all(
        tmp_path,
        True,
        {
            "wfi18p99": np.array([4030.0, 4034.0, 4044.0, 4060.0, 4068.0]),
            "wfi18p1": np.array([3946.0, 3979.0, 3984.0, 3984.0, 3982.0]),
        },
    )

    # Test without moving the reference.
    im1, mask1 = run_all(
        tmp_path,
        False,
        {
            "wfi18p99": np.array([5925.0, 5927.0, 5931.0, 5937.0, 5944.0, 5947.0]),
            "wfi18p1": np.array([4780.0, 4785.0, 4788.0, 4793.0, 4796.0, 4797.0]),
        },
    )

    # Check that we got the same answer.
    thresh = 2  # want to allow the occasional floating point moves over a flag boundary.
    assert np.count_nonzero(mask0 != mask1) <= thresh
    err = np.abs(im1 - im0) / (1.0 + np.abs(im1))
    assert np.count_nonzero(err > 3.0e-4) <= thresh


def test_flip(tmp_path):
    """Test of flipping the SCA."""

    # Build the data.
    # Note this makes a header with SIP coefficients.
    tmpdir = str(tmp_path)
    genfile(tmpdir + "/test1.fits", v=1)
    with fits.open(tmpdir + "/test1.fits", memmap=False, lazy_load_hdus=False) as f:
        my_hdu = fits.PrimaryHDU(f[0].data, header=f[0].header.copy())

    # Now the flipped image
    sim_to_isim.hdu_sip_hflip(my_hdu.data, my_hdu.header)
    my_hdu.writeto(tmpdir + "/test2.fits", overwrite=True)

    with fits.open(tmpdir + "/test1.fits") as f_orig, fits.open(tmpdir + "/test2.fits") as f_new:
        diff = f_orig[0].data - f_new[0].data[:, ::-1]
        assert np.amax(np.abs(diff)) < 1.0e-7

        # WCSs
        wcs_orig = WCS(f_orig[0].header)
        wcs_new = WCS(f_new[0].header)

        # map points
        points_orig = np.array([[100.0, 250.0], [3000.0, 800.0]])
        points_sky = wcs_orig.all_pix2world(points_orig, 0)
        points_new = wcs_new.all_world2pix(points_sky, 0)
        print(points_new)
        points_compare = np.copy(points_orig)
        points_compare[:, 0] = 4087.0 - points_compare[:, 0]
        print(np.abs(points_compare - points_new))
        err = np.amax(np.abs(points_compare - points_new))
        print(err)
        assert err < 1.0e-4
