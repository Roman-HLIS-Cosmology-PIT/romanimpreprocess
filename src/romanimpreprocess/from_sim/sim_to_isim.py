"""
Functions to convert external simulated images to Roman L1/L2-like format.

This works entirely at the single exposure level. Some parts wrap romanisim.

Functions
---------
hdu_sip_hflip
    Flips an SCA in the 3n row in Detector coordinates to align with Science coordinates.
hdu_sip_vflip
    Flips an SCA in the 3n+1 or 3n+2 rows in Detector coordinates to align with Science coordinates.
make_l1_fullcal
    Makes an L1 image using an OpenUniverse input and the calibration data.
    (Merges with romanisim routines.)
noise_1f_frame
    Generates 1/f noise.
fill_in_refdata_and_1f
    Fills in reference pixels and reference output, as well as 1/f noise.
simpletest
    Quick look tool for Level 1 to Level 2 conversion (not for production).
runconfig
    Configuration-driven function to convert a simulation to Level 1 format.

Classes
-------
Image2D
    2D image (may be from simulation).
Image2D_from_L1
    2D image from Level 1 data (useful for 'shortcut' workflow, for most applications use L1_to_L2).

"""

import copy
import re
import sys
import warnings

import asdf
import galsim
import numpy as np
import roman_datamodels
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from romanisim import __version__ as rstversion
from romanisim import image as rimage
from romanisim import l1 as rstl1
from romanisim import parameters, util, wcs
from romanisim import persistence as rip
from romanisim import ris_make_utils as ris

from .. import pars
from ..utils.coordutils import pixelarea

# local imports
from ..utils.ipc_linearity import IL, ipc_rev

# gcrsim import
from gcrpipe.roman_pipeline_interface import generate_singleframe_cr

def hdu_sip_hflip(data, header):
    """
    Horizontal flip of SCA and WCS. Assumes SIP convention.

    This function operates on the data and WCS header in place.

    Parameters
    ----------
    data : np.array
        2D image of an SCA.
    header : astropy.io.fits.header.Header
        Header containing the WCS.

    Returns
    -------
    None

    See Also
    --------
    hdu_sip_vflip : Similar, but for vertical flip instead.

    """

    (ny, nx) = np.shape(data)
    data[:, :] = data[:, ::-1]  # flipping the data is the easy part

    # now flip the WCS
    header["CRPIX1"] = nx + 1 - header["CRPIX1"]
    header["CD1_1"] = -header["CD1_1"]
    header["CD2_1"] = -header["CD2_1"]
    try:
        # if there is a SIP table, we flip it.
        # for A: the even p's need a sign flip to reverse the direction of the SIP u-axis
        # for B: the odd p's need a sign flip to reverse the direction of the SIP u-axis
        a_order = int(header["A_ORDER"])
        b_order = int(header["B_ORDER"])
        for p in range(0, a_order + 1, 2):
            for q in range(a_order + 1 - p):
                keyword = f"A_{p:1d}_{q:1d}"
                if keyword in header:
                    header[keyword] = -float(header[keyword])
        for p in range(1, b_order + 1, 2):
            for q in range(b_order + 1 - p):
                keyword = f"B_{p:1d}_{q:1d}"
                if keyword in header:
                    header[keyword] = -float(header[keyword])
    except (ValueError, KeyError):
        print("Exception in SIP table, skipping ...")


def hdu_sip_vflip(data, header):
    """
    Vertical flip of SCA and WCS. Assumes SIP convention.

    This function operates on the data and WCS header in place.

    Parameters
    ----------
    data : np.array
        2D image of an SCA.
    header : astropy.io.fits.header.Header
        Header containing the WCS.

    Returns
    -------
    None

    See Also
    --------
    hdu_sip_hflip : Similar, but for horizontal flip instead.

    """

    (ny, nx) = np.shape(data)
    data[:, :] = data[::-1, :]  # flipping the data is the easy part

    # now flip the WCS
    header["CRPIX2"] = ny + 1 - header["CRPIX2"]
    header["CD1_2"] = -header["CD1_2"]
    header["CD2_2"] = -header["CD2_2"]
    try:
        # if there is a SIP table, we flip it.
        # for A: the odd q's need a sign flip to reverse the direction of the SIP v-axis
        # for B: the even q's need a sign flip to reverse the direction of the SIP v-axis
        a_order = int(header["A_ORDER"])
        b_order = int(header["B_ORDER"])
        for q in range(1, a_order + 1, 2):
            for p in range(a_order + 1 - q):
                keyword = f"A_{p:1d}_{q:1d}"
                if keyword in header:
                    header[keyword] = -float(header[keyword])
        for q in range(0, b_order + 1, 2):
            for p in range(b_order + 1 - q):
                keyword = f"B_{p:1d}_{q:1d}"
                if keyword in header:
                    header[keyword] = -float(header[keyword])
    except (ValueError, KeyError):
        print("Exception in SIP table, skipping ...")


def make_l1_fullcal(counts, read_pattern, caldir, rng=None, persistence=None, tstart=None):
    """
    Make an L1 image with the full calibration information.

    This carries out similar steps to romanisim.l1.make_l1, but
    provides us a bit more control over the settings.

    Parameters
    ----------
    counts : galsim.Image
        Contains mean number electrons per pixel per exposure.
    read_pattern : list of list of int
        MultiAccum table.
    caldir : dict
        Dictionary of the reference files.
    rng : galsim.BaseDeviate, optional
        The random number generator.
    persistence : romanisim.persistence.Persistence, optional
        Persistence object, not used yet.
    tstart : float, optional
        Start time to feed to romanisim.

    Returns
    -------
    l1 : np.array
        3D image array
    dq : np.array
        3D quality array

    """

    # generate the reset noise (will complain if rng is None!)
    resetnoise = np.zeros_like(counts.array)
    nb = (8192 - np.shape(resetnoise)[-1] // 2) % 256  # get border size
    galsim.GaussianDeviate(rng).generate(resetnoise)
    with asdf.open(caldir["read"]) as f:
        resetnoise *= f["roman"]["resetnoise"][nb:-nb, nb:-nb]
    with asdf.open(caldir["gain"]) as f:
        resetnoise *= f["roman"]["data"][nb:-nb, nb:-nb]
    # now resetnoise is a random image in electrons

    tij = rstl1.read_pattern_to_tij(read_pattern)

    # subtract the apporopriate amount of electrons so that we will be at the
    # zero level (on average) after some dark current
    if "biascorr" in caldir:
        with asdf.open(caldir["biascorr"]) as f:
            tbias = float(f["roman"]["t0"])
        with asdf.open(caldir["gain"]) as g:
            with asdf.open(caldir["dark"]) as f:
                resetnoise -= (
                    tbias * f["roman"]["dark_slope"][nb:-nb, nb:-nb] / g["roman"]["data"][nb:-nb, nb:-nb]
                )

    # this model includes linearity *and* IPC.
    # default application is electrons_in=False, electrons_out=False
    # (the .apply method is called in apportion_counts_to_resultants with
    # electrons_in = True -- this means that electrons go in and raw DN go out,
    # so actually includes the gain as well!)
    e2dn_model = IL(caldir["linearitylegendre"], caldir["gain"], caldir["ipc4d"], start_e=resetnoise)
    # set the size of the data quality array
    e2dn_model.set_dq(ngroup=len(read_pattern), nborder=pars.nborder)

    # print(read_pattern)
    # print(len(read_pattern))
    # print('-->', counts.array[3890,237], e2dn_model.linearity_file, e2dn_model.gain_file,
    #     e2dn_model.ipc_file)
    # sys.stdout.flush()

    # generates resultants in DN
    resultants, dq = rstl1.apportion_counts_to_resultants(
        counts.array,
        tij,
        inv_linearity=e2dn_model,
        crparam={},
        persistence=persistence,
        tstart=tstart,
        rng=rng,
        seed=None,
    )

    #gcrsim pipeline insertion
    rng = np.random.default_rng()
    out_array_img = generate_singleframe_cr(rng)
    # print('resultants array', resultants[:,3890,237])

    with asdf.open(caldir["read"]) as f:
        resultants = rstl1.add_read_noise_to_resultants(
            resultants,
            tij,
            rng=rng,
            seed=None,
            read_noise=f["roman"]["data"][nb:-nb, nb:-nb],
            pedestal_extra_noise=None,
        )

    if "biascorr" in caldir:
        with asdf.open(caldir["biascorr"]) as f:
            resultants += f["roman"]["data"]

    resultants[:, :, :] = np.round(resultants)

    return resultants * u.DN, dq


def noise_1f_frame(rng):
    """
    Generates a 4096x128 block of 1/f noise.

    The frame is normalized to variance of 1 per logarithmic range in frequency,
    i.e., S(f) = 1/f, where Var X = int_0^infty S(f) df.

    Parameters
    ----------
    rng : galsim.BaseDeviate
        The random number generator.

    Returns
    -------
    np.array of float
        Shape (4096,128).

    """

    len = 2 * pars.nside * pars.channelwidth

    this_array = np.zeros(2 * len)
    galsim.GaussianDeviate(rng).generate(this_array)

    # get frequencies and amplitude ~ sqrt{power}
    freq = np.linspace(0, 1 - 1.0 / len, len)
    freq[len // 2 :] -= 1.0
    amp = (1.0e-99 + np.abs(freq * len)) ** (-0.5)
    amp[0] = 0.0

    ftsignal = np.zeros((len,), dtype=np.complex128)
    ftsignal[:] = this_array[:len]
    ftsignal[:] += 1j * this_array[len:]
    ftsignal *= amp
    block = np.fft.fft(ftsignal).real[: len // 2] / np.sqrt(2.0)
    block -= np.mean(block)
    # print('generated noise, std -->', np.std(block))

    return block.reshape((pars.nside, pars.channelwidth)).astype(np.float32)


def fill_in_refdata_and_1f(im, caldir, rng, tij, fill_in_banding=True, amp33=None):
    """
    Fills in the reference pixel data in an image, and adds 1/f noise.

    If `amp33` is provided, then also tries to fill in the reference output
    (if the calibration reference files have that information).

    Noise is added in-place to `im` and (if provided) `amp33`, keeping the same
    data type.

    Parameters
    ----------
    im : np.array
        The image data cube. Shape (ngrp, ny, nx).
    caldir : dict
        The dictionary of calibration reference files.
    rng : galsim.BaseDeviate
        The random number generator.
    tij : list of list of float
        Time stamps of the reads in seconds.
    fill_in_banding : bool, optional
        Whether to gennerate 1/f noise.
    amp33 : np.array, optional
        If provided, array to put the reference output.

    Returns
    -------
    None

    """

    (ngrp, ny, nx) = np.shape(im)  # get shape
    nborder = parameters.nborder

    # the extra layer in noise is for the reset noise, which gets added to everything
    noise = np.zeros((ngrp + 1, ny, nx), dtype=np.float32)
    galsim.GaussianDeviate(rng).generate(noise)
    with asdf.open(caldir["read"]) as f:
        noise[:-1, :, :] *= f["roman"]["data"][None, :, :]
        noise[-1, :, :] *= f["roman"]["resetnoise"]
    for j in range(len(tij)):
        noise[j, :, :] /= len(tij[j]) ** 0.5
    noise[:-1, :, :] += noise[-1, :, :][None, :, :]  # adds the reset noise to the reference pixels

    with asdf.open(caldir["dark"]) as f:
        noise[:-1, :, :] += np.copy(f["roman"]["data"])

    # what we have above is a dark image, but we want to fill in the
    # active pixels with the data from im
    noise[:-1, nborder : ny - nborder, nborder : nx - nborder] = im[
        :, nborder : ny - nborder, nborder : nx - nborder
    ].astype(noise.dtype)

    # reference output
    amp33info = {
        "valid": False,
        "med": np.zeros((pars.nside, pars.channelwidth), dtype=np.float32),
        "std": np.zeros((pars.nside, pars.channelwidth), dtype=np.float32),
        "M_PINK": 0.0,
        "RU_PINK": 0.0,
    }
    if amp33 is not None:
        with asdf.open(caldir["read"]) as f:
            if "amp33" in f["roman"]:
                amp33info = copy.deepcopy(f["roman"]["amp33"])
            else:
                warnings.warn("Didn't find reference output information. Skipping ...")

    # generate correlated noise
    if fill_in_banding:
        with asdf.open(caldir["read"]) as f:
            u_pink = float(f["roman"]["anc"]["U_PINK"])
            c_pink = float(f["roman"]["anc"]["C_PINK"])
        print("adding correlated noise", u_pink, c_pink)
        for j in range(len(tij)):
            common_noise = noise_1f_frame(rng) * c_pink
            for ch in range(32):
                pinknoise = noise_1f_frame(rng) * u_pink + common_noise
                if ch % 2 == 1:
                    pinknoise = pinknoise[:, ::-1]
                noise[j, :, pars.channelwidth * ch : pars.channelwidth * (ch + 1)] += (
                    pinknoise / len(tij[j]) ** 0.5
                ).astype(noise.dtype)

            # reference output is completely built here (signal + noise)
            if amp33info["valid"]:
                whitenoise = np.zeros((pars.nside, pars.channelwidth), dtype=np.float32)
                galsim.GaussianDeviate(rng).generate(whitenoise)
                whitenoise *= amp33info["std"]
                pinknoise = amp33info["RU_PINK"] * noise_1f_frame(rng) + amp33info["M_PINK"] * common_noise
                amp33[j, :, :] = (amp33info["med"] + (whitenoise + pinknoise) / len(tij[j]) ** 0.5).astype(
                    amp33.dtype
                )

    # write back to the original
    im[:, :, :] = np.clip(np.round(noise[:-1, :, :]), 0, 2**16 - 1).astype(im.dtype)


class Image2D:
    """
    2D image, along with WCS and sky information.

    It can be constructed from simulations or (ultimately) from Roman data.

    Parameters
    ----------
    intype : str
        Input type (e.g., "anlsim")

    Attributes
    ----------
    image : np.array of flat
        A 2D image. Units are e/p/s if generated from a simulation in e.
    galsimwcs : galsim.wcs.CelestialWCS
        The world coordinate system for this image.
    header : astropy.io.fits.header.Header
        The world coordinate system for this image in FITS WCS format.
    date : str
        The observation date (ISO 8601 string).
    filter : str
        The observation filter (4 characters, e.g., R062).
    idsca : (int, int)
        The observation ID and SCA.
    ra_ : float
        Right ascension (in degrees) of the WFI.
    dec_ : float
        Declination (in degrees) of the WFI.
    pa_ : float
        Position angle (in degrees) of the WFI.
    refdata : dict
        Reference data locations.
    af : asdf.AsdfFile
        Level 1 data
    af2 : asdf.AsdfFile
        Level 2 data

    Methods
    -------
    __init__
        Constructor
    init_anlsim
        Constructor from Open Universe simulation file.
    simulate
        Simulates the ramps, including L1 and L2 data.
    L1_write_to
        Write simulated L1 data file (ASDF)
    L2_write_to
        Write simulated L2 data file (ASDF)

    Notes
    -----
    The legal `intype` strings are:

    * ``anlsim`` : The Open Universe 2024 simulation "truth" (or equivalent)

    """

    def __init__(self, intype, **kwargs):
        if intype == "anlsim":
            self.init_anlsim(kwargs["fname"])

    def init_anlsim(self, fname, flip=True):
        """
        Constructor from Open Universe 2024-type simulation.

        Parameters
        ----------
        fname : str
            The input file name.
        flip : bool, optional
            If True, then flips from SCA native coordinates to science-aligned
            (SOC-like product).

        Returns
        -------
        None

        """

        # get (id,sca)
        m = re.search(r"_(\d+)_(\d+)\.fits", fname)
        self.idsca = (int(m.group(1)), int(m.group(2)))

        # read header and data
        with fits.open(fname) as f:
            data = f[0].data
            self.header = f[0].header

        # flip SCAs depending on which row they are in
        if flip:
            if self.idsca[1] % 3 == 0:
                hdu_sip_hflip(data, self.header)
            else:
                hdu_sip_vflip(data, self.header)

        self.image = data / self.header["EXPTIME"]  # get this in electrons per second
        # offset from FITS -> GWCS convention
        self.header["CRPIX1"] -= 1
        self.header["CRPIX2"] -= 1
        self.galsimwcs, origin = galsim.wcs.readFromFitsHeader(self.header)
        try:
            self.date = self.header["DATE-OBS"]
            print(self.date)
            self.date = re.sub(" ", "T", self.date) + "Z"
            print(self.date)
        except (ValueError, KeyError):
            self.date = "2025-01-01T00:00:00.000000"
        self.filter = self.header["FILTER"][:4]

        self.ra_ = float(self.header["RA_TARG"])
        self.dec_ = float(self.header["DEC_TARG"])
        self.pa_ = float(self.header["PA_OBSY"])

    def simulate(self, use_read_pattern, caldir=None, config={}, seed=43):
        """
        Performs Level 1 & 2 simulations.

        This is based on the ``romanisim.image.simulate`` function,
        but some functionality has been changed to be useful for this class.

        Parameters
        ----------
        use_read_pattern : list of list of int
            The MultiAccum table.
        caldir : dict, optional
            Dictionary of where the calibration files are located.
            (Otherwise uses internal defaults, only good for testing.)
        config : dict, optional
            Configuration file (usually expanded from YAML).
        seed : int, optional
            Random number seed.

        Returns
        -------
        None

        """

        target_pattern = 1000000
        parameters.read_pattern[target_pattern] = use_read_pattern
        metadata = ris.set_metadata(
            date=self.date, bandpass=self.filter, sca=self.idsca[1], ma_table_number=target_pattern
        )

        print("::", self.ra_, self.dec_, self.pa_)
        coord = SkyCoord(ra=self.ra_ * u.deg, dec=self.dec_ * u.deg, frame="icrs")
        wcs.fill_in_parameters(metadata, coord, boresight=False, pa_aper=self.pa_)

        rng = galsim.UniformDeviate(seed)

        ### steps below are from romanisim.image.simulate ###

        image_mod = roman_datamodels.datamodels.ImageModel.create_fake_data()
        meta = image_mod.meta
        meta["wcs"] = None

        for key in parameters.default_parameters_dictionary:
            meta[key].update(parameters.default_parameters_dictionary[key])

        for key in metadata:
            meta[key].update(metadata[key])

        util.add_more_metadata(meta)

        read_pattern = metadata["exposure"].get("read_pattern", use_read_pattern)

        # for this simulation, we want to build something self-contained
        refdata = rimage.gather_reference_data(image_mod, usecrds=False)
        # reffiles = refdata["reffiles"]

        # persistence -> None
        persistence = rip.Persistence()

        # boder reference pixels
        nborder = parameters.nborder

        # simulate a blank image
        if caldir is None:
            counts, simcatobj = rimage.simulate_counts(
                image_mod.meta,
                [],
                rng=rng,
                usecrds=False,
                darkrate=refdata["dark"],
                stpsf=False,
                flat=refdata["flat"],
                psf_keywords=dict(),
            )
        else:
            # get dark current in DN/p/s
            with asdf.open(caldir["dark"]) as f:
                this_dark = f["roman"]["dark_slope"][nborder:-nborder, nborder:-nborder]
            # get flat field
            with asdf.open(caldir["flat"]) as f:
                this_flat = f["roman"]["data"][nborder:-nborder, nborder:-nborder]
            # convert to e/p/s
            with asdf.open(caldir["gain"]) as f:
                this_dark = this_dark * f["roman"]["data"][nborder:-nborder, nborder:-nborder]
                g = np.copy(f["roman"]["data"][nborder:-nborder, nborder:-nborder])  # save gain for later
            # de-convolve the IPC kernel
            with asdf.open(caldir["ipc4d"]) as f:
                this_dark = ipc_rev(this_dark, f["roman"]["data"])  # dark is in e/s
                this_flat = ipc_rev(
                    this_flat, f["roman"]["data"], gain=g
                )  # but flat was measured in DN_lin so need gain=g
                this_flat = np.clip(this_flat, 0.0, 2 - 2**-21)
                this_dark = np.clip(
                    this_dark, -0.1 * this_flat, None
                )  # prevent a spurious negative dark from giving an illegal negative total count rate
            # now run with this version of the dark rate
            counts, simcatobj = rimage.simulate_counts(
                image_mod.meta,
                [],
                rng=rng,
                usecrds=False,
                darkrate=this_dark,
                stpsf=False,
                flat=this_flat,
                psf_keywords=dict(),
            )
        util.update_pointing_and_wcsinfo_metadata(image_mod.meta, counts.wcs)

        # convert from e/s --> e using the parameters file and read pattern
        t = parameters.read_time * (use_read_pattern[-1][-1] - use_read_pattern[0][0])

        # the input simulations include the pixel area in their estimated e/s
        # but the flat field will ultimately include the pixel area as well!
        # therefore we need to re-scale the flat to the ideal pixel area (0.11 arcsec)^2
        flat_witharea = this_flat / (pixelarea(self.galsimwcs, N=pars.nside_active) / pars.Omega_ideal)
        C = float(config["CNORM"]) if "CNORM" in config else 1.0
        print(pixelarea(self.galsimwcs, N=pars.nside_active)[::1024, ::1024])
        print(flat_witharea[::1024, ::1024])
        sys.stdout.flush()
        counts.array[:, :] += rng.np.poisson(
            lam=np.clip(C * t * g / pars.g_ideal * self.image * flat_witharea, 0, None)
        ).astype(counts.array.dtype)

        # this is where the (simulated) L1 data is created
        if caldir is None:
            l1, l1dq = rstl1.make_l1(
                counts,
                read_pattern,
                read_noise=refdata["readnoise"],
                pedestal_extra_noise=parameters.pedestal_extra_noise,
                rng=rng,
                gain=refdata["gain"],
                crparam={},
                inv_linearity=refdata["inverselinearity"],
                tstart=image_mod.meta.exposure.start_time,
                persistence=persistence,
                saturation=refdata["saturation"],
            )
        else:
            # need to run the individual steps ourselves
            l1, l1dq = make_l1_fullcal(
                counts,
                read_pattern,
                caldir,
                rng=rng,
                tstart=image_mod.meta.exposure.start_time,
                persistence=persistence,
            )

        # convert to asdf
        im, extras = rstl1.make_asdf(l1, dq=l1dq, metadata=image_mod.meta, persistence=persistence)

        # fill in the reference pixels and reference output
        if caldir is not None:
            # NO_AMP33 would allow us to bypass the reference output, if it isn't in the config file
            amp33struct = im["amp33"]
            if "NO_AMP33" in caldir:
                if caldir["NO_AMP33"]:
                    amp33struct = None

            fill_in_refdata_and_1f(
                im["data"],
                caldir,
                rng,
                rstl1.read_pattern_to_tij(read_pattern),
                fill_in_banding=True,
                amp33=amp33struct,
            )

        # Create metadata for simulation parameter
        romanisimdict = {"version": rstversion}
        # for storage reasons, I took out all the large metadata arrays in romanisimdict

        # Write file
        self.af = asdf.AsdfFile()
        self.af.tree = {"roman": im, "romanisim": romanisimdict}

        # Make idealized L2 data
        slopeinfo = rimage.make_l2(
            l1,
            read_pattern,
            read_noise=refdata["readnoise"],
            gain=refdata["gain"],
            flat=refdata["flat"],
            linearity=refdata["linearity"],
            darkrate=refdata["dark"],
            dq=l1dq,
        )
        l2dq = np.bitwise_or.reduce(l1dq, axis=0)

        # package header so that there is a obj.header.header
        # this is a hack for compatibility with convert_wcs_to_gwcs
        class Blank:
            pass

        obj = Blank()
        obj.header = Blank()
        obj.header.header = self.header
        #
        im2, extras2 = rimage.make_asdf(
            *slopeinfo,
            metadata=image_mod.meta,
            persistence=persistence,
            dq=l2dq,
            imwcs=obj,
            gain=refdata["gain"],
        )
        # functionality to pull over the WCS from L1 without dependence on wcs.convert_wcs_to_gwcs
        # im2['roman']['meta'].update(wcs=this_gwcs)
        # im2['roman']['meta']['wcsinfo']['s_region'] = wcs.create_s_region(this_gwcs)

        # Create metadata for simulation parameter
        romanisimdict2 = {"version": rstversion}
        romanisimdict2.update(**extras2)

        # Write file
        self.af2 = asdf.AsdfFile()
        self.af2.tree = {"roman": im2, "romanisim": romanisimdict2}

        self.refdata = refdata

    def L1_write_to(self, filename):
        """
        Writes L1 data to a file if available.

        Parameters
        ----------
        filename : str
            Where to write the file (should end with ``.asdf``).

        Returns
        -------
        bool
            True if successful, False if not written.
        """

        if hasattr(self, "af"):
            with open(filename, "wb") as f:
                self.af.write_to(f)
        else:
            return False

    def L2_write_to(self, filename):
        """
        Writes L2 data to a file if available.

        Parameters
        ----------
        filename : str
            Where to write the file (should end with ``.asdf``).

        Returns
        -------
        bool
            True if successful, False if not written.
        """

        if hasattr(self, "af2"):
            with open(filename, "wb") as f:
                self.af2.write_to(f)
        else:
            return False


class Image2D_from_L1(Image2D):
    """Similar to Image2D, but constructed from L1 data file.

    with Image2D_from_L1(infile, refdata, thewcs) as L1:
        ...
    """

    # Context manager functions
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.af.close()

    # Constructor
    def __init__(self, infile, refdata, thewcs, verbose_err=True):
        """Constructor. The arguments are:
        infile : L1 data file (ASDF format)
        refdata : the calibration reference data
        thewcs
            WCS object (in some form -- currently FITS Header or GalSimWCS, though we've had issues
            with the latter)
        """

        self.af = asdf.open(infile)
        self.refdata = refdata
        self.thewcs = thewcs

    def psuedocalibrate(self):
        """Generates a simple calibrated (L2) image.

        This doesn't use romancal, but can be useful as a pass-through function.
        """

        # collect information
        nborder = parameters.nborder

        # Make idealized L2 data
        refdata = self.refdata
        l1dq = np.zeros(
            np.shape(self.af["roman"]["data"][:, nborder:-nborder, nborder:-nborder]), dtype=np.uint32
        )
        slopeinfo = rimage.make_l2(
            self.af["roman"]["data"][:, nborder:-nborder, nborder:-nborder] * u.DN,
            self.af["roman"]["meta"]["exposure"]["read_pattern"],
            read_noise=refdata["readnoise"],
            gain=refdata["gain"],
            flat=refdata["flat"],
            linearity=refdata["linearity"],
            darkrate=refdata["dark"],
            dq=l1dq,
        )
        l2dq = np.bitwise_or.reduce(l1dq, axis=0)

        # make WCS --- a few ways of doing this
        while True:
            wcsobj = None

            class Blank:
                pass

            # first try a FITS header
            if isinstance(self.thewcs, fits.Header):
                wcsobj = Blank()
                wcsobj.header = Blank()
                wcsobj.header.header = self.thewcs
                break

            # should work if this is a GalSim WCS
            try:
                header = fits.Header()
                self.thewcs.writeToFitsHeader(
                    header, galsim.BoundsI(0, pars.nside_active, 0, pars.nside_active)
                )
                # offset to FITS convention -- this is undone later
                header["CRPIX1"] += 1
                header["CRPIX2"] += 1
                wcsobj = Blank()
                wcsobj.header = Blank()
                wcsobj.header.header = header
                warnings.warn("Use of GalSim WCS in calibrate is not fully working yet!")
                break
            except Exception:
                wcsobj = None

            raise Exception("Unrecognized WCS")

        persistence = rip.Persistence()
        im2, extras2 = rimage.make_asdf(
            *slopeinfo,
            metadata=self.af["roman"]["meta"],
            persistence=persistence,
            dq=l2dq,
            imwcs=wcsobj,
            gain=refdata["gain"],
        )

        # Create metadata for simulation parameter
        romanisimdict2 = {"version": rstversion}
        romanisimdict2.update(**extras2)

        # Write file
        self.af2 = asdf.AsdfFile()
        self.af2.tree = {"roman": im2, "romanisim": romanisimdict2}


def simpletest():
    """
    This is a simple script to convert Roman to L1/L2.
    For internal testing only, not production.
    """

    use_read_pattern = [
        [0],
        [1],
        [2, 3],
        [4, 5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
        [34],
    ]

    x = Image2D(
        "anlsim",
        fname="/fs/scratch/PCON0003/cond0007/anl-run-in-prod/truth/Roman_WAS_truth_F184_14747_10.fits",
    )
    print(x.galsimwcs)
    print(x.date, x.idsca)
    print(">>", x.image)
    x.simulate(use_read_pattern)
    x.L1_write_to("sim1.asdf")
    x.L2_write_to("sim2-direct.asdf")

    f = asdf.open("sim1.asdf")
    print(f.info())
    print("corners:")
    print(f["romanisim"]["wcs"])
    print(f["romanisim"]["wcs"]((0, 0, 4087, 4087), (0, 4087, 0, 4087)))
    print(f["roman"]["meta"])
    fits.PrimaryHDU(f["roman"]["data"]).writeto("L1.fits", overwrite=True)

    with Image2D_from_L1("sim1.asdf", x.refdata, x.header) as ff:
        ff.pseudocalibrate()
        ff.L2_write_to("sim2.asdf")

    f = asdf.open("sim2.asdf")
    print(f.info())
    print("corners:")
    print(f["roman"]["meta"]["wcs"]((0, 0, 4087, 4087), (0, 4087, 0, 4087)))
    fits.PrimaryHDU(f["roman"]["data"]).writeto("L2.fits", overwrite=True)


def run_config(config):
    """
    This allows the L1 image construction to be called as a Python function instead of a
    stand-alone code.

    Parameters
    ----------
    config : dict
        Configuration file (usually a dictionary unpacked from YAML).

    Returns
    -------
    None

    """

    # calibration files
    caldir = config.get("CALDIR", None)

    print("Reading from <--", config["IN"])
    print("Writing to -->", config["OUT"])
    print("Using calibration data:")
    print(caldir)
    use_read_pattern = []
    ng = len(config["READS"]) // 2
    for j in range(ng):
        use_read_pattern.append(list(range(int(config["READS"][2 * j]), int(config["READS"][2 * j + 1]))))
    print("Read pattern:", use_read_pattern)

    # Optional inputs
    seed = 43
    if "SEED" in config:
        seed = int(config["SEED"])

    x = Image2D("anlsim", fname=config["IN"])
    x.simulate(use_read_pattern, caldir=caldir, config=config, seed=seed)
    x.L1_write_to(config["OUT"])

    # header information for the WCS
    x.header["COMMENT"] = "truth wcs from sim_to_isim"
    x.header.tofile(config["OUT"][:-5] + "_asdf_wcshead.txt", overwrite=True)

    # also write the FITS file for viewing
    if "FITSOUT" in config:
        if config["FITSOUT"]:
            image_out = np.zeros((ng, pars.nside, pars.nside_augmented), dtype=np.uint16)
            with asdf.open(config["OUT"]) as f:
                image_out[:, :, : pars.nside] = f["roman"]["data"]
                image_out[:, :, pars.nside :] = f["roman"]["amp33"]
                fits.PrimaryHDU(image_out).writeto(config["OUT"][:-5] + "_asdf_to.fits", overwrite=True)

    # simpletest()


if __name__ == "__main__":
    """Stand-alone function to convert from OpenUniverse to L1. Call it with:

    python sim_to_isim <config file>

    The config file is in YAML format and has the fields:
    Required:
    'IN': input file name (FITS)
    'OUT': output file name (must end in '.asdf')
    'READS': a list of length 2*Ngrp: 0th group is [READS[0]:READS[1]], then [READS[2]:READS[3]], etc.

    Optional:
    'CALDIR': Python dictionary with the calibration reference files. This can contain:
        LINEARITY
    'FITSOUT': also write a FITS output (default: False; mostly useful for visualization in ds9)
    'SEED': RNG seed
    """

    # read settings
    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)

    run_config(config)
