"""
An attempt at a simpler LST binner that makes more assumptions but runs faster.

In particular, we assume that all baselines have the same time array and frequency array,
and that each is present throughout the data array. This allows a vectorization.
"""
import numpy as np
from . import utils
import warnings
from pathlib import Path
from .lstbin import config_lst_bin_files
from . import abscal
import os
from . import io
import logging
import h5py
from hera_qm.metrics_io import read_a_priori_ant_flags
from . import apply_cal
from .datacontainer import DataContainer
from .utils import mergedicts
import gc
from typing import Sequence
from pyuvdata.utils import polnum2str

try:
    profile
except NameError:
    def profile(fnc):
        return fnc

logger = logging.getLogger(__name__)

@profile
def simple_lst_bin(
    data: np.ndarray,
    data_lsts: np.ndarray,
    baselines: list[tuple[int, int]],
    pols: list[str],
    lst_bin_edges: np.ndarray,
    freq_array: np.ndarray,
    flags: np.ndarray | None = None,
    nsamples: np.ndarray | None = None,
    rephase: bool = True,
    antpos: np.ndarray | None = None,
    lat: float = -30.72152,
    out_data: np.ndarray | None = None,
    out_flags: np.ndarray | None = None,
    out_std: np.ndarray | None = None,
    out_counts: np.ndarray | None = None,    
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required_shape = (len(data_lsts), len(baselines), len(freq_array), len(pols))
    if data.shape != required_shape:
        raise ValueError(
            f"data should have shape {required_shape} but got {data.shape}"
        )

    if flags is None:
        flags = np.zeros(data.shape, dtype=bool)

    if flags.shape != required_shape:
        raise ValueError(
            f"flags should have shape {required_shape} but got {flags.shape}"
        )

    if nsamples is None:
        nsamples = np.ones(data.shape, dtype=float)
    
    if nsamples.shape != required_shape:
        raise ValueError(
            f"nsampels should have shape {required_shape} but got {nsamples.shape}"
        )

    if len(lst_bin_edges) < 2:
        raise ValueError("lst_bin_edges must have at least 2 elements")

    # Ensure the lst bin edges start within (0, 2pi)
    while lst_bin_edges[0] < 0:
        lst_bin_edges += 2*np.pi
    while lst_bin_edges[0] >= 2*np.pi:
        lst_bin_edges -= 2*np.pi

    if not np.all(np.diff(lst_bin_edges) > 0):
        raise ValueError(
            "lst_bin_edges must be monotonically increasing."
        )

    # Now ensure that all the observed LSTs are wrapped so they start above the first bin edges
    data_lsts %= 2*np.pi
    data_lsts[data_lsts < lst_bin_edges[0]] += 2* np.pi

    lst_bin_centres = (lst_bin_edges[1:] + lst_bin_edges[:-1])/2

    grid_indices = np.digitize(data_lsts, lst_bin_edges, right=True) - 1

    # Now, any grid index that is less than zero, or len(edges) - 1 is not included in this grid.
    lst_mask = (grid_indices >= 0) & (grid_indices < len(lst_bin_centres))

    # TODO: check whether this creates a data copy. Don't want the extra RAM...
    data = data[lst_mask]  # actually good if this is copied, because we do LST rephase in-place
    flags = flags[lst_mask]
    nsamples = nsamples[lst_mask]
    data_lsts = data_lsts[lst_mask]
    grid_indices = grid_indices[lst_mask]

    logger.info(f"Data Shape: {data.shape}")

    # Now, rephase the data to the lst bin centres.
    if rephase:
        logger.info("Rephasing data")
        if freq_array is None or antpos is None:
            raise ValueError("freq_array and antpos is needed for rephase")

        bls = np.array([antpos[k[0]] - antpos[k[1]] for k in baselines])

        # get appropriate lst_shift for each integration, then rephase
        lst_shift = lst_bin_centres[grid_indices] - data_lsts

        # this makes a copy of the data in d
        utils.lst_rephase_vectorized(data, bls, freq_array, lst_shift, lat=lat, inplace=True)

    # TODO: check for baseline conjugation stuff.

    if out_data is None:
        out_data = np.zeros((len(baselines), len(lst_bin_centres), len(freq_array), len(pols)), dtype=complex)
    if out_flags is None:
        out_flags = np.zeros(out_data.shape, dtype=bool)
    if out_std is None:
        out_std = np.ones(out_data.shape, dtype=complex)
    if out_counts is None:
        out_counts = np.zeros(out_data.shape, dtype=float)

    assert out_data.shape == out_flags.shape == out_std.shape == out_counts.shape
    assert out_data.shape == (len(baselines), len(lst_bin_centres), len(freq_array), len(pols))

    for lstbin in range(len(lst_bin_centres)):
        logger.info(f"Computing LST bin {lstbin+1} / {len(lst_bin_centres)}")
        # TODO: check that this doesn't make yet another copy...
        # This is just the data in this particular lst-bin.
        mask = grid_indices==lstbin
        if np.any(mask):
            d = data[mask]
            n = nsamples[mask]
            f = flags[mask]

            (
                out_data[:, lstbin], 
                out_flags[:, lstbin], 
                out_std[:, lstbin], 
                out_counts[:, lstbin]
            ) = lst_average(d, n, f)
        else:
            out_data[:, lstbin] = 1.0
            out_flags[:, lstbin] = True
            out_std[:, lstbin] = 1.0
            out_counts[:, lstbin] = 0.0
            
    return lst_bin_centres, out_data, flags, out_std, out_counts

@profile
def lst_average(
    data: np.ndarray, nsamples: np.ndarray, flags: np.ndarray, 
    flag_thresh: float = 0.7, median: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # data has shape (ntimes, nbl, npols, nfreqs)
    # all data is assumed to be in the same LST bin.

    assert data.shape == nsamples.shape == flags.shape
    
    flags[np.isnan(data) | np.isinf(data) | (nsamples == 0)] = True

    # Flag entire LST bins if there are too many flags over time
    flag_frac = np.sum(flags, axis=0) / flags.shape[0]
    flags |= flag_frac > flag_thresh

    data[flags] *= np.nan  # do this so that we can do nansum later. multiply to get both real/imag as nan

    # get other stats
    logger.info("Calculating std")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")
        std = np.nanstd(data.real, axis=0) + 1j*np.nanstd(data.imag, axis=0)
        
    nsamples[flags] = 0
    norm = np.sum(nsamples, axis=0)  # missing a "clip" between 1e-99 and inf here...
    
    if median:
        logger.info("Calculating median")
        data = np.nanmedian(data, axis=0)
    else:
        logger.info("Calculating mean")
        data = np.nansum(data * nsamples, axis=0)
        data[norm>0] /= norm[norm>0]
        data[norm<=0] = 1  # any value, it's flagged anyway
        
    f_min = np.all(flags, axis=0)
    std[f_min] = 1.0
    norm[f_min] = 0  # This is probably redundant.

    return data, f_min, std, norm

@profile
def lst_bin_files(
    data_files: list[list[str]], 
    input_cals: list[list[str]] | None = None, 
    dlst: float | None=None, 
    n_lstbins_per_outfile: int=60,
    file_ext: str="{type}.{time:7.5f}.uvh5", 
    outdir: str | Path | None=None, 
    overwrite: bool=False, 
    history: str='', 
    lst_start: float | None=None,
    atol: float=1e-6,  
    rephase: bool=False,
    output_file_select: int | Sequence[int] | None=None, 
    Nbls_to_load: int | None=None, 
    ignore_flags: bool=False, 
    include_autos: bool=True, 
    ex_ant_yaml_files=None, 
    ignore_ants: tuple[int]=(),
    write_kwargs: dict | None = None,
):
    """
    LST bin a series of UVH5 files.
    
    This takes a series of UVH5 files where each file has the same frequency bins and 
    pols, grids them onto a common LST grid, and then averages all integrations
    that appear in that LST bin.

    Output file meta data (frequency bins, antennas positions, time_array)
    are taken from the zeroth file on the last day. Can only LST bin drift-phased data.

    Note: Only supports input data files that have nsample_array == 1, and a single
    integration_time equal to np.diff(time_array), i.e. doesn't support baseline-dependent
    averaging yet. Also, all input files must have the same integration_time, as this
    metadata is taken from zeroth file but applied to all files.

    Parameters:
    -----------
    data_files : type=list of lists: nested set of lists, with each nested list containing
        paths to files from a particular night. Frequency axis of each file must be identical.
        Metadata like x_orientation is inferred from the lowest JD file on the night with the
        highest JDs (i.e. the last night) and assumed to be the same for all files
    dlst : type=float, LST bin width. If None, will get this from the first file in data_files.
    lst_start : type=float, starting LST for binner as it sweeps from lst_start to lst_start + 2pi.
    ntimes_per_file : type=int, number of LST bins in a single output file
    file_ext : type=str, extension to "zen." for output files. This must have at least a ".{type}." field
        where either "LST" or "STD" is inserted for data average or data standard dev., and also a ".{time:7.5f}"
        field where the starting time of the data file is inserted. If this also has a ".{pol}." field, then
        the polarizations of data is also inserted. Example: "{type}.{time:7.5f}.uvh5"
    outdir : type=str, output directory
    overwrite : type=bool, if True overwrite output files
    history : history to insert into output files
    rephase : type=bool, if True, rephase data points in LST bin to center of bin
    bin_kwargs : type=dictionary, keyword arguments for lst_bin.
    atol : type=float, absolute tolerance for LST bin float comparison
    output_file_select : type=int or integer list, list of integer indices of the output files to run on.
        Default is all files.
    input_cals : type=list of lists: nested set of lists matching data_files containing
        filepath to calfits, UVCal or HERACal objects with gain solutions to
        apply to data on-the-fly before binning via hera_cal.apply_cal.calibrate_in_place.
        If no apply cal is desired for a particular file, feed as None in input_cals.
    Nbls_to_load : int, default=None, Number of baselines to load and bin simultaneously. If Nbls exceeds this
        than iterate over an outer loop until all baselines are binned. Default is to load all baselines at once.
    ignore_flags : bool, if True, ignore the flags in the input files, such that all input data in included in binning.
    average_redundant_baselines : bool, if True, baselines that are redundant between and within nights will be averaged together.
        When this is set to true, Nbls_to_load is interpreted as the number of redundant groups
        to load simultaneously. The number of data waterfalls that are loaded can be substantially larger in some
        cases.
    include_autos : bool, if True, include autocorrelations in redundant baseline averages.
                    default is True.
    bl_error_tol : float, tolerance within which baselines are considered redundant
                   between and within nights for purposes of average_redundant_baselines.
    ex_ant_yaml_files : list of strings, optional
        list of paths of yaml files specifying antennas to flag and remove from data on each night.
    kwargs : type=dictionary, keyword arguments to pass to io.write_vis()

    Result:
    -------
    zen.{pol}.LST.{file_lst}.uv : holds LST bin avg (data_array) and bin count (nsample_array)
    zen.{pol}.STD.{file_lst}.uv : holds LST bin stand dev along real and imag (data_array)
    """
    # get file lst arrays
    lst_grid, dlst, file_lsts, begin_lst, lst_arrs, time_arrs = config_lst_bin_files(
        data_files, 
        dlst=dlst, 
        atol=atol, 
        lst_start=lst_start,
        ntimes_per_file=n_lstbins_per_outfile, 
        verbose=False
    )

    nfiles = len(file_lsts)

    logger.info("Setting output files")

    # select file_lsts
    if output_file_select is not None:
        if isinstance(output_file_select, (int, np.integer)):
            output_file_select = [output_file_select]
        output_file_select = [int(o) for o in output_file_select]
        try:
            file_lsts = [file_lsts[i] for i in output_file_select]
        except IndexError:
            warnings.warn(
                f"One or more indices in output_file_select {output_file_select} "
                f"caused an index error with length {nfiles} file_lsts list, exiting..."
            )
            return

    # get metadata from the zeroth data file in the last day
    last_day_index = np.argmax([np.min([time for tarr in tarrs for time in tarr]) for tarrs in time_arrs])
    zeroth_file_on_last_day_index = np.argmin([np.min(tarr) for tarr in time_arrs[last_day_index]])

    logger.info("Getting metadata from last data...")    
    hd = io.HERAData(data_files[last_day_index][zeroth_file_on_last_day_index])
    x_orientation = hd.x_orientation

    # get metadata
    freq_array = hd.freqs
    antpos = hd.antpos
    times = hd.times
    start_jd = np.floor(times.min())
    integration_time = np.median(hd.integration_time)
    if not  np.all(np.abs(np.diff(times) - np.median(np.diff(times))) < 1e-6):
        raise ValueError('All integrations must be of equal length (BDA not supported)')

    logger.info("Compiling all unflagged baselines...")
    all_baselines, all_pols, fls_with_ants = get_all_unflagged_baselines(
        data_files, 
        ex_ant_yaml_files, 
        include_autos=include_autos, 
        ignore_ants=ignore_ants
    )
    all_baselines = sorted(all_baselines)

    antpos = get_all_antpos_from_files(fls_with_ants, all_baselines)
    # Split up the baselines into chunks that will be LST-binned together.
    # This is just to save on RAM.
    if Nbls_to_load is None:
        Nbls_to_load = len(all_baselines) + 1
    n_bl_chunks = len(all_baselines) // Nbls_to_load + 1
    bl_chunks = [all_baselines[i * Nbls_to_load:(i + 1) * Nbls_to_load] for i in range(n_bl_chunks)]
    bl_chunks = [blg for blg in bl_chunks if len(blg) > 0]

    # iterate over output LST files
    for i, outfile_lsts in enumerate(file_lsts):
        logger.info(f"LST file {i+1} / {len(file_lsts)}")

        outfile_lst_min = outfile_lsts[0] - (dlst / 2 + atol)
        outfile_lst_max = outfile_lsts[-1] + (dlst / 2 + atol)
        
        tinds = []
        all_lsts = []
        file_list = []
        time_arrays = []
        cals = []
        # This loop just gets the number of times that we'll be reading.
        for night, night_files in enumerate(data_files):
            # iterate over files in each night, and open files that fall into this output file LST range
            
            for k_file, fl in enumerate(night_files):
                
                # unwrap la relative to itself
                larr = lst_arrs[night][k_file]
                larr[larr < larr[0]] += 2 * np.pi

                # phase wrap larr to get it to fall within 2pi of file_lists
                while larr[0] + 2 * np.pi < outfile_lst_max:
                    larr += 2 * np.pi
                while larr[-1] - 2 * np.pi > outfile_lst_min:
                    larr -= 2 * np.pi

                tind = (larr > outfile_lst_min) & (larr < outfile_lst_max)

                if np.any(tind):
                    tinds.append(tind)
                    time_arrays.append(time_arrs[night][k_file][tind])
                    all_lsts.append(larr[tind])
                    file_list.append(fl)
                    if input_cals is not None:
                        cals.append(input_cals[night][k_file])
                    else:
                        cals.append(None)

        all_lsts = np.concatenate(all_lsts)

        # If we have no times at all for this bin, just continue to the next bin.
        if len(all_lsts) == 0:
            continue
        
        # iterate over baseline groups (for memory efficiency)
        out_data = np.zeros((len(all_baselines), len(outfile_lsts), len(freq_array), len(all_pols)), dtype='complex')
        out_stds = np.zeros_like(out_data)
        out_nsamples = np.zeros(out_data.shape, dtype=float)
        out_flags = np.zeros(out_data.shape, dtype=bool)

        nbls_so_far = 0
        for bi, bl_chunk in enumerate(bl_chunks):
            logger.info(f"Baseline Chunk {bi+1} / {len(bl_chunks)}")

            # Now we can set up our master arrays of data. 
            data = np.full((
                len(all_lsts), len(bl_chunk), len(hd.freqs), len(all_pols)), 
                np.nan+np.nan*1j, dtype=complex
            )
            flags = np.ones(data.shape, dtype=bool)
            nsamples = np.zeros(data.shape, dtype=float)

            # This loop actually reads the associated data in this LST bin.
            ntimes_so_far = 0
            for fl, calfl, tind, tarr in zip(file_list, cals, tinds, time_arrays):
                hd = io.HERAData(fl, filetype='uvh5')

                bls_to_load = [bl for bl in bl_chunk if bl in hd.antpairs]
                _data, _flags, _nsamples  = hd.read(
                    bls=bls_to_load, 
                    times=tarr
                )

                # load calibration
                if calfl is not None:
                    logger.info(f"Opening and applying {calfl}")
                    uvc = io.to_HERACal(calfl)
                    gains, cal_flags, _, _ = uvc.read()
                    # down select times in necessary
                    if False in tind and uvc.Ntimes > 1:
                        # If uvc has Ntimes == 1, then broadcast across time will work automatically
                        uvc.select(times=uvc.time_array[tind])
                        gains, cal_flags, _, _ = uvc.build_calcontainers()
                    
                    apply_cal.calibrate_in_place(
                        _data, gains, data_flags=_flags, cal_flags=cal_flags,
                        gain_convention=uvc.gain_convention
                    )

                slc = slice(ntimes_so_far,ntimes_so_far+_data.shape[0])
                for i, bl in enumerate(bl_chunk):
                    for j, pol in enumerate(all_pols):
                        if bl + (pol,) in _data:
                            data[slc, i, :, j] = _data[bl+(pol,)]
                            flags[slc, i, :, j] = _flags[bl+(pol,)]
                            nsamples[slc, i, :, j] = _nsamples[bl+(pol,)]
                        else:
                            # This baseline+pol doesn't exist in this file. That's
                            # OK, we don't assume all baselines are in every file.
                            data[slc, i, :, j] = np.nan
                            flags[slc, i, :, j] = True
                            nsamples[slc, i, :, j] = 0

                ntimes_so_far += _data.shape[0]

            logger.info("About to run LST binning...")
            # LST bin edges are the actual edges of the bins, so should have length
            # +1 of the LST centres. We use +dlst instead of +dlst/2 on the top edge
            # so that np.arange definitely gets the last edge.
            lst_edges = np.arange(outfile_lsts[0] - dlst/2, outfile_lsts[-1] + dlst, dlst)
            bin_lst, _, _, _, _ = simple_lst_bin(
                data=data, 
                flags=None if ignore_flags else flags,
                nsamples=nsamples,
                data_lsts=all_lsts,
                baselines=bl_chunk,
                pols=all_pols,
                lst_bin_edges=lst_edges,
                freq_array = hd.freqs,
                rephase = rephase,
                antpos=antpos,
                out_counts=out_nsamples[nbls_so_far:nbls_so_far + len(bl_chunk)],
                out_data=out_data[nbls_so_far:nbls_so_far + len(bl_chunk)],
                out_flags=out_flags[nbls_so_far:nbls_so_far + len(bl_chunk)],
                out_std=out_stds[nbls_so_far:nbls_so_far + len(bl_chunk)]
            )
            
            nbls_so_far += len(bl_chunk)

        
        logger.info("Writing output files")

        # get outdir
        if outdir is None:
            outdir = os.path.dirname(os.path.commonprefix(abscal.flatten(data_files)))

        # update kwrgs
        # If None, set to empty dict
        write_kwargs = write_kwargs or {}

        # update history
        file_list_str = "-".join(os.path.basename(ff)for ff in file_list)
        file_history = f"{history} Input files: {file_list_str}"
        _history = file_history + utils.history_string()

        # form integration time array
        integration_time = integration_time*np.ones(
            len(bin_lst) * len(all_baselines), 
            dtype=np.float64
        )

        # file in data ext
        fkwargs = {"type": "LST", "time": bin_lst[0] - dlst / 2.0}
        if "{pol}" in file_ext:
            fkwargs['pol'] = '.'.join(all_pols)

        # configure filenames
        bin_file = "zen." + file_ext.format(**fkwargs)
        fkwargs['type'] = 'STD'
        std_file = "zen." + file_ext.format(**fkwargs)

        # check for overwrite
        if os.path.exists(bin_file) and overwrite is False:
            logger.warning(f"{bin_file} exists, not overwriting")
            continue

        write_kwargs.update(
            lst_array=bin_lst,
            freq_array=freq_array,
            antpos=antpos,
            pols=all_pols,
            antpairs=all_baselines,
            flags=out_flags,
            nsamples=out_nsamples,
            x_orientation=x_orientation,
            integration_time=integration_time,
            history=_history,
            start_jd=start_jd,
            lst_branch_cut=write_kwargs.get('lst_branch_cut', lst_start or file_lsts[0][0])
        )
        uvd_data = io.create_uvd_from_hera_data(data = out_data, **write_kwargs)
        uvd_data.write_uvh5(os.path.join(outdir, bin_file), clobber=overwrite)

        uvd_data = io.create_uvd_from_hera_data(data = out_stds, **write_kwargs)
        uvd_data.write_uvh5(os.path.join(outdir, std_file), clobber=overwrite)
        
@profile
def get_all_unflagged_baselines(
    data_files: list[list[str | Path]], 
    ex_ant_yaml_files: list[str] | None = None,
    include_autos: bool = True,
    ignore_ants: tuple[int] = (),
    only_last_file_per_night: bool = False,
) -> tuple[set[tuple[int, int]], list[str]]:
    """Generate a set of all antpairs that have at least one un-flagged entry.
    
    This is performed over a list of nights, each of which consists of a list of 
    individual uvh5 files. Each UVH5 file is *assumed* to have the same set of times
    for each baseline internally (difference nights obviously have different times).
    
    Returns
    -------
    all_baselines
        The set of all antpairs in all files in the given list.
    all_pols
        A list of all polarizations in the files in the given list, as strings like 
        'ee' and 'nn' (i.e. with x_orientation information).
    """
    all_pols = set()
    xorient_bytes = None

    all_baselines = set()
    files_with_ants = set()
    unique_ants = set()

    for night, fl_list in enumerate(data_files):
        if ex_ant_yaml_files:
            a_priori_antenna_flags = read_a_priori_ant_flags(
                ex_ant_yaml_files[night], ant_indices_only=True
            )
        else:
            a_priori_antenna_flags = set()

        if only_last_file_per_night:
            fl_list = fl_list[-1:]

        for fl in fl_list:
            # To go faster, let's JUST read the antpairs and pols from the files.
            with h5py.File(fl, 'r') as hfl:
                ntimes= int(hfl['Header']['Ntimes'][()])
                nblts = int(hfl['Header']['Nblts'][()])
                if nblts % ntimes:
                    raise ValueError(f'Datafile {fl} has different number of times for different baselines!')

                times = hfl['Header']["time_array"][:2]

                if times[0] != times[0]:
                    # Assume time-first ordering.
                    ant1 = hfl['Header']['ant_1_array'][::ntimes]
                    ant2 = hfl['Header']['ant_2_array'][::ntimes]
                else:
                    nbls = nblts // ntimes
                    ant1 = hfl['Header']['ant_1_array'][:nbls]
                    ant2 = hfl['Header']['ant_2_array'][:nbls]

                all_pols.update(list(hfl['Header']["polarization_array"][:]))
                xb = bytes(hfl['Header']["x_orientation"][()])
                if xorient_bytes is not None and xorient_bytes != xb:
                    raise ValueError("Not all input files have the same x_orientation!")
                xorient_bytes = xb

            for a1, a2 in zip(ant1, ant2):
                if (
                    (a1, a2) not in all_baselines and # Do this first because after the
                    (a2, a1) not in all_baselines and # first file it often triggers.
                    a1 not in ignore_ants and 
                    a2 not in ignore_ants and 
                    (include_autos or a1 != a2) and 
                    a1 not in a_priori_antenna_flags and 
                    a2 not in a_priori_antenna_flags
                ):  
                    all_baselines.add((a1, a2))


                    if a1 not in unique_ants:
                        unique_ants.add(a1)
                        files_with_ants.add(fl)
                    if a2 not in unique_ants:
                        unique_ants.add(a2)
                        files_with_ants.add(fl)
                    

    return all_baselines, [polnum2str(p, x_orientation=xorient_bytes.decode("utf8")) for p in all_pols], files_with_ants


def get_all_antpos_from_files(
    data_files: list[str | Path], 
    all_baselines: list[tuple[int, int]]
) -> dict[tuple[int, int], np.ndarray]:

    antpos_out = {}
    
    # ants will be a set of integers antenna numbers.
    ants = set(sum(all_baselines, start=()))

    for fl in data_files:
        # unfortunately, we're reading in way more than we need to here,
        # because the conversion from the antpos in the file to the ENU antpos is 
        # non trivial and hard to trace in uvdata.
        hd = io.HERAData(fl)

        for ant in ants:
            if ant in hd.antpos and ant not in antpos_out:
                antpos_out[ant] = hd.antpos[ant]

    return antpos_out

