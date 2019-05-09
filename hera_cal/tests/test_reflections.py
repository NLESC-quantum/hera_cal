# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import

import unittest
import nose.tools as nt
import numpy as np
from copy import deepcopy
import os
import sys
import shutil
from six.moves import zip
from scipy import stats
from pyuvdata import UVCal, UVData
import operator
import functools
from sklearn.gaussian_process import kernels
import hera_sim as hs
import copy

from hera_cal import io
from hera_cal import reflections
from hera_cal import datacontainer
from hera_cal.data import DATA_PATH
from hera_cal import apply_cal


def simulate_reflections(uvd=None, camp=1e-2, cdelay=155, cphase=2, add_cable=True, cable_ants=None,
                         xamp=1e-2, xdelay=300, xphase=0, add_xtalk=False):
    # create a simulated dataset
    if uvd is None:
        uvd = UVData()
        uvd.read(os.path.join(DATA_PATH, 'PyGSM_Jy_downselect.uvh5'))
    else:
        if isinstance(uvd, (str, np.str)):
            _uvd = UVData()
            _uvd.read(uvd)
            uvd = _uvd
        elif isinstance(uvd, UVData):
            uvd = deepcopy(uvd)

    # TODO: use hera_sim.simulate.Simulator
    freqs = np.unique(uvd.freq_array)
    Nbls = len(np.unique(uvd.baseline_array))

    if cable_ants is None:
        cable_ants = uvd.antenna_numbers

    def noise(n, sig):
        return stats.norm.rvs(0, sig / np.sqrt(2), n) + 1j * stats.norm.rvs(0, sig / np.sqrt(2), n)

    np.random.seed(0)

    # get antenna vectors
    antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=True)
    antpos_d = dict(zip(ants, antpos))
    ant_dist = dict(zip(ants, map(np.linalg.norm, antpos)))

    # get autocorr
    autocorr = uvd.get_data(23, 23, 'xx')

    # form cable gains
    if add_cable:
        if isinstance(cdelay, (float, np.float, int, np.int)):
            cdelay = [cdelay]
        if isinstance(camp, (float, np.float, int, np.int)):
            camp = [camp]
        if isinstance(cphase, (float, np.float, int, np.int)):
            cphase = [cphase]

        cable_gains = dict([(k, np.ones((uvd.Ntimes, uvd.Nfreqs), dtype=np.complex)) for k in uvd.antenna_numbers])

        for ca, cd, cp in zip(camp, cdelay, cphase):
            cg = hs.sigchain.gen_reflection_gains(freqs / 1e9, cable_ants, amp=[ca for a in cable_ants],
                                                  dly=[cd for a in cable_ants], phs=[cp for a in cable_ants])
            for k in cg:
                cable_gains[k] *= cg[k]

    # iterate over bls
    for i, bl in enumerate(np.unique(uvd.baseline_array)):
        bl_inds = np.where(uvd.baseline_array == bl)[0]
        antpair = uvd.baseline_to_antnums(bl)

        # add xtalk
        if add_xtalk:
            if antpair[0] != antpair[1]:
                # add xtalk to both pos and neg delays
                xt = hs.sigchain.gen_cross_coupling_xtalk(freqs / 1e9, autocorr, amp=xamp, dly=xdelay, phs=xphase)
                xt += hs.sigchain.gen_cross_coupling_xtalk(freqs / 1e9, autocorr, amp=xamp, dly=xdelay, phs=xphase, conj=True)
                uvd.data_array[bl_inds, 0] += xt[:, :, None]

        # add a cable reflection term eps_11
        if add_cable:
            gain = cable_gains[antpair[0]] * np.conj(cable_gains[antpair[1]])
            uvd.data_array[bl_inds, 0] *= gain[:, :, None]

    # get fourier modes
    uvd.frates = np.fft.fftshift(np.fft.fftfreq(uvd.Ntimes, np.diff(np.unique(uvd.time_array))[0] * 24 * 3600)) * 1e3
    uvd.delays = np.fft.fftshift(np.fft.fftfreq(uvd.Nfreqs, uvd.channel_width)) * 1e9

    return uvd


class Test_ReflectionFitter_Cables(unittest.TestCase):
    uvd_clean = simulate_reflections(add_cable=False, add_xtalk=False)
    uvd = simulate_reflections(cdelay=255.0, cphase=2.0, camp=1e-2, add_cable=True, cable_ants=[23], add_xtalk=False)

    def test_model_auto_reflections(self):
        RF = reflections.ReflectionFitter(self.uvd)
        bl_k = (23, 23, 'xx')
        g_k = (23, 'Jxx')
        RF.fft_data(window='blackmanharris', overwrite=True, ax='freq')  # for inspection

        # basic run through
        RF.model_auto_reflections(RF.data, (200, 300), keys=[bl_k], window='blackmanharris',
                                  zeropad=100, overwrite=True, fthin=1, verbose=True)
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_dly.values())), 255.0, atol=1e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_amp.values())), 1e-2, atol=1e-4).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_phs.values())), 2.0, atol=1e-1).all())

        # try with a small edgecut
        RF = reflections.ReflectionFitter(self.uvd)
        edgecut = 5
        RF.model_auto_reflections(RF.data, (200, 300), keys=[bl_k], window='blackmanharris', reject_edges=False,
                                  zeropad=100, overwrite=True, fthin=1, verbose=True, edgecut_low=edgecut, edgecut_hi=edgecut)
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_dly.values())), 255.0, atol=1e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_amp.values())), 1e-2, atol=1e-4).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_phs.values())), 2.0, atol=1e-1).all())

        # try a high ref_sig cut: assert ref_flags are True
        RF.model_auto_reflections(RF.data, (200, 300), keys=[bl_k], window='blackmanharris',
                                  ref_sig_cut=100, overwrite=True)
        nt.assert_true(RF.ref_flags[g_k].all())

        # assert refinement uses flags to return zeros
        output = RF.refine_auto_reflections(RF.data, (20, 80), RF.ref_amp, RF.ref_dly, RF.ref_phs,
                                            keys=[bl_k], ref_flags=RF.ref_flags, window='blackmanharris', zeropad=100,
                                            maxiter=100, method='Nelder-Mead', tol=1e-5)
        nt.assert_true(np.isclose(output[0][g_k], 0.0).all())

        # try filtering the visibilities
        RF.vis_clean(data=RF.data, ax='freq', min_dly=100, overwrite=True, window='blackmanharris', alpha=0.1, tol=1e-8, keys=[bl_k])
        RF.model_auto_reflections(RF.clean_resid, (200, 300), clean_data=RF.clean_data, keys=[bl_k],
                                  window='blackmanharris', zeropad=100, overwrite=True, fthin=1, verbose=True)
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_dly.values())), 255.0, atol=1e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_amp.values())), 1e-2, atol=1e-4).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_phs.values())), 2.0, atol=1e-1).all())

        # try optimization on time-averaged data
        RF.timeavg_data(RF.data, RF.times, RF.lsts, 5000, keys=None, overwrite=True)
        RF.model_auto_reflections(RF.avg_data, (200, 300), keys=[bl_k], window='blackmanharris',
                                  zeropad=100, overwrite=True, fthin=1, verbose=True)
        output = RF.refine_auto_reflections(RF.avg_data, (20, 80), RF.ref_amp, RF.ref_dly, RF.ref_phs,
                                            keys=[bl_k], window='blackmanharris', zeropad=100,
                                            maxiter=100, method='Nelder-Mead', tol=1e-5)
        ref_amp = output[0]
        ref_dly = output[1]
        ref_phs = output[2]
        # assert equivalence to higher precision
        nt.assert_true(np.isclose(np.ravel(list(ref_dly.values())), 255.0, atol=1e-2).all())
        nt.assert_true(np.isclose(np.ravel(list(ref_amp.values())), 1e-2, atol=1e-5).all())
        nt.assert_true(np.isclose(np.ravel(list(ref_phs.values())), 2.0, atol=1e-2).all())

        # now reverse delay range
        RF.model_auto_reflections(RF.avg_data, (-300, -200), keys=[bl_k], window='blackmanharris',
                                  zeropad=100, overwrite=True, fthin=1, verbose=True)
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_dly.values())), -255.0, atol=1e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_amp.values())), 1e-2, atol=1e-4).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_phs.values())), 2 * np.pi - 2.0, atol=1e-1).all())

        output = RF.refine_auto_reflections(RF.avg_data, (80, 20), RF.ref_amp, RF.ref_dly, RF.ref_phs,
                                            keys=[bl_k, (39, 39, 'xx')], window='blackmanharris', zeropad=100,
                                            maxiter=100, method='BFGS', tol=1e-5)
        ref_amp = output[0]
        ref_dly = output[1]
        ref_phs = output[2]
        # assert equivalence to higher precision
        nt.assert_true(np.isclose(np.ravel(list(ref_dly.values())), -255.0, atol=1e-2).all())
        nt.assert_true(np.isclose(np.ravel(list(ref_amp.values())), 1e-2, atol=1e-5).all())
        nt.assert_true(np.isclose(np.ravel(list(ref_phs.values())), 2 * np.pi - 2.0, atol=1e-2).all())

        # test flagged data
        RF.model_auto_reflections(RF.avg_data, (-300, -200), keys=[bl_k], window='blackmanharris',
                                  zeropad=100, overwrite=True, fthin=1, verbose=True)
        RF.avg_flags[bl_k][:] = True
        output = RF.refine_auto_reflections(RF.avg_data, (80, 20), RF.ref_amp, RF.ref_dly, RF.ref_phs,
                                            keys=[bl_k], window='blackmanharris', zeropad=100, clean_flags=RF.avg_flags,
                                            maxiter=100, method='BFGS', tol=1e-5)
        nt.assert_false(output[3][(23, 'Jxx')].any())
        RF.avg_flags[bl_k][:] = False

        # non-even Nfreqs
        RF = reflections.ReflectionFitter(self.uvd.select(frequencies=np.unique(self.uvd.freq_array)[:-1], inplace=False))
        RF.model_auto_reflections(RF.data, (200, 300), keys=[bl_k], window='blackmanharris',
                                  zeropad=100, overwrite=True, fthin=1, verbose=True)
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_dly.values())), 255.0, atol=1e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_amp.values())), 1e-2, atol=1e-4).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_phs.values())), 2.0, atol=1e-1).all())

        # exceptions
        nt.assert_raises(ValueError, RF.model_auto_reflections, RF.data, (4000, 5000), window='none', overwrite=True, edgecut_low=edgecut)

        # test reject_edges: choose dly_range to make max on edge
        # assert peak is in main lobe, not at actual reflection delay
        RF.model_auto_reflections(RF.data, (25, 300), keys=[bl_k], window='blackmanharris', reject_edges=False,
                                  zeropad=100, overwrite=True, fthin=1, verbose=True)
        nt.assert_true(np.all(np.ravel(list(RF.ref_dly.values())) < 200))
        # assert peak is correct
        RF.model_auto_reflections(RF.data, (25, 300), keys=[bl_k], window='blackmanharris', reject_edges=True,
                                  zeropad=100, overwrite=True, fthin=1, verbose=True)
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_dly.values())), 255.0, atol=1e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_amp.values())), 1e-2, atol=1e-4).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_phs.values())), 2.0, atol=1e-1).all())
        # assert valley results in flagged reflection (make sure zeropad=0)
        RF.model_auto_reflections(RF.data, (25, 225), keys=[bl_k], window='blackmanharris', reject_edges=True,
                                  zeropad=0, overwrite=True, fthin=1, verbose=True)
        nt.assert_true(np.all(RF.ref_flags[g_k]))

        # try clear
        RF.clear(exclude=['data'])
        nt.assert_equal(len(RF.ref_eps), 0)
        nt.assert_equal(len(RF.ref_gains), 0)
        nt.assert_true(len(RF.data) > 0)

        # try soft copy
        RF2 = RF.soft_copy()
        nt.assert_true(RF2.__class__, reflections.ReflectionFitter)

    def test_write_auto_reflections(self):
        RF = reflections.ReflectionFitter(self.uvd)
        bl_k = (23, 23, 'xx')
        RF.model_auto_reflections(RF.data, (200, 300), window='blackmanharris', zeropad=100, overwrite=True, fthin=1, verbose=True)
        uvc = RF.write_auto_reflections("./ex.calfits", overwrite=True)
        nt.assert_equal(uvc.Ntimes, 100)
        np.testing.assert_array_equal(len(uvc.ant_array), 65)
        nt.assert_true(np.isclose(uvc.gain_array[0], 1.0).all())
        nt.assert_false(np.isclose(uvc.gain_array[uvc.ant_array.tolist().index(23)], 1.0).all())

        # test w/ input calfits
        uvc = RF.write_auto_reflections("./ex.calfits", input_calfits="./ex.calfits", overwrite=True)
        RF.model_auto_reflections(RF.data, (200, 300), window='blackmanharris', zeropad=100, overwrite=True, fthin=1, verbose=True)
        uvc = RF.write_auto_reflections("./ex.calfits", input_calfits='./ex.calfits', overwrite=True)
        nt.assert_equal(uvc.Ntimes, 100)
        np.testing.assert_array_equal(len(uvc.ant_array), 65)

        # test data is corrected by taking ratio w/ clean data
        data = deepcopy(RF.data)
        g = reflections.form_gains(RF.ref_eps)
        apply_cal.calibrate_in_place(data, g, gain_convention='divide')
        r = data[bl_k] / self.uvd_clean.get_data(bl_k)
        nt.assert_true(np.abs(np.mean(r) - 1) < 1e-1)

        os.remove('./ex.calfits')

    def test_auto_reflection_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--output_fname', 'ex.calfits', '--dly_ranges', '10,20', '10,20', '--overwrite', '--opt_buffer', '25', '75']
        parser = reflections.auto_reflection_argparser()
        a = parser.parse_args()
        nt.assert_equal(a.data[0], 'a')
        nt.assert_equal(a.output_fname, 'ex.calfits')
        nt.assert_equal(a.dly_ranges[0], '10,20')
        nt.assert_equal(len(a.dly_ranges), 2)
        nt.assert_true(np.isclose(a.opt_buffer, [25, 75]).all())

    def test_auto_reflection_run(self):
        # most of the code tests have been done above, this is just to ensure this wrapper function runs
        uvd = simulate_reflections(cdelay=[150.0, 250.0], cphase=[0.0, 0.0], camp=[1e-2, 1e-2], add_cable=True, cable_ants=[23], add_xtalk=False)
        reflections.auto_reflection_run(uvd, [(100, 200), (200, 300)], "./ex.calfits", time_avg=True, window='blackmanharris', write_npz=True, overwrite=True, ref_sig_cut=1.0)
        nt.assert_true(os.path.exists("./ex.calfits"))
        nt.assert_true(os.path.exists("./ex.npz"))
        nt.assert_true(os.path.exists("./ex.ref2.calfits"))
        nt.assert_true(os.path.exists("./ex.ref2.npz"))

        # ensure gains have two humps at 150 and 250 ns
        uvc = UVCal()
        uvc.read_calfits('./ex.calfits')
        uvc2 = UVCal()
        uvc2.read_calfits('./ex.ref2.calfits')
        uvc.gain_array *= uvc2.gain_array
        aind = np.argmin(np.abs(uvc.ant_array - 23))
        g = uvc.gain_array[aind, 0, :, :, 0].T
        delays = np.fft.fftfreq(uvc.Nfreqs, np.diff(uvc.freq_array[0])[0]) * 1e9
        gfft = np.mean(np.abs(np.fft.fft(g, axis=1)), axis=0)

        nt.assert_equal(delays[np.argmax(gfft * ((delays > 100) & (delays < 200)))], 150)
        nt.assert_equal(delays[np.argmax(gfft * ((delays > 200) & (delays < 300)))], 250)

        os.remove("./ex.calfits")
        os.remove("./ex.npz")
        os.remove("./ex.ref2.calfits")
        os.remove("./ex.ref2.npz")


class Test_ReflectionFitter_XTalk(unittest.TestCase):
    # simulate
    uvd = simulate_reflections(add_cable=False, xdelay=250.0, xphase=0, xamp=1e-3, add_xtalk=True)

    def test_svd_functions(self):
        RF = reflections.ReflectionFitter(self.uvd)
        bl = (23, 24, 'xx')

        # fft data
        RF.fft_data(data=RF.data, window='blackmanharris', overwrite=True)

        # test sv_decomposition on positive side
        wgts = RF.svd_weights(RF.dfft, RF.delays, min_dly=200, max_dly=300, side='pos')
        RF.sv_decomp(RF.dfft, wgts=wgts, keys=[bl], overwrite=True)

        # build a model
        RF.build_pc_model(RF.umodes, RF.vmodes, RF.svals, Nkeep=1, increment=False, overwrite=True)

        # test containers exist
        nt.assert_true(np.all([hasattr(RF, o) for o in ['umodes', 'vmodes', 'svals', 'uflags', 'pcomp_model', 'dfft']]))
        # test good information compression
        nt.assert_true(RF.svals[bl][0] / RF.svals[bl][1] > 20)

        # assert its a good fit to the xtalk at 250 ns delay
        ind = np.argmin(np.abs(RF.delays - 250))
        Vrms = np.sqrt(np.mean(RF.dfft[bl][:, ind].real**2))
        Rrms = np.sqrt(np.mean((RF.dfft[bl][:, ind].real - RF.pcomp_model[bl][:, ind].real)**2))
        # says that residual is small compared to original array
        nt.assert_true(Rrms / Vrms < 0.01)

        # increment the model
        wgts = RF.svd_weights(RF.dfft, RF.delays, min_dly=200, max_dly=300, side='neg')
        RF.sv_decomp(RF.dfft, wgts=wgts, overwrite=True)
        RF.build_pc_model(RF.umodes, RF.vmodes, RF.svals, Nkeep=1, increment=True)

        # says that the two are similar to each other at -250 ns, which they should be
        ind = np.argmin(np.abs(RF.delays - -250))
        Vrms = np.sqrt(np.mean(RF.dfft[bl][:, ind].real**2))
        Rrms = np.sqrt(np.mean((RF.dfft[bl][:, ind].real - RF.pcomp_model[bl][:, ind].real)**2))
        # says that residual is small compared to original array
        nt.assert_true(Rrms / Vrms < 0.01)

        # overwrite the model with double side modeling
        wgts = RF.svd_weights(RF.dfft, RF.delays, min_dly=200, max_dly=300, side='both')
        RF.sv_decomp(RF.dfft, wgts=wgts, overwrite=True)
        RF.build_pc_model(RF.umodes, RF.vmodes, RF.svals, Nkeep=2, increment=False, overwrite=True)
        # says the residual is small compared to original array
        ind = np.argmin(np.abs(RF.delays - 250))
        Vrms = np.sqrt(np.mean(RF.dfft[bl][:, ind].real**2))
        Rrms = np.sqrt(np.mean((RF.dfft[bl][:, ind].real - RF.pcomp_model[bl][:, ind].real)**2))
        nt.assert_true(Rrms / Vrms < 0.01)

        # subtract the model from the data
        RF.subtract_model(RF.data, overwrite=True)
        nt.assert_equal(RF.pcomp_model_fft[bl].shape, (100, 128))
        nt.assert_equal(RF.data_pcmodel_resid[bl].shape, (100, 128))

    def test_misc_svd_funcs(self):
        # setup RF object
        RF = reflections.ReflectionFitter(self.uvd)
        # add noise
        np.random.seed(0)
        Namp = 3e0
        for k in RF.data:
            RF.data += stats.norm.rvs(0, Namp, RF.Ntimes * RF.Nfreqs).reshape(RF.Ntimes, RF.Nfreqs) + 1j * stats.norm.rvs(0, Namp, RF.Ntimes * RF.Nfreqs).reshape(RF.Ntimes, RF.Nfreqs)
        bl = (23, 24, 'xx')

        # fft data
        RF.fft_data(data=RF.data, window='blackmanharris', overwrite=True)

        # sv decomp
        svd_wgts = RF.svd_weights(RF.dfft, RF.delays, min_dly=150, max_dly=500, side='both')
        RF.sv_decomp(RF.dfft, wgts=svd_wgts, keys=[bl], overwrite=True, Nkeep=None)
        nt.assert_equal(RF.umodes[bl].shape, (100, 100))
        nt.assert_equal(RF.vmodes[bl].shape, (100, 128))

        # test interpolation of umodes
        gp_frate = 0.2
        RF.interp_u(RF.umodes, RF.times, overwrite=True, gp_frate=gp_frate, gp_nl=1e-10, optimizer=None, Ninterp=None)
        nt.assert_equal(RF.umode_interp[bl].shape, (100, 100))
        RF.interp_u(RF.umodes, RF.times, overwrite=True, gp_frate=gp_frate, gp_nl=1e-10, optimizer=None, Ninterp=10)
        nt.assert_equal(RF.umode_interp[bl].shape, (100, 10))

        # get fft and assert a good match within gp_frate
        RF.fft_data(data=RF.umodes, assign='ufft', window='blackmanharris', ax='time', overwrite=True, edgecut_low=5, edgecut_hi=5)
        RF.fft_data(data=RF.umode_interp, assign='uifft', window='blackmanharris', ax='time', overwrite=True, edgecut_low=5, edgecut_hi=5)
        select = np.abs(RF.frates) < gp_frate / 2
        nt.assert_true(np.mean(np.abs(RF.ufft[bl][select, 0] - RF.uifft[bl][select, 0]) / np.abs(RF.ufft[bl][select, 0])) < 0.01)
        # plt.plot(RF.frates, np.abs(RF.ufft[bl][:, 0]));plt.plot(RF.frates, np.abs(RF.uifft[bl][:, 0]));plt.yscale('log')

        # test mode projection after interpolation (smoothing)
        umodes = copy.deepcopy(RF.umodes)
        for k in umodes:
            umodes[k][:, :10] = RF.umode_interp[k][:, :10]  # fill in umodes with smoothed components
        vmodes = RF.project_svd_modes(RF.dfft * svd_wgts, umodes=umodes, svals=RF.svals)

        # build systematic models with original vmodes and projected vmodes
        RF.build_pc_model(umodes, RF.vmodes, RF.svals, overwrite=True, Nkeep=10)
        pcomp1 = RF.pcomp_model[bl]
        RF.build_pc_model(umodes, vmodes, RF.svals, overwrite=True, Nkeep=10)
        pcomp2 = RF.pcomp_model[bl]
        # assert pcomp model with projected vmode has less noise in it for a delay with no systematic
        ind = np.argmin(np.abs(RF.delays - 400))  # no systematic at this delay, only noise
        nt.assert_true(np.mean(np.abs(pcomp1[:, ind])) > np.mean(np.abs(pcomp2[:, ind])))

        # test projection of other SVD matrices
        _svals = RF.project_svd_modes(RF.dfft * svd_wgts, umodes=RF.umodes, vmodes=RF.vmodes)
        _umodes = RF.project_svd_modes(RF.dfft * svd_wgts, vmodes=RF.vmodes, svals=RF.svals)

        # assert original is nearly the same as projected
        nt.assert_true(np.all(np.isclose(_svals[bl], RF.svals[bl], atol=1e-10)))  
        nt.assert_true(np.all(np.isclose(_umodes[bl][:, 0], RF.umodes[bl][:, 0], atol=1e-10)))  

        # assert custom kernel works
        gp_len = 1.0 / (0.4 * 1e-3) / (24.0 * 3600.0)
        kernel = 1**2 * kernels.RBF(gp_len) + kernels.WhiteKernel(1e-10)
        RF.interp_u(RF.umodes, RF.times, overwrite=True, kernels=kernel, optimizer=None)
        nt.assert_equal(RF.umode_interp[bl].shape, (100, 100))

        # assert broadcasting to full time resolution worked
        RF.timeavg_data(RF.data, RF.times, RF.lsts, 500, overwrite=True, verbose=False)
        RF.fft_data(data=RF.avg_data, window='blackmanharris', overwrite=True, assign='adfft', dtime=np.diff(RF.avg_times)[0] * 24 * 3600)
        wgts = RF.svd_weights(RF.adfft, RF.delays, min_dly=200, max_dly=300, side='both')
        RF.sv_decomp(RF.adfft, wgts=wgts, keys=[bl], overwrite=True)
        nt.assert_equal(RF.umodes[bl].shape, (34, 34))
        RF.interp_u(RF.umodes, RF.avg_times, full_times=RF.times, overwrite=True, gp_frate=1.0, gp_nl=1e-10, optimizer=None)
        nt.assert_equal(RF.umode_interp[bl].shape, (100, 34))


if __name__ == '__main__':
    unittest.main()
