# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import pytest
import numpy as np
from copy import deepcopy
import warnings
import os
import sys
import shutil
from hera_sim.antpos import linear_array, hex_array
from hera_sim.vis import sim_red_data
from hera_sim.sigchain import gen_gains

from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler, Estimator
from qiskit.algorithms import optimizers as opt
from vqls_prototype.vqls import VQLS, VQLSLog
from vqls_prototype.qst_vqls import QST_VQLS
from vqls_prototype.hybrid_qst_vqls import Hybrid_QST_VQLS

from .. import quantum_redcal as om
from .. import io, abscal
from ..utils import split_pol, conj_pol, split_bl
from ..apply_cal import calibrate_in_place
from ..data import DATA_PATH
from ..datacontainer import DataContainer

np.random.seed(0)


class TestQuantumRedundantCalibrator(object):
    def test_init(self):
        # test a very small array
        pos = hex_array(3, split_core=False, outriggers=0)
        pos = {ant: pos[ant] for ant in range(4)}
        reds = om.get_reds(pos)
        rc = om.QuantumRedundantCalibrator(reds)
        with pytest.raises(ValueError):
            rc = om.QuantumRedundantCalibrator(reds, check_redundancy=True)

        # test disconnected redundant array
        pos = hex_array(5, split_core=False, outriggers=0)
        pos = {ant: pos[ant] for ant in pos if ant in [0, 1, 5, 6, 54, 55, 59, 60]}
        reds = om.get_reds(pos)
        try:
            rc = om.QuantumRedundantCalibrator(reds, check_redundancy=True)
        except ValueError:
            assert (
                False
            ), "This array is actually redundant, so check_redundancy should not raise a ValueError."

    def get_vqls_solver(self, nqubits):
        """_summary_"""
        ansatz = RealAmplitudes(nqubits, entanglement="full", reps=3)
        optimizer = opt.COBYLA(maxiter=5)
        return VQLS(Estimator(), ansatz, optimizer)

    def test_solver(self):
        nqbits = 2
        nants = 2**nqbits
        antpos = linear_array(nants)
        vqls = self.get_vqls_solver(nqbits)
        reds = om.get_reds(antpos, pols=["xx"], pol_mode="1pol")
        info = om.QuantumRedundantCalibrator(reds)
        gains, true_vis, d = sim_red_data(reds)
        w = {}
        w = dict([(k, 1.0) for k in d.keys()])

        def solver(data, wgts, **kwargs):
            np.testing.assert_equal(data["g_0_Jxx * g_1_Jxx_ * u_0_xx"], d[0, 1, "xx"])
            np.testing.assert_equal(data["g_1_Jxx * g_2_Jxx_ * u_0_xx"], d[1, 2, "xx"])
            np.testing.assert_equal(data["g_0_Jxx * g_2_Jxx_ * u_1_xx"], d[0, 2, "xx"])
            if len(wgts) == 0:
                return
            np.testing.assert_equal(wgts["g_0_Jxx * g_1_Jxx_ * u_0_xx"], w[0, 1, "xx"])
            np.testing.assert_equal(wgts["g_1_Jxx * g_2_Jxx_ * u_0_xx"], w[1, 2, "xx"])
            np.testing.assert_equal(wgts["g_0_Jxx * g_2_Jxx_ * u_1_xx"], w[0, 2, "xx"])
            return

        info._solver(solver, vqls, d)
        info._solver(solver, vqls, d, w)

    def test_firstcal(self):
        np.random.seed(21)
        antpos = hex_array(2, split_core=False, outriggers=0)
        reds = om.get_reds(antpos, pols=["xx"], pol_mode="1pol")
        vqls = self.get_vqls_solver(np.log(len(antpos)))

        rc = om.QuantumRedundantCalibrator(reds, vqls)
        freqs = np.linspace(1e8, 2e8, 1024)

        # test firstcal where the degeneracies of the phases and delays have already been removed so no abscal is necessary
        gains, true_vis, d = sim_red_data(reds, gain_scatter=0, shape=(2, len(freqs)))
        fc_delays = {
            ant: [[100e-9 * np.random.randn()]] for ant in gains.keys()
        }  # in s
        fc_delays = om.remove_degen_gains(reds, fc_delays)
        fc_offsets = {
            ant: [[0.49 * np.pi * (np.random.rand() > 0.90)]] for ant in gains.keys()
        }  # the .49 removes the possibly of phase wraps that need abscal
        fc_offsets = om.remove_degen_gains(reds, fc_offsets)
        fc_gains = {
            ant: np.reshape(
                np.exp(-2.0j * np.pi * freqs * delay - 1.0j * fc_offsets[ant]),
                (1, len(freqs)),
            )
            for ant, delay in fc_delays.items()
        }
        for ant1, ant2, pol in d.keys():
            d[(ant1, ant2, pol)] *= fc_gains[(ant1, split_pol(pol)[0])] * np.conj(
                fc_gains[(ant2, split_pol(pol)[1])]
            )
        for ant in gains.keys():
            gains[ant] *= fc_gains[ant]
        meta, sol_fc = rc.firstcal(d, freqs)
        np.testing.assert_array_almost_equal(
            np.linalg.norm([sol_fc[ant] - gains[ant] for ant in sol_fc.gains]),
            0,
            decimal=3,
        )

        # test firstcal with only phases (no delays)
        gains, true_vis, d = sim_red_data(reds, gain_scatter=0, shape=(2, len(freqs)))
        fc_delays = {ant: [[0 * np.random.randn()]] for ant in gains.keys()}  # in s
        fc_offsets = {
            ant: [[0.49 * np.pi * (np.random.rand() > 0.90)]] for ant in gains.keys()
        }  # the .49 removes the possibly of phase wraps that need abscal
        fc_offsets = om.remove_degen_gains(reds, fc_offsets)
        fc_gains = {
            ant: np.reshape(
                np.exp(-2.0j * np.pi * freqs * delay - 1.0j * fc_offsets[ant]),
                (1, len(freqs)),
            )
            for ant, delay in fc_delays.items()
        }
        for ant1, ant2, pol in d.keys():
            d[(ant1, ant2, pol)] *= fc_gains[(ant1, split_pol(pol)[0])] * np.conj(
                fc_gains[(ant2, split_pol(pol)[1])]
            )
        for ant in gains.keys():
            gains[ant] *= fc_gains[ant]
        meta, sol_fc = rc.firstcal(d, freqs)
        np.testing.assert_array_almost_equal(
            np.linalg.norm([sol_fc[ant] - gains[ant] for ant in sol_fc.gains]),
            0,
            decimal=10,
        )  # much higher precision
