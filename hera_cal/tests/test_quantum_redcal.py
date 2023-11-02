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
from qalcore.qiskit.vqls import VQLS

from .. import quantum_redcal as om
from .. import io, abscal
from ..utils import split_pol, conj_pol, split_bl
from ..apply_cal import calibrate_in_place
from ..data import DATA_PATH
from ..datacontainer import DataContainer

np.random.seed(0)


class TestQuantumRedundantCalibrator(object):

    def get_vqls_solver(self, nqubits):
        """_summary_"""
        nqubits = int(nqubits)
        ansatz = RealAmplitudes(nqubits, entanglement="full", reps=3)
        optimizer = opt.COBYLA(maxiter=5)
        return VQLS(Estimator(), ansatz, optimizer)

    def test_init(self):
        # test a very small array
        nqbits = 2
        nants = 2**nqbits
        pos = linear_array(nants)
        pos = {ant: pos[ant] for ant in range(4)}
        reds = om.get_reds(pos)

        vqls = self.get_vqls_solver(nqbits)
        rc = om.QuantumRedundantCalibrator(reds, solver=vqls)

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
        meta, sol_fc, solver_data = rc.firstcal(d, freqs)
        # np.testing.assert_array_almost_equal(
        #     np.linalg.norm([sol_fc[ant] - gains[ant] for ant in sol_fc.gains]),
        #     0,
        #     decimal=3,
        # )

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
        meta, sol_fc, solver_data_fc = rc.firstcal(d, freqs)
        # np.testing.assert_array_almost_equal(
        #     np.linalg.norm([sol_fc[ant] - gains[ant] for ant in sol_fc.gains]),
        #     0,
        #     decimal=10,
        # )  # much higher precision
