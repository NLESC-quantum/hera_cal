
from pyexpat import native_encoding
import numpy as np
from copy import deepcopy
import warnings
import os
import sys
import shutil

from hera_sim.antpos import linear_array, hex_array
from hera_sim.vis import sim_red_data
from hera_sim.sigchain import gen_gains

from hera_cal import redcal as om
from hera_cal import io, abscal
from hera_cal.utils import split_pol, conj_pol, split_bl
from hera_cal.apply_cal import calibrate_in_place
from hera_cal.data import DATA_PATH
from hera_cal.datacontainer import DataContainer
from hera_cal.quantum_circuits import QuantumCircuitsLinearArray


from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit import Aer
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService

NANTS = 18
antpos = linear_array(NANTS)
reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
info = om.RedundantCalibrator(reds)
gains, true_vis, d = sim_red_data(reds, gain_scatter=.0099999)
w = dict([(k, 1.) for k in d.keys()])
sol0 = dict([(k, np.ones_like(v)) for k, v in gains.items()])
sol0.update(info.compute_ubls(d, sol0))

def wgt_func1(abs2):
    return 1.

def wgt_func2(abs2):
    return np.where(abs2 > 0, 5 * np.tanh(abs2 / 5) / abs2, 1)

for wgt_func in [wgt_func1, wgt_func2]:
    meta, sol = info.omnical(d, sol0, conv_crit=1e-12, gain=.5, maxiter=500, check_after=30, check_every=6, wgt_func=wgt_func)
    for i in range(NANTS):
        assert sol[(i, 'Jxx')].shape == (10, 10)
    for bls in reds:
        ubl = sol[bls[0]]
        assert ubl.shape == (10, 10)
        for bl in bls:
            d_bl = d[bl]
            mdl = sol[(bl[0], 'Jxx')] * sol[(bl[1], 'Jxx')].conj() * ubl
            np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), decimal=10)
            np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, decimal=10)