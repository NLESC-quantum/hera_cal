
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

NANTS = 8
NFREQ = 64
antpos = linear_array(NANTS)
reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
info = om.RedundantCalibrator(reds)
fqs = np.linspace(.1, .2, NFREQ)
g, true_vis, d = sim_red_data(reds, shape=(1, NFREQ), gain_scatter=0)
delays = {k: np.random.randn() * 30 for k in g.keys()}  # in ns
fc_gains = {k: np.exp(2j * np.pi * v * fqs) for k, v in delays.items()}
delays = {k: np.array([[v]]) for k, v in delays.items()}
fc_gains = {i: v.reshape(1, NFREQ) for i, v in fc_gains.items()}
gains = {k: v * fc_gains[k] for k, v in g.items()}
gains = {k: v.astype(np.complex64) for k, v in gains.items()}
calibrate_in_place(d, gains, old_gains=g, gain_convention='multiply')
d = {k: v.astype(np.complex64) for k, v in d.items()}
dly_sol, off_sol = info._firstcal_iteration(d, df=fqs[1] - fqs[0], f0=fqs[0], medfilt=False, mode='quantum')
sol_degen = info.remove_degen_gains(dly_sol, degen_gains=delays, mode='phase')
for i in range(NANTS):
    assert dly_sol[(i, 'Jxx')].dtype == np.float64
    assert dly_sol[(i, 'Jxx')].shape == (1, 1)
    assert np.allclose(np.round(sol_degen[(i, 'Jxx')] - delays[(i, 'Jxx')], 0), 0)