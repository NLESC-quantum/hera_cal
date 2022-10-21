
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

from qalcore.qiskit.vqls.numpy_unitary_matrices import UnitaryDecomposition 
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit import Aer
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService

NANTS = 2**2
NFREQ = 5
antpos = linear_array(NANTS)
circuits = QuantumCircuitsLinearArray(NANTS, NFREQ)

# print(isinstance(circuits, UnitaryDecomposition))
# exit()

# for _,p in antpos.items():
#     plt.plot(p[0],p[1],'o',color='black')
# plt.show()

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


# info.set_quantum_backend('simulator_statevector')
info.set_quantum_backend('ibmq_belem')
num_qubits = int(np.ceil(np.log2(NANTS)))
info.set_quantum_ansatz(RealAmplitudes(num_qubits, entanglement='full' , reps=3, insert_barriers=False))
info.set_quantum_optimizer(COBYLA(maxiter=50, disp=True))

info.set_quantum_circuits(circuits)

info.set_ibmq_credential(ibmq_token="494a8792f270fe0072c01aa9fe2235dc645248bf699bf3473de20a36a31fcb6e4e5369614581bc30d27c3b1c888ef9204130908ecd05d80e5d6a82a7791d3430",
                         hub="ibm-q-qal", 
                         group="escience", 
                         project="qradio")

info.set_ibmq_runtime_program_options(program_id='vqls-6W8lAdQ3gW', 
                                      shots=2500)

dly_sol, off_sol = info._firstcal_iteration(d, df=fqs[1] - fqs[0], f0=fqs[0], 
                                            medfilt=False, 
                                            mode='vqls_runtime')
                                            
sol_degen = info.remove_degen_gains(dly_sol, degen_gains=delays, mode='phase')
for i in range(NANTS):
    assert dly_sol[(i, 'Jxx')].dtype == np.float64
    assert dly_sol[(i, 'Jxx')].shape == (1, 1)
    assert np.allclose(np.round(sol_degen[(i, 'Jxx')] - delays[(i, 'Jxx')], 0), 0)