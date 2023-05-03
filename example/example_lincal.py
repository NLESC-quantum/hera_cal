
import numpy as np
from hera_sim.antpos import linear_array
from hera_sim.vis import sim_red_data
from hera_sim.sigchain import gen_gains

from hera_cal import redcal as om
from hera_cal.apply_cal import calibrate_in_place


from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit.algorithms import optimizers as opt
import matplotlib.pyplot as plt




solver = 'vqls_runtime'

NANTS = 4
NFREQ = 64
antpos = linear_array(NANTS)
# antpos = hex_array(NANTS)
# baseline_pairs = int(np.math.factorial(NANTS-1)/2/np.math.factorial(NANTS-3))
# circuits = QuantumCircuitsLinearArray(NANTS, NFREQ)

reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
info = om.RedundantCalibrator(reds)
fqs = np.linspace(.1, .2, NFREQ)


g, true_vis, d = sim_red_data(reds, shape=(1, NFREQ), gain_scatter=0)


delays = {k: np.random.randn() * 30 for k in g.keys()}  # in ns
delays = {k: np.array([[v]]) for k, v in delays.items()}

fc_gains = {k: np.exp(2j * np.pi * v * fqs) for k, v in delays.items()}
fc_gains = {i: v.reshape(1, NFREQ) for i, v in fc_gains.items()}

gains = {k: v * fc_gains[k] for k, v in g.items()}
gains = {k: v.astype(np.complex64) for k, v in gains.items()}

calibrate_in_place(d, gains, old_gains=g, gain_convention='multiply')
d = {k: v.astype(np.complex64) for k, v in d.items()}


dly_sol_ref, off_sol_ref = info._firstcal_iteration(d, df=fqs[1] - fqs[0], f0=fqs[0], 
                                            medfilt=False, 
                                            mode='solve')

sol_degen_ref = info.remove_degen_gains(dly_sol_ref, degen_gains=delays, mode='phase')

# for i in range(NANTS):
#     assert dly_sol_ref[(i, 'Jxx')].dtype == np.float64
#     assert dly_sol_ref[(i, 'Jxx')].shape == (1, 1)
#     assert np.allclose(np.round(sol_degen_ref[(i, 'Jxx')] - delays[(i, 'Jxx')], 0), 0)

if solver == 'vqls':
    
    num_qubits = int(np.ceil(np.log2(NANTS)))
    info.set_vqls_ansatz(RealAmplitudes(num_qubits, entanglement='full', reps=3, insert_barriers=False))
    info.set_vqls_optimizer(opt.COBYLA(maxiter=250, disp=True))


    dly_sol, off_sol = info._firstcal_iteration(d, df=fqs[1] - fqs[0], f0=fqs[0], 
                                                medfilt=False, 
                                                mode='vqls')

elif solver == 'vqls_runtime':

    
    num_qubits = int(np.ceil(np.log2(NANTS)))
    info.set_vqls_ansatz(RealAmplitudes(num_qubits, entanglement='full', reps=3, insert_barriers=False))
    info.set_vqls_optimizer(opt.COBYLA(maxiter=5, disp=True))
    info.set_ibmq_backend('simulator_statevector')


    dly_sol, off_sol = info._firstcal_iteration(d, df=fqs[1] - fqs[0], f0=fqs[0], 
                                                medfilt=False, 
                                                mode='vqls_runtime')

elif solver == 'qubols':

    info.set_qubo_num_qbits(11)
    info.set_qubo_num_reads(1000)
    dly_sol, off_sol = info._firstcal_iteration(d, df=fqs[1] - fqs[0], f0=fqs[0], 
                                                medfilt=False, 
                                                mode='qubols')

dly = [dly_sol[(i, 'Jxx')] for i in range(NANTS)]
dly_ref = np.array([dly_sol_ref[(i, 'Jxx')] for i in range(NANTS)])
# dly_ref /= np.linalg.norm(dly_ref)
# dly /= np.linalg.norm(dly)
print(dly)
print(dly_ref)
plt.scatter(dly_ref, dly)
plt.show()

sol_degen = info.remove_degen_gains(dly_sol, degen_gains=delays, mode='phase')

sol = [sol_degen[(i, 'Jxx')] for i in range(NANTS)]
ref = [delays[(i, 'Jxx')] for i in range(NANTS)]

plt.scatter(sol, ref)
plt.show()

for i in range(NANTS):
    assert dly_sol[(i, 'Jxx')].dtype == np.float64
    assert dly_sol[(i, 'Jxx')].shape == (1, 1)
    assert np.allclose(np.round(sol_degen[(i, 'Jxx')] - delays[(i, 'Jxx')], 0), 0)