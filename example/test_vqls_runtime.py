import numpy as np
from vqls_prototype import VQLS, VQLSLog
from vqls_prototype.hadamard_test import BatchHadammardTest
from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit.algorithms import optimizers as opt
import matplotlib.pyplot as plt 


from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeNairobi, FakeGuadalupeV2, FakeGuadalupe

from qiskit.primitives import BackendEstimator
from qiskit.providers.fake_provider import FakeNairobi, FakeGuadalupeV2

from zne import zne, ZNEStrategy
from zne.noise_amplification import LocalFoldingAmplifier
from zne.extrapolation import PolynomialExtrapolator

from qiskit.algorithms.gradients import FiniteDiffEstimatorGradient

N = 4
A = np.random.rand(N,N)
A = A + A.T 

# A = np.loadtxt('./matrix_linear/4linear.mat')
# A /= np.linalg.norm(A)
 
b = np.random.rand(N)
b /= np.linalg.norm(b)

num_qubits = int(np.log2(N))
ansatz = RealAmplitudes(num_qubits, entanglement='reverse_linear', 
                        reps=2, insert_barriers=False)

optimizer = opt.COBYLA(maxiter=250)
# optimizer = opt.SLSQP(maxiter=100, disp=True)

estimator = 'exact'
device = FakeGuadalupe()

# AER Estimator
if estimator == 'aer':    
    seed = 170
    coupling_map = device.configuration().coupling_map
    noise_model = NoiseModel.from_backend(device)
    estimator = AerEstimator(
        backend_options={
            "method": "density_matrix",
            "coupling_map": coupling_map,
            "noise_model": noise_model,
        },
        run_options={"seed": seed, "shots": 10000},
        transpile_options={"seed_transpiler": seed},
    )

# ZNE Estimator
if estimator == 'zne':
    ZNEEstimator = zne(BackendEstimator)
    estimator = ZNEEstimator(backend=device)

# device estimator
if estimator == 'device':
    estimator = BackendEstimator(device)

# noise free estimator
if estimator == 'exact':
    estimator = Estimator()


gradients = FiniteDiffEstimatorGradient(estimator, epsilon=0.01)

log = VQLSLog([], [])
vqls = VQLS(estimator, ansatz, optimizer, gradient=gradients, callback=log.update)
vqls_options = vqls._validate_solve_options({"matrix_decomposition":'contracted_pauli', 
                                             'shots':None})
vqls.solve(A,b,vqls_options)

plt.plot(log.values)
plt.show()