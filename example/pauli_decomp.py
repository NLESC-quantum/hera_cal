import numpy as np
from linsolve.pauli_decomposition import PauliDecomposition 
from hera_sim.antpos import linear_array, hex_array, HexArray
import matplotlib.pyplot as plt 
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator, Pauli


# import sys
# np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(linewidth=1000)


from qiskit.opflow import (
    Z,
    X,
    Y,
    I,
    Plus,
    Minus,
    Zero, One,
    CX, S, H, T, CZ, Swap,
    TensoredOp
)

# hex_array = HexArray(split_core=True, outriggers=0)

# pos = hex_array(6)
# plt.scatter([v[0] for k,v in pos.items()], [v[1]for k,v in pos.items()])
# plt.show()


Abl = np.loadtxt('8linear_baseline.mat')

A = np.loadtxt('8linear.mat')

A0 = Abl[:,:21] @ Abl[:,:21].T
PD = PauliDecomposition(A)



A5 = Abl[:,-1:]@Abl[:,-1:].T 
A4 = Abl[:,52:55] @ Abl[:,52:55].T
A3 = Abl[:,46:52]@ Abl[:,46:52].T
A2 = Abl[:,36:46]@ Abl[:,36:46].T
A1 = Abl[:,21:36]@ Abl[:,21:36].T
A0 = Abl[:,:21]@ Abl[:,:21].T