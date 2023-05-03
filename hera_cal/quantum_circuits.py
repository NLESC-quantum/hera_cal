from qalcore.qiskit.vqls.matrix_decomposition import SymmetricDecomposition 
from qiskit import QuantumCircuit


def QuantumCircuitsLinearArray(nant, nfreq):


    def _init_four(nfreq):

        circuits, coefficients = [], []

        # -4 X^I
        circ = QuantumCircuit(2)
        circ.x(0)
        # circ.i(1)
        circuits.append(circ)
        coefficients.append(-4*nfreq)

        # -1 I^X
        circ = QuantumCircuit(2)
        # circ.i(0)
        circ.x(1)
        circuits.append(circ)
        coefficients.append(-nfreq)

        # 3 Swap
        circ = QuantumCircuit(2)
        circ.swap(0,1)
        circuits.append(circ)
        coefficients.append(3*nfreq)

        # 7 Swap XX
        circ = QuantumCircuit(2)
        circ.swap(0,1)
        circ.x(0)
        circ.x(1)
        circuits.append(circ)
        coefficients.append(7*nfreq)

        # -5 XX
        circ = QuantumCircuit(2)
        circ.x(0)
        circ.x(1)
        circuits.append(circ)
        coefficients.append(-5*nfreq)

        return circuits, coefficients

    init_fn = {4 : _init_four}

    if nant not in init_fn.keys():
        raise ValueError("Not implmented for %d antennas" %nant)

    circuits, coefficients = init_fn[nant](nfreq)
    return SymmetricDecomposition(circuits=circuits, coefficients=coefficients)

        







