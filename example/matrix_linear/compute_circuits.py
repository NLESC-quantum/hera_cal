import numpy as np 
import matplotlib.pyplot as plt
from vqls_prototype.matrix_decomposition import SymmetricDecomposition, PauliDecomposition

files = ['4linear.mat','8linear.mat','16linear.mat',
         '32linear.mat', '64linear.mat','128linear.mat']

size = [4,8,16,32,64,128]

ncircuits = []
for fmat in files:
    matrix = np.loadtxt(fmat)
    print(matrix)
    PD = PauliDecomposition(matrix)
    ncircuits.append(len(PD._circuits))

plt.plot(size,ncircuits,'-o')
# plt.show()



depth = []
for fmat in files:
    matrix = np.loadtxt(fmat)
    SD = SymmetricDecomposition(matrix)
    depth.append(SD._circuits[0].decompose().depth())

plt.plot(size,depth,'-o')
plt.show()


# matrix = np.loadtxt(files[0])
# print(matrix)
# PD = PauliDecomposition(matrix)