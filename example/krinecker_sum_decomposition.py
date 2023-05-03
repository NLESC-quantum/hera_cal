import numpy as np

def gram_matrix(Xs):
  temp = np.vstack([np.ravel(X) for X in Xs])
  return np.dot(temp, temp.T)

def eig(X):
  vals, vecs = np.linalg.eig(X)
  idx = np.argsort(np.abs(vals))
  return vals[idx], vecs[...,idx]

def eig_both(X):
  # could call ctrevc to get both left and right at once
  return eig(X.T)[1], eig(X)[1]

def nkp_sum(As, Bs):
  """Nearest Kronecker product to a sum of Kronecker products.
  Given As = [A_1, ..., A_K] and Bs = [B_1, ..., B_K], solve
    min || \sum_i kron(A_i, B_i) - kron(Ahat, Bhat) ||_{Fro}^2
  where the minimization is over Ahat and Bhat, two N x N matrices.
  The size of the eigendecomposition computed in this implementation is K x K,
  and so the complexity scales like O(K^3 + K^2 N^2), where K is the length of
  the input lists.
  Args:
    As: list of N x N matrices
    Bs: list of N x N matrices
  Returns:
    Approximating factors (Ahat, Bhat)
  """

  GK = np.dot(gram_matrix(As), gram_matrix(Bs))
  lvecs, rvecs = eig_both(GK)
  Ahat = np.einsum('i,ijk->jk', lvecs[-1], As)
  Bhat = np.einsum('i,ijk->jk', rvecs[-1], Bs)
  return Ahat.reshape(As[0].shape), Bhat.reshape(Bs[0].shape)

def nkp(A, Bshape):
  """Nearest Kronecker product to a matrix.
  Given a matrix A and a shape, solves the problem
    min || A - kron(B, C) ||_{Fro}^2
  where the minimization is over B with (the specified shape) and C.
  The size of the SVD computed in this implementation is the size of the input
  argument A, and so to compare to nkp_sum if the output is two N x N matrices
  the complexity scales like O((N^2)^3) = O(N^6).
  Args:
    A: m x n matrix
    Bshape: pair of ints (a, b) where a divides m and b divides n
  Returns:
    Approximating factors (B, C)
  """

  blocks = map(lambda blockcol: np.split(blockcol, Bshape[0], 0),
                                np.split(A,        Bshape[1], 1))
  Atilde = np.vstack([block.ravel() for blockcol in blocks
                                    for block in blockcol])
  U, s, V = np.linalg.svd(Atilde)
  Cshape = A.shape[0] // Bshape[0], A.shape[1] // Bshape[1]
  idx = np.argmax(s)
  B = np.sqrt(s[idx]) * U[:,idx].reshape(Bshape).T
  C = np.sqrt(s[idx]) * V[idx,:].reshape(Cshape)
  return B, C