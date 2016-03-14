import scipy.sparse
import numpy as np

# Make the result reproducible...
np.random.seed(1977)

def generate_random_sparse_array(nrows, ncols, numdense):
    """Generate a random sparse array with -1 or 1 in the non-zero portions"""
    i = np.random.randint(0, nrows-1, numdense)
    j = np.random.randint(0, ncols-1, numdense)
    data = np.random.random(numdense)
    data[data <= 0.5] = -1
    data[data > 0.5] = 1
    ij = np.vstack((i,j))
    return scipy.sparse.coo_matrix((data, ij), shape=(nrows, ncols))

#A = generate_random_sparse_array(4, 300000, 1000)
#B = generate_random_sparse_array(300000, 5, 1000)

#C = A * B

#print(C.todense())

n = 1024
N = n*n

print(N, 'x', N)
print(N*N*4/1000000000, 'GB')

A = generate_random_sparse_array(N, N, 2*N)
B = generate_random_sparse_array(N, 1, N)

C = A * B
