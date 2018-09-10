# Modified gram schmidt as described here:
# https://www.inf.ethz.ch/personal/gander/papers/qrneu.pdf
def gram_schmidt(A):
    n = len(A[0])
    m = len(A)
    r = np.zeros([n,n])
    q = np.zeros([m,n])
    matrix = np.copy(A)
    for k in range(n):
        s = 0
        for j in range(m):
            s = s + matrix[j][k] ** 2
        r[k,k] = s ** 0.5
        for j in range(m):
            q[j][k] = matrix[j][k] / r[k][k]
        for i in range(k+1, n):
            aux = 0
            for j in range(m):
                aux+= matrix[j][i] * q[j][k]
            r[k][i] = aux
            for j in range(m):
                matrix[j][i] = matrix[j][i] - r[k][i] * q[j][k]
            
    return q, r

# Given a square matrix and an eigenvalue, it uses inverse iteration to calculate the corresponding eigenvector
def inverse_iteration(matrix, eigenvalue):
     n = len(matrix)
     I = np.identity(n)
     eigenvector = np.random.rand(n, 1)
     inv = np.linalg.inv(matrix - eigenvalue * I)
     for i in range(100):
         aux = np.dot(inv, eigenvector)
         aux = aux/np.linalg.norm(aux)
         eigenvector = aux

     return eigenvector

# Given a diagonalizable matrix, it returns a list with its eigenvalues in descending absolute value and a matrix with the corresponing eigenvectors
def eigen_values_and_vectors(A):
    n = len(A)
    Q, R = householder_QR(A)
    matrix = np.copy(A)
    for k in range(10000):
        Q, R = householder_QR(matrix)
        matrix = R.dot(Q)

    eigen_vectors = np.zeros([n, n])
    eigen_values = sorted(np.diagonal(matrix), key=abs)[::-1]
    for i in range(len(eigen_values)):
       eigen_vector = inverse_iteration(A, eigen_values[i])
       for j in range(n):
           eigen_vectors[j][i] = eigen_vector[j]
    return eigen_values, eigen_vectors



# Householder QR method as described here:
# https://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf
def householder_QR(A):
    m = len(A)
    n = len(A[0])
    Q = np.identity(m)
    R = np.copy(A)
    for j in range(n):
        normx = np.linalg.norm(R[j:m,j])
        s = - np.sign(R[j,j])
        u1 = R[j,j] - s * normx
        w = R[j:m, j].reshape((-1,1)) / u1
        w[0] = 1
        tau = -s * u1 / normx
        R[j:m, :] = R[j:m, :] - (tau * w) * np.dot(w.reshape((1,-1)),R[j:m,:])
        Q[:, j:n] = Q[:,j:n] - (Q[:,j:m].dot(w)).dot(tau*w.transpose())

    return Q, R

# Schur decomposition as described here:
# https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter4.pdf
def schur_decomposition(A):
    """docstring for schurDecomposition"""
    matrix = np.copy(A)
    U = np.identity(len(A))
    for k in range(1000):
        Q, R = householder_QR(matrix)
        matrix = R.dot(Q)
        U = U.dot(Q)
    return U, matrix