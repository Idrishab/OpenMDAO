import numpy as np
from scipy.linalg import eig 


def arnoldi(A, b, k):
    """
    Computes the Arnoldi decomposition of matrix A using the starting vector b.

    Parameters:
    A : (n, n) array-like
        The input square matrix.
    b : (n,) array-like
        The starting vector for Arnoldi iteration. Can be randomly generated
    k : int
        The number of Arnoldi iterations (dimension of Krylov subspace).

    Returns:
    Q : (n, k+1) array
        Orthonormal basis of the Krylov subspace.
    H : (k+1, k) array
        Upper Hessenberg matrix.
    """
    n = A.shape[0]
    Q = np.zeros((n, k+1))
    H = np.zeros((k+1, k))

    # Normalize the starting vector
    Q[:, 0] = b / np.linalg.norm(b)

    for j in range(k):
        v = A @ Q[:, j]  # Compute A * q_j
        for i in range(j+1):  # Gram-Schmidt orthogonalization
            H[i, j] = np.dot(Q[:, i], v)
            v = v - H[i, j] * Q[:, i]

        H[j+1, j] = np.linalg.norm(v)
        if H[j+1, j] > 1e-12:  # Avoid division by zero
            Q[:, j+1] = v / H[j+1, j]
        else:
            break  # If v is numerically zero, stop iteration

    return Q, H

def extract_tridiag(H):
    """
        This function keeps the tridiagomal entries of matrix H 
        and set all other entries to zeros.
    """
    m, n = H.shape
    assert m >= n, f"extract_tridiag requires m >= n, but got shape {H.shape}"
    
    td_H = np.zeros_like(H)
    
    for i in range(m):
        # main diagonal
        if i < n:
            td_H[i, i] = H[i, i]

        # superdiagonal
        if i + 1 < n:
            td_H[i, i+1] = H[i, i+1]

        # subdiagonal
        if i - 1 >= 0:
            td_H[i, i-1] = H[i, i-1]
            
    return td_H
    
def compute_eigenvalues_and_eigenvectors(A, b, k):
    """
    Computes the approximate eigenvalues and eigenvectors of A using Arnoldi iteration.

    Parameters:
    A : (n, n) array-like
        The input square matrix.
    b : (n,) array-like
        The starting vector for Arnoldi iteration.
    k : int
        The number of Arnoldi iterations.

    Returns:
    eigvals : array
        Approximate eigenvalues of A.
    eigvecs : array
        Approximate eigenvectors of A.
    """
    Q, H = arnoldi(A, b, k)
    
    # Make the tridiagonal form of H symmetric
    td_H = extract_tridiag(H)
    H = 0.5 *(td_H[:-1, :] + np.transpose(td_H[:-1, :])) # Ignore last row for eigenvalues
    eigvals, eigvecs_H = eig(H)
    eigvals = np.real(eigvals)                           # extract the real parts
    
    # Compute eigenvalues and eigenvectors of the Hessenberg matrix H
    # eigvals, eigvecs_H = eig(H[:-1, :])  # Ignore last row for eigenvalues
    # eigvals = np.real(eigvals)           # extract the real parts

    # Transform eigenvectors of H to approximate eigenvectors of A
    eigvecs = Q[:, :-1] @ eigvecs_H
    
    return np.abs(eigvals), eigvecs

def sort_eigs(unsorted_eigvals, unsorted_eigvecs):
    """
    Sorts eigenvalues and eigenvectors in descending order of eigenvalues.

    Parameters:
    unsorted_eigvals : (k,) array-like
        The array of eigenvalues.
    unsorted_eigvecs : (n, k) array-like
        The corresponding matrix of eigenvectors.

    Returns:
    sorted_eigvals : (k,) array
        Eigenvalues sorted in descending order.
    sorted_eigvecs : (n, k) array
        Corresponding eigenvectors sorted accordingly.
    """
    
    # Get sorting indices (descending order)
    sorted_indices = np.argsort(np.abs(unsorted_eigvals))[::-1]
    
    # Sort eigenvalues and eigenvectors
    sorted_eigvals = unsorted_eigvals[sorted_indices]
    sorted_eigvecs = unsorted_eigvecs[:, sorted_indices]
    
    return sorted_eigvals, sorted_eigvecs
    

def extract_eigpairs(A, b, iter, m):
    """
    provides m largest eigenvalues and eigenvectors of the matrix (or linear operator) A
    
    Parameters:
    A : (n, n) array-like
        The input square matrix. It can be a matrix or linear operator
    b : (n,) array-like
        The starting vector for Arnoldi iteration.
    iter : int
        The number of Arnoldi iterations.
        
    Returns:
    m_eigvals : (m, ) array
        m largest approximate eigenvalues of A.
    m_eigvecs : (n, m) array 
        Corresponding m approximate eigenvectors of A.
    """
    k = iter
    eigvals, eigvecs = compute_eigenvalues_and_eigenvectors(A, b, k)
    sorted_vals, sorted_vecs = sort_eigs(eigvals, eigvecs)
    m_eigvals = np.copy(sorted_vals[0:m])
    m_eigvecs = np.copy(sorted_vecs[:, 0:m])

    return m_eigvals, m_eigvecs

def modified_arnoldi(A, b, max_iter, min_cond):
    """
    Computes the eigenpairs of A through Arnoldi method using the starting vector b.
    If the min_cond requirement is not satisfied, max_iter prevails

    Parameters:
    A : (n, n) array-like
        The input square matrix.
    b : (n,) array-like
        The starting vector for Arnoldi iteration. Can be randomly generated
    max_iter : int
        The maximum number of Arnoldi iterations 
        
    Returns:
    m_eigvals : (m,) array
        absolute values of the eigenvalues sorted in descending order.
    m_eigvecs : (n, m) array
        matrix of approximate eigenvectors.
    m : integer
        dimension of krylov subspace        
    """
    k = max_iter
    n = A.shape[0]
    assert max_iter <= n, "Error! max_iter for the arnoldi method must be less than or equal to the number of rows in A"
    assert isinstance(max_iter, int), f"max_iter must be an integer, but you have provided {max_iter}"
    Q = np.zeros((n, k+1))
    H = np.zeros((k+1, k))

    # Normalize the starting vector
    Q[:, 0] = b / np.linalg.norm(b)
    m = 0
    # cond_num = 1

    for j in range(k):
        v = A @ Q[:, j]  # Compute A * q_j
        for i in range(j+1):  # Gram-Schmidt orthogonalization
            H[i, j] = np.dot(Q[:, i], v)
            v = v - H[i, j] * Q[:, i]

        H[j+1, j] = np.linalg.norm(v)
        if H[j+1, j] > 1e-12:  # Avoid division by zero
            Q[:, j+1] = v / H[j+1, j]
        else:
            break  # If v is numerically zero, stop iteration
        
        m = j + 1
        
        # modified part of the method
        if j > 0:
            # Compute eigenvalues and eigenvectors of the Hessenberg matrix H
            eigvals_unsorted, eigvecs_H = eig(H[:m, :m])  
            eigvals_unsorted = np.real(eigvals_unsorted)           # extract the real parts

            # Transform eigenvectors of H to approximate eigenvectors of A
            eigvecs_unsorted = Q[:, :m] @ eigvecs_H
            
            abs_eigvals = np.abs(eigvals_unsorted)
            min_eig = np.min(abs_eigvals)
            max_eig = np.max(abs_eigvals)           
            
            if min_eig <= 1e-8:
                print(f"\nSmallest eigenvalue is too small (less than 1e-8): val = {min_eig}")
                m -= 1
                print(f"Returning only {m} eigenpairs!!!\n")
                break
            
            cond_num = max_eig / min_eig
            print(f"m = {m}, condition number = {cond_num}")
            if cond_num >= min_cond:
                print(f"Terminating Arnoldi iteration, the condition number is {cond_num}!\n")
                break
            
    eigvals_unsorted, eigvecs_H = eig(H[:m, :m])  
    eigvals_unsorted = np.real(eigvals_unsorted)           # extract the real parts
    eigvecs_unsorted = Q[:, :m] @ eigvecs_H
    
    # sort the eigenpairs
    eigvals, eigvecs = sort_eigs(np.abs(eigvals_unsorted), eigvecs_unsorted)
    m_eigvals = np.abs(eigvals[:m])
    m_eigvecs = eigvecs[:,:m]

    return m_eigvals, m_eigvecs, m
