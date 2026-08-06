import numpy as np
from scipy.linalg import svd, sqrtm, kron, norm

#%% GENERAL ##############################

def compute_fidelity(rho,sigma):
    # Fidelity computation
    sqrt_sig = sqrtm(sigma)
    return np.real(np.trace(sqrtm(sqrt_sig @ rho @ sqrt_sig)) ** 2)

# alternate matrix square-root calculation
def sqrtm_numpy(A):
    # Eigen-decomposition
    w, V = np.linalg.eig(A)
    
    # Square root of eigenvalues
    W_sqrt = np.diag(np.sqrt(w))
    
    # Reconstruct sqrt(A)
    return V @ W_sqrt @ np.linalg.inv(V)

# should work on Hermitian or symmetric matrices
def sqrtm_sym(A):
    w, V = np.linalg.eigh(A)
    return V @ np.diag(np.sqrt(w)) @ V.T

################# CHATGPT SUGGESTIONS ###############

# def fidelity_numpy(rho, sigma):
#     # sqrt(rho)
#     sqrt_rho = sqrtm_numpy(rho)

#     # intermediate product
#     M = sqrt_rho @ sigma @ sqrt_rho

#     # sqrt of M
#     sqrt_M = sqrtm_numpy(M)

#     # fidelity = (Tr sqrt(M))^2
#     F = np.real((np.trace(sqrt_M))**2)

#     return F

# def sqrtm_numpy2(A):
#     # Eigen-decompose A
#     w, V = np.linalg.eigh(A)

#     # Numerical cleanup (remove tiny negatives from eigvals)
#     w = np.clip(w, 0, None)

#     # Build sqrt(A)
#     sqrt_w = np.diag(np.sqrt(w))
    # return V @ sqrt_w @ V.conj().T


#%% 2 QUBIT METHODS ######################

def beautify_2qubit(rho,verbose=False):

    # Eigendecomposition using SVD
    V, s_vals, Vh = svd(rho)
    
    # Compare reconstruction
    if verbose:
        lam = np.diag(s_vals)  # Singular values in diagonal matrix
        reconstruction_error = V @ lam @ Vh - rho
        print('Reconstruction error:',norm(reconstruction_error))

    # Find index of max singular value
    idx = np.argmax(s_vals)
    vecM = V[:, idx]

    # Reshape into 2 x 2 matrix
    M = vecM.reshape((2, 2))

    # Perform SVD on M
    u, s, vh = svd(M)

    # Compensation matrices
    Ua = np.conj(u.T)
    Ub = np.conj(vh)
            
    return Ua, Ub

### NUMPY IMPLEMENTATION OF KRONECKER PRODUCT
def kron_numpy_fast(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(
        A.shape[0]*B.shape[0],
        A.shape[1]*B.shape[1]
    )
###############################################


# @njit #Numba-compatible version?
def kron_numba(A, B):
    a_rows, a_cols = A.shape
    b_rows, b_cols = B.shape
    
    # Preallocate output (dtype must be explicit)
    out = np.zeros((a_rows * b_rows, a_cols * b_cols), dtype=A.dtype)

    for i in range(a_rows):
        for j in range(a_cols):
            for p in range(b_rows):
                for q in range(b_cols):
                    out[i*b_rows + p, j*b_cols + q] = A[i, j] * B[p, q]

    return out

########################################


def compensate_2qubit(rho,Ua,Ub,bell_choice=0):
    
    if len(Ua) != 2 or len(Ub) != 2:
        identity = np.identity(2)
        Ua, Ub = identity, identity
    
    # apply local unitary to Bob to match requested Bell state
    match bell_choice:
        case 1:
            Uc = np.array([[1,0],[0,-1]])
        case 2:
            Uc = np.array([[0,1],[1,0]])
        case 3:
            Uc = np.array([[0,-1],[1,0]])
        case _: #including case 0
            Uc = np.array([[1,0],[0,1]])
            
    Ub = Uc @ Ub
    
    Ucomp = kron(Ua, Ub)
        
    # Apply compensation
    rho1 = Ucomp @ rho @ np.conj(Ucomp.T)
    
    return rho1

def generate_werner_2qubit(bell_choice,w=1):
    #Generate ideal bell state matrix (w=1) or mixed Werner state (w<1)
    
    bell_state = np.zeros(4)

    sq2inv = 1/np.sqrt(2)

    match bell_choice:
        case 0:
            # print('Phi +')
            bell_state[0] = sq2inv
            bell_state[3] = sq2inv
        case 1:
            # print('Phi -')
            bell_state[0] = sq2inv
            bell_state[3] = -sq2inv
        case 2:
            # print('Psi +')
            bell_state[1] = sq2inv
            bell_state[2] = sq2inv
        case 3:
            # print('Phi -')
            bell_state[1] = sq2inv
            bell_state[2] = -sq2inv

    return w*np.outer(bell_state,bell_state) + (1-w)*np.identity(4)