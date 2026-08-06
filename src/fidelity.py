from scipy.special import psi
from pypolar import mueller
from sim import fidelity_between_stokes
import numpy as np


def overall_fidelity(rho, S0, S):
    """
    Calculate the overall fidelity of the entangled photons after transmission.
    The difference between the two stokes vectors represents a single-qubit unitary shift that was applied to the second qubit of the entangled pair.
    Args:
        rho: Density matrix 
        S0: Reference stokes parameters [S1, S2, S3]
        S: Observed stokes parameters [S1, S2, S3]
    Returns: Fidelity of the overall system
    """
    # Cast S0 and S to fully polarized stokes vectors (1x4 arrays where the first element is 1)
    S0 = np.array(S0) / np.linalg.norm(S0)
    S = np.array(S) / np.linalg.norm(S)
    if S0.size != 3 or S.size != 3:
        raise ValueError("S0 and S must be 3-element arrays")
    S0 = np.array([1, *S0])
    S = np.array([1, *S])

    # Convert Stokes vectors to qubit representation
    J0 = mueller.stokes_to_jones(S0)
    J = mueller.stokes_to_jones(S)
    # print("S0:", S0, "->", J0)
    # print("S:", S, "->", J)

    # Compute unitaries V0, V that rotate |0> to J0 and J
    a, b = J0
    c, d = J
    V0 = np.array([[a, -np.conjugate(b)],
                   [b,  np.conjugate(a)]])
    V  = np.array([[c, -np.conjugate(d)],
                   [d,  np.conjugate(c)]])

    # Compute a unitary U that rotates J0 to J
    U = V @ V0.conj().T
    np.testing.assert_allclose(U@J0, J)

    # Apply U to rho
    U = np.kron(np.eye(2), U)
    rho_prime = U @ rho @ U.conj().T

    # Calculate fidelity between rho_prime and the target state |psi^+> = (|01> + |10>)/sqrt(2)
    psi_plus = ((np.array([0, 1, 0, 0]) + np.array([0, 0, 1, 0])) / np.sqrt(2)).reshape(4,1)
    fidelity = np.real(psi_plus.conj().T @ rho_prime @ psi_plus) # Fidelity with pure state |psi^+>
    return fidelity[0][0]
    

psi_plus = ((np.array([0, 1, 0, 0]) + np.array([0, 0, 1, 0])) / np.sqrt(2)).reshape(4,1)
rho = psi_plus @ psi_plus.conj().T
S0 = [1,0,-1]
S = [1,1,0]
F = overall_fidelity(rho, S0, S)
print("Fidelity:", F)
print("Fidelity between S0 and S:", fidelity_between_stokes(np.array(S0), np.array(S)))

# IH(|01> + |10>) = |0-> + |1+>