import numpy as np
from scipy.interpolate import interp1d
from src.functions import *

def do_cld_1side(oz: np.ndarray, rho: np.ndarray, estep: int, minrho: float):
    """
    Obtain the representative energies (repE, E in Campo2005) and
    the integral of the hybridization function (repT) for each discretization
    interval, for either positive or negative energy side.

    """
    
    max_rho = np.max(rho) if rho.size > 0 else 0
    if max_rho == 0:
        return np.array([]), np.array([])    
    threshold = minrho * max_rho
    valid_indices = np.where(rho >= threshold)[0]
    if valid_indices.size == 0:
        return np.array([]), np.array([])
        
    ids0 = valid_indices[-1] - estep
    start_idx = max(0, ids0) 
    indices_sequence = np.arange(start_idx, -estep, -estep) # Goes down to 0 or less
    last_idx = len(oz) - 1
    indices_sequence = np.arange(start_idx, -1, -estep) 
    ids_0based = np.unique(np.concatenate(([last_idx], indices_sequence)))
    ids_0based = np.sort(ids_0based)[::-1]
    
    num_intervals = len(ids_0based) - 1
    repT = np.zeros(num_intervals)
    repE_num = np.zeros(num_intervals) # Numerator of repE before division by repT
    
    for itx in range(num_intervals):
        start = ids_0based[itx + 1]
        end = ids_0based[itx] + 1 # +1 to make it exclusive end for slicing
        
        ozp = oz[start:end]
        rhop = rho[start:end][..., np.newaxis]
        
        if ozp.size < 2:
            # Should not happen with correct ids_0based logic, but a safeguard
            repT[itx] = 0.0
            repE_num[itx] = 0.0
            continue
            
        # Numerical Integration for repT (Trapezoidal Rule)
        repT[itx] = np.sum((rhop[1:] + rhop[:-1]) * (ozp[1:] - ozp[:-1]))/ 2.0
        # Numerical Integration for repE (Custom formula from Campo2005)
        
        term1 = rhop[-1] - rhop[0]
        
        numerator = ozp[1:] * rhop[:-1] - ozp[:-1] * rhop[1:]
        
        denominator = ozp[1:] - ozp[:-1]
        
        log_term = np.log(np.abs(ozp[1:] / ozp[:-1]))
        
        sum_term = np.sum((numerator / denominator) * log_term)
        
        repE_num[itx] = term1 + sum_term

    repE = np.divide(repT, repE_num, out=np.zeros_like(repT), where=repE_num != 0)

    return repE.reshape(-1, 1), repT.reshape(-1, 1)

def do_cld(ozin: np.ndarray, RhoV2in: np.ndarray, Lambda: float, N: int, **kwargs):
    """
    Performs the Campo-Oliveira logarithmic discretization (CLD) and maps
    the resulting star-geometry Hamiltonian onto the Wilson chain.
    
    Parameters:
        ozin (np.ndarray): Original frequency grid.
        RhoV2in (np.ndarray): Original spectral function $\rho(\omega) = (-1/\pi) \text{imag}[\Delta(\omega)]$.
        Lambda (float): Discretization parameter.
        N (int): Wilson chain length.
        **kwargs: Optional parameters ('estep', 'emax', 'emin', 'minrho', 'fftol').: they are for logarithmic frequency grid "oz". 
        emax, emin : maximum/minimum frequency value of grid "oz"
        minrho : cutoff to set effective band edges, when RhoV2in has long tails.         
    Returns:
        ff (np.ndarray): Hopping amplitudes.
        gg (np.ndarray): On-site energies.
    """
    
    # Default parameters
    estep = kwargs.get('estep', 10)
    
    # MATLAB: emax = max(abs(ozin))
    emax = kwargs.get('emax', np.max(np.abs(ozin)) if ozin.size > 0 else 1.0)
    
    emin = kwargs.get('emin', 1e-20)
    minrho = kwargs.get('minrho', 0.01)
    fftol = kwargs.get('fftol', 0.01)

    # Input parsing and validation
    ozin = ozin.reshape(-1, 1) # Force column vector
    RhoV2in = RhoV2in.reshape(-1, 1) # Force column vector
    
    if ozin.size == 0 or RhoV2in.size == 0 or ozin.shape != RhoV2in.shape:
        raise ValueError('ERR: Check frequency and hybridization inputs. They must be non-empty and have the same number of elements.')
    
    
    max_x_val = np.ceil(np.log(emax) / np.log(Lambda) * estep)
    min_x_val = np.floor(np.log(emin) / np.log(Lambda) * estep)
    xs_steps = np.arange(max_x_val, min_x_val - 1, -1)
    xs = np.flipud(xs_steps / estep).reshape(-1, 1)
    oz = Lambda**xs
    
    # Discretization for positive (rho1) and negative (rho2) frequencies
    
    # Interpolation for positive frequencies: $\rho(\omega)$ at $\omega = +oz$
    interp_func = interp1d(ozin.flatten(), RhoV2in.flatten(), kind='linear', fill_value='extrapolate')
    
    rho1 = interp_func(oz.flatten())
    
    rho1[rho1 < 0] = 0 # To ensure non-negativity
    
    repE1, repT1 = do_cld_1side(oz, rho1, estep, minrho)
    # Interpolation for negative frequencies: $\rho(\omega)$ at $\omega = -oz$
    rho2 = interp_func((-oz).flatten())
    rho2[rho2 < 0] = 0 # To ensure non-negativity
    
    repE2, repT2 = do_cld_1side(oz, rho2, estep, minrho)
    
    # Combine results
    Xis = np.concatenate((repE1, -repE2)).flatten()
    Gammas = np.concatenate((np.sqrt(repT1), np.sqrt(repT2))).flatten()
    
    # Determine the actual chain length N2
    # The number of discretization intervals is the size of Xis (or Gammas)
    num_intervals = Xis.size
    
    if num_intervals < N:
        print(f'WRN: Number of discretization intervals (= {num_intervals}) is smaller than the chain length N (= {N})')
        N2 = num_intervals
    else:
        N2 = N
    
    if N2 == 0:
        return np.array([]), np.array([])
        
    # Star-geometry Hamiltonian matrix (H), H = [0 Gammas'; Gammas, diag(Xis)];
    H_size = num_intervals + 1
    H = np.zeros((H_size, H_size))
    
    # Top row and left column
    H[0, 1:] = Gammas
    H[1:, 0] = Gammas
    
    # Diagonal (from Xis)
    np.fill_diagonal(H[1:, 1:], Xis)
    
    # Lanczos Tridiagonalization
    # H (matrix) is in the $v_0, |\epsilon_n\rangle$ basis, where $v_0$ is the dot state.
    
    # U: Transformation matrix from the original basis to the Wilson basis
    # U is a matrix of Wilson basis vectors (columns) in the original basis
    U = np.zeros((H_size, N2 + 1))
    
    # The first vector is the dot state in the star basis, normalized to 1.
    U[0, 0] = 1.0 # The starting vector v0 for Lanczos is the dot state: |0> = |dot>
    
    ff = np.zeros(N2)
    gg = np.zeros(N2)
    
    for itN in range(N2): 
        # v = H*U(:,itN);
        # In the loop, U(:,itN) is the previous Wilson chain vector |n-1>
        v = H @ U[:, itN]
        
        # Orthgonalize to U(:,1:itN) = {v0, ..., v_{n-1}}, Python: v = v - U[:, :itN+1] @ (U[:, :itN+1].T @ v)
        # We need to orthogonalize v against the vectors U[:, 0] to U[:, itN] (inclusive)
        U_current = U[:, :itN + 1] # Vectors v0 to v_itN (where v_itN is |itN-1>)
        
        # Standard Lanczos: orthogonalize against the *two* previous vectors (U[:, itN] and U[:, itN-1])
        # The full orthogonalization here (Gram-Schmidt) is for robustness/numerical stability.
         
        for _ in range(2):# Double orthogonalization for stability
            v = v - U_current @ (U_current.T @ v)

        ff[itN] = np.linalg.norm(v)

        # Truncation check
        if ff[itN] < (Lambda**(-itN / 2.0) * fftol):
            # break; in MATLAB
            N_final = itN 
            break
        
        # U[:, itN+1] is the (itN+1)-th column, which is the vector |itN>
        U[:, itN + 1] = v / ff[itN]
        
        # gg[itN] is the on-site energy $\epsilon_{itN}$ of the $\text{itN}$-th site.
        gg[itN] = U[:, itN + 1].T @ H @ U[:, itN + 1]
    
    else:
        # Loop completed without break
        N_final = N2

    # Truncate ff and gg to the actual number of sites (N_final)
    ff = ff[:N_final].reshape(-1, 1)
    gg = gg[:N_final].reshape(-1, 1)

    return ff, gg




    
# NRG Result Structure to store data 
class NRGResult:
    """Class to store the results of the NRG iterative diagonalization."""
    def __init__(self):
        self.Lambda: float = 0.0
        self.EScale: np.ndarray = np.array([])
        self.EK: List[np.ndarray] = []  # Kept energy eigenvalues
        self.AK: List[np.ndarray] = []  # Kept isometries (rank-3)
        self.ED: List[np.ndarray] = []  # Discarded energy eigenvalues
        self.AD: List[np.ndarray] = []  # Discarded isometries (rank-3)
        self.E0: np.ndarray = np.array([]) # Ground-state energy at each iteration
    def print(self, itN):
        print(self.EK[itN], self.AK[itN], self.ED[itN], self.AD[itN], self.E0[itN])


# Iterative diagonalization for nrg 
def nrg_iter_diag(H0: np.ndarray, A0: np.ndarray, Lambda: float, ff: np.ndarray, F: np.ndarray, gg: np.ndarray, NF: np.ndarray, Z: np.ndarray, Nkeep: int) -> NRGResult:
    """
    Iterative diagonalization of the numerical renormalization group (NRG) method.

    Parameters:
        H0 (np.ndarray): Rank-2 Impurity Hamiltonian.
        A0 (np.ndarray): Rank-3 Impurity isometry (left-right-bottom).
        Lambda (float): Logarithmic discretization parameter.
        ff (np.ndarray): Hopping amplitudes in the Wilson chain.
        F (np.ndarray): Fermion annihilation operator.
        gg (np.ndarray): On-site energies in the Wilson chain.
        NF (np.ndarray): Particle number operator.
        Z (np.ndarray): Fermion anti-commutation sign operator.
        Nkeep (int): Number of states to be kept.

    Returns:
        Inrg (NRGResult): NRG result structure.
    """
    Nfac = 0.1  # up to 10% more states can be kept

    # Error checking
    if H0.ndim != 2:
        raise ValueError("ERR: 'H0' should be of rank 2.")
    if A0.ndim != 3:
        raise ValueError("ERR: 'A0' should be of rank 3.")

    Inrg = NRGResult()
    Inrg.Lambda = Lambda
    N = len(ff) + 1  # number of iterations (from 0 to N-1)

    # Rescaling factor (to divide the energy values):
    # EScale(0) = 1 (no rescaling for the impurity).
    # EScale(N-1) rescales ff(end) to be 1.
    # MATLAB: Inrg.EScale = [1, (Lambda.^(((N-2):-1:0)/2))*ff(end)];
    
    if N >= 2:
        # Array of exponents: (N-2), (N-3), ..., 0
        exponents = np.arange(N - 2, -1, -1)
        scale_factors = (Lambda**(exponents / 2.0)) * ff[-1]
        Inrg.EScale = np.concatenate(([1.0], scale_factors))
    else:
        Inrg.EScale = np.array([1.0])
    
    # Initialize results lists
    Inrg.EK = [np.array([])] * N
    Inrg.AK = [np.array([])] * N
    Inrg.ED = [np.array([])] * N
    Inrg.AD = [np.array([])] * N
    Inrg.E0 = np.zeros(N)

    tobj = tic2()
    disptime('NRG: start')

    Hprev = H0

    # Iteration loop: itN runs from 0 to N-1 (MATLAB 1 to N)
    for itN in range(N):
        
        # The site index added is itN - 1 (for itN=1, site 0)
        
        if itN == 0: # only include the legs of A0 (Impurity site)
            
            # Inrg.AK{itN} = A0; # don't rotate the basis
            Inrg.AK[itN] = A0
            
            eigvals, _ = np.linalg.eig(H0)
            # Ensure eigenvalues are real if H0 is Hermitian (which it should be)
            Inrg.EK[itN] = np.sort(np.real(eigvals))
            
            Inrg.AD[itN] = np.zeros((A0.shape[0], 0, A0.shape[2]))
            Inrg.ED[itN] = np.zeros(0)
            
            
           
        else: # including additional sites (itN >= 1, for site itN-1)
            
            # Construct the enlarged Hamiltonian Hnow

            Anow = get_identity(Inrg.AK[itN-1], 1, Z, 1, [0, 2, 1]) # Corrected to use physical legs
            Hnow = update_left(Hprev, 2, Anow, None, None, Anow)
            # rescaling
            Hnow = Hnow * (Inrg.EScale[itN-1] / Inrg.EScale[itN])

            # Fermionic operators for the hopping term: F and Fprev
            Fnow = np.transpose(np.conjugate(F), (1, 0, 2)) # F^dagger at site n-1
            
            Fnow = contract(Fnow, 1, Z, 0, permute_order=[0, 2, 1]) 
            
            Hhop = update_left(Fprev, 3, Anow, Fnow, 3, Anow)
            Hhop = (ff[itN-1] / Inrg.EScale[itN]) * Hhop
            # Hhop = (ff[itN-1]) * Hhop
            Hhop = Hhop + np.conjugate(np.transpose(Hhop, (1, 0))) # H + H^dagger
            Hon = update_left(None, None, Anow, NF, 2, Anow)             
            Hon = (gg[itN-1] / Inrg.EScale[itN]) * Hon
            Hnow = Hnow + Hhop + Hon
            

            H_herm = (Hnow + np.conjugate(np.transpose(Hnow, (1, 0)))) / 2.0
            eigvals, V = np.linalg.eigh(H_herm)
            D = eigvals # Sorted eigenvalues
            Inrg.E0[itN] = D[0]
            # print(D)
            D = D - Inrg.E0[itN]
            # Truncation logic
            D_size = len(D)
            if itN < N - 1: # Not the last iteration
                if D_size > Nkeep:
                    start_idx = Nkeep
                    end_idx = min(D_size, int(np.ceil(Nkeep * (1 + Nfac))))
                    
                    ids_to_check = np.arange(start_idx, end_idx)
                    
                    if len(ids_to_check) > 1:
                        # [~,maxid] = max(diff(D(ids)));
                        diffs = np.diff(D[ids_to_check])
                        # maxid is the index within diffs where the max gap is found
                        maxid_relative = np.argmax(diffs)
                        
                        Ntr = ids_to_check[maxid_relative] + 1 
                    else:
                        # Not enough states to check for a gap, keep Nkeep
                        Ntr = Nkeep
                else:
                    # Keep all states if less than Nkeep
                    Ntr = D_size
            else:
                # Discard all the states at the last iteration
                Ntr = 0

            # Update NRG results
            Inrg.EK[itN] = D[:Ntr]
            
            Inrg.ED[itN] = D[Ntr:]
            
            V_kept = V[:, :Ntr]
            Inrg.AK[itN] = contract(Anow, 1, V_kept, 0, permute_order=[0, 2, 1])

            V_disc = V[:, Ntr:]
            Inrg.AD[itN] = contract(Anow, 1, V_disc, 0, permute_order=[0, 2, 1])

            Hprev = np.diag(Inrg.EK[itN])
            
        if itN < N - 1:
            
            Fprev = update_left(None, None, Inrg.AK[itN], F, 3, Inrg.AK[itN])
            
        # Information on truncation (logging)
        if Inrg.EK[itN].size == 0:
            Etr1 = 0.0
        else:
            Etr1 = Inrg.EK[itN][-1]
        Ntr1 = Inrg.EK[itN].size
        
        if Inrg.ED[itN].size == 0:
            Etr2 = Etr1
        else:
            Etr2 = Inrg.ED[itN][-1]
        Ntr2 = Ntr1 + Inrg.ED[itN].size
        
        disptime(f"#{itN:02d}/{N-1:02d} : NK={Ntr1}/{Ntr2}, EK={Etr1:.4g}/{Etr2:.4g}")
        
    chkmem()
    toc2(tobj, '-v')

    return Inrg


def get_td_conv(Inrg, Sz: np.ndarray, beta0: float, S_A0L: np.ndarray = None):
    """
    Compute thermodynamic properties T*chi and entropy using the conventional NRG method.

    Parameters:
        Inrg (Any): NRG result object (instance of NRGResult).
        Sz (np.ndarray): Spin-z operator for the local physical space.
        beta0 (float): Prefactor to define the effective temperature.
        S_A0L (np.ndarray, optional): Spin-z operator acting on the left leg (first leg) of A0.

    Returns:
        T (np.ndarray): Temperature values.
        Tchi (np.ndarray): Temperature * static spin susceptibility (after even-odd averaging).
        Sent (np.ndarray): Entropy (after even-odd averaging).
    """
    tobj = tic2()
    disptime('Compute thermodynamic properties with the conventional NRG method.')

    # --- Input Sanity Checks  ---
    if Sz.shape[0] != Sz.shape[1] or Sz.ndim > 2:
        raise ValueError("ERR: 'Sz' should be a square matrix (rank 2).")
    
    A0_left_dim = Inrg.AK[0].shape[0] if Inrg.AK and Inrg.AK[0] is not None else 1
    
    if A0_left_dim > 1 and S_A0L is None:
        raise ValueError("ERR: If the first leg (= left leg) of A0 is non-singleton, 'S_A0L' should be set.")
    
    if S_A0L is not None:
        if S_A0L.ndim > 2 or S_A0L.shape[0] != A0_left_dim:
            raise ValueError("ERR: 'S_A0L' should be a square matrix whose dimensions match with the first leg (= left leg) of A0.")

    # Check Sz dimension consistency with isometries (AK/AD)
    sz_phys_dim = Sz.shape[1]
    for itN in range(len(Inrg.AK)):
        # Check against AK (kept isometries)
        if Inrg.AK[itN].size > 0 and Inrg.AK[itN].shape[2] != sz_phys_dim:
             raise ValueError("ERR: Dimensions of 'Sz' are not consistent with the third-leg dimension of AK isometries.")
        # Check against AD (discarded isometries)
        if Inrg.AD[itN].size > 0 and Inrg.AD[itN].shape[2] != sz_phys_dim:
             raise ValueError("ERR: Dimensions of 'Sz' are not consistent with the third-leg dimension of AD isometries.")
    
    # --- Initialization ---
    N = len(Inrg.EScale) # total number of iterations
    
    # Temperature: T = [Inrg.EScale(2)*sqrt(Inrg.Lambda), Inrg.EScale(2:end)]/beta0;
    EScale_for_T = Inrg.EScale[1:] # EScale(2:end)
    T = np.concatenate(([Inrg.EScale[1] * np.sqrt(Inrg.Lambda)], EScale_for_T)) / beta0
    
    Tchi0 = np.zeros(N)
    Sent0 = np.zeros(N)
    SzKprev = np.array([]) # SzK from previous iteration

    # --- Main Iteration (Conventional NRG) ---
    for itN in range(N):
        # 0-based index 'itN' corresponds to 1-based MATLAB index 'itN+1'
        
        AK = Inrg.AK[itN]
        AD = Inrg.AD[itN]
        
        # --- Operator Construction (SzK and SzD) ---
        if itN == 0:
            # Sz operator for the bottom-leg space (site 0)
            SzK = update_left(None, None, AK, Sz, 2, AK)
            SzD = update_left(None, None, AD, Sz, 2, AD)

            # Left-leg space (Impurity part):
            if A0_left_dim > 1 and S_A0L is not None:
                # Add the S_A0L operator acting on the left leg
                # updateLeft(S_A0L, 3, AK, [], [], AK) is used for the left-most operator
                SzK = SzK + update_left(S_A0L, 3, AK, None, None, AK)
                SzD = SzD + update_left(S_A0L, 3, AD, None, None, AD)
        else:
            # Iterations itN > 0
            # SzK = (SzKprev * I_new) + (I_prev * Sz_new)
            
            # SzK: Sz operator in kept state basis
            # Term 1: SzKprev extended by identity (on the current physical site)
            SzK_prev_ext = update_left(SzKprev, 2, AK, None, None, AK)
            # Term 2: Sz operator on the current site
            SzK_new_site = update_left(None, None, AK, Sz, 2, AK)
            SzK = SzK_prev_ext + SzK_new_site
            
            # SzD: Sz operator in discarded state basis
            SzD_prev_ext = update_left(SzKprev, 2, AD, None, None, AD)
            SzD_new_site = update_left(None, None, AD, Sz, 2, AD)
            SzD = SzD_prev_ext + SzD_new_site
            
        # --- Thermodyamic Calculation ---
        
        # Matrix of the Sz operator for the kept+discarded space
        # Sztot = blkdiag(SzK,SzD);
        Sztot = lin.block_diag(SzK, SzD)
        
        # Energy eigenvalues (rescaled)
        # E = [Inrg.EK{itN};Inrg.ED{itN}];
        E = np.concatenate((Inrg.EK[itN], Inrg.ED[itN]))
        
        # Zfac = exp(-E*(Inrg.EScale(itN)/T(itN))); # Boltzmann weights
        exponent = -E * (Inrg.EScale[itN] / T[itN])
        Zfac = np.exp(exponent.real) # Use real part for stability
        
        # Zsum = sum(Zfac); # partiton function
        Zsum = np.sum(Zfac)
        
        # rho = Zfac(:)/Zsum; # diagonal of density matrix
        # Ensure rho is a column vector if needed, but for dot product a flattened array is fine.
        rho = Zfac / Zsum
        
        # T*chi = < S_z^2 > - < S_z >^2 
        # Tchi0(itN) = (rho.'*diag(Sztot*Sztot)) - (rho.'*diag(Sztot))^2;
        Sz2_diag = np.diag(Sztot @ Sztot)
        
        mean_Sz2 = np.dot(rho, Sz2_diag)
        mean_Sz = np.dot(rho, np.diag(Sztot))
        
        Tchi0[itN] = mean_Sz2 - (mean_Sz)**2
        
        # Entropy: S = < H > / T + ln (Z)
        # Sent0(itN) = (rho.'*E)*(Inrg.EScale(itN)/T(itN)) + log(Zsum);
        mean_H_term = np.dot(rho, E) * (Inrg.EScale[itN] / T[itN])
        Sent0[itN] = mean_H_term + np.log(Zsum)
        
        # SzKprev = SzK; (for next iteration)
        SzKprev = SzK
    
    # --- Even-Odd Averaging ---
    
    T_all = T
    
    # Even and Odd temperature indices (0-based)
    T_even_indices = np.arange(0, N, 2)
    T_odd_indices = np.arange(1, N, 2)
    
    T_even = T_all[T_even_indices]
    T_odd = T_all[T_odd_indices]
    
    Tchi0_even = Tchi0[T_even_indices]
    Tchi0_odd = Tchi0[T_odd_indices]
    
    Sent0_even = Sent0[T_even_indices]
    Sent0_odd = Sent0[T_odd_indices]
    
    # Interpolation functions
    interp_Tchi_even = interp1d(T_even, Tchi0_even, kind='linear', fill_value='extrapolate')
    interp_Tchi_odd = interp1d(T_odd, Tchi0_odd, kind='linear', fill_value='extrapolate')
    
    interp_Sent_even = interp1d(T_even, Sent0_even, kind='linear', fill_value='extrapolate')
    interp_Sent_odd = interp1d(T_odd, Sent0_odd, kind='linear', fill_value='extrapolate')
    
    # Tchi = (interp1(T_even, Tchi0_even, T_all, ...) + interp1(T_odd, Tchi0_odd, T_all, ...))/2;
    Tchi = (interp_Tchi_even(T_all) + interp_Tchi_odd(T_all)) / 2.0
    
    # Sent = (interp1(T_even, Sent0_even, T_all, ...) + interp1(T_odd, Sent0_odd, T_all, ...))/2;
    Sent = (interp_Sent_even(T_all) + interp_Sent_odd(T_all)) / 2.0
    
    toc2(tobj,'-v')
    
    return T, Tchi, Sent

