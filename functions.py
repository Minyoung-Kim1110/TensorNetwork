import numpy as np 
import numpy.linalg as lin 
from typing import List, Tuple

MATLAB_style=True
order_type = 'F' if MATLAB_style else 'C'

def get_MPS_QR(tensor:np.array)->List[np.array]:
    """Decompose a high rank tensor to the matrix product states(MPS) by QR factorization 
    
    Args:
        tensor (np.array): a high rank tensor

    Returns:
        MPS (List[np.array]): Matrix product state of a given high rank tensor 
    
    Written by M.Kim (Sep.08 2022)
    """
    tensor_dim = list(tensor.shape)
    MPS = []
    R = tensor 
    szl = 1 # bond dimension of left leg of MPS[i]
    for i in range(len(tensor_dim)-1):
        R = R.reshape((szl * tensor_dim[i], np.prod(tensor_dim[i+1:])), order=order_type)
        Q, R = lin.qr(R, mode = 'reduced')
        Q = np.transpose(Q.reshape((szl, tensor_dim[i], -1), order=order_type), (0, 2, 1))
        MPS.append(Q)
        (_, szl, _)  = Q.shape 
        R = R.reshape((tensor_dim[i+1:].insert(0, szl)), order=order_type)
    MPS.append(np.transpose(R[:, np.newaxis], (0, 2, 1)))
    return MPS

def get_MPS_SVD(tensor:np.array, tol = 10**(-16))->Tuple[List[np.array], List[float]]:
    """Decompose a high rank tensor to the matrix product states(MPS) by SVD decomposition  
    
    Args:
        tensor (np.array): a high rank tensor
        tol (float) : tolerance for floating point noise

    Returns:
        MPS, entropys (Tuple[List[np.array], List[float]]): 
        MPS: Matrix product state of a given high rank tensor
        entropys: entropy of each bipartition
    
    Written by M.Kim (Sep.08 2022)
    """
    tensor_dim = list(tensor.shape) 
    MPS = []
    entropys = []
    szl = 1 
    A = tensor
    for i in range(len(tensor_dim)-1):
        A = A.reshape((szl*tensor_dim[i], np.prod(tensor_dim[i+1:])), order=order_type)
        U, S, Vh = lin.svd(A, full_matrices=False)
        S_filtered = S>tol
        U , S, Vh= U[:, S_filtered], S[S_filtered], Vh[S_filtered, :]
        entropys.append(entropy(S))
        U = np.transpose(U.reshape((szl, tensor_dim[i], -1), order=order_type), (0, 2, 1))
        MPS.append(U)
        (_, szl, _) = U.shape
        A = (np.diag(S)@Vh).reshape((tensor_dim[i+1:].insert(0, szl)), order=order_type)
    MPS.append(A[:, np.newaxis, :])
    return (MPS, entropys)

def MPS_to_tensor(MPS:List[np.array]):
    """Reconstruct the high rank tensor from Matrix product states 

    Args:
        MPS (List[np.array]): matrix product states

    Returns:
        A (np.array):a high rank tensor

    Written by M.Kim (Sep.10 2022)
    """
    A = MPS[0]
    for i in range(1, len(MPS)):
        rank = len(A.shape)
        A = contract(A, np.transpose(MPS[i], (0, 2, 1)), [len(A.shape)-1], [0])
    
    return A.squeeze()

def entropy(s: np.array)->float: 
    """from singular values, compute entropy 

    Args:
        s (np.array): singluar values 

    Returns:
        entropy (float): Shannon entropy with log_2
    
    Written by M.Kim (Sep.08 2022)
    """
    s = s*s 
    return - np.dot(s, np.log2(s))

def contract(A:np.array, B: np.array, contract_idx_A:List[int], contract_idx_B: List[int]): 
    """tensor contraction using numpy library 

    Args:
        A (np.array): a tensor 
        B (np.array): a tensor 
        contract_idx_A (List[int]): indices of A to contract 
        contract_idx_B (List[int]): indices of B to contract 

    Returns:
        tensor(np.array): contracted tensor 
    Written by M.Kim (Sep.10 2022)
    """
    return np.tensordot(A, B, axes = (contract_idx_A, contract_idx_B))

def check_equality_tensor(A: np.array, B : np.array, tol = 10 ** ( -15) ): 
    """check equality of tensors A and B 
        Return True if A == B upto tolerance(tol)

    Args:
        A (np.array): a tensor 
        B (np.array): a tensor 
        tol (float, optional): criteria for checking equality of double precision variables. Defaults to 10**( -15).

    Returns:
        (Boolean): Whether two tensors are equal or not 

    Written by M.Kim (Sep.10 2022)
    """
    if A.shape != B.shape:
        return False 
    if np.sum(np.abs(A-B)>tol)>1:
        return False 
    else:
        return True 



if __name__ == '__main__':
    dim = [2,3,2,3,4]
    T = (np.arange(np.prod(dim))+1).reshape(dim, order=order_type).astype(np.float64) # using 64 bit float 
    T = T/lin.norm(T.flatten())
    print(T.shape)

    
    def check_integrity_QR(T):     
        MPS = get_MPS_QR(T)
        T_reconstructed = MPS_to_tensor(MPS)
        return check_equality_tensor(T, T_reconstructed)

    def check_integrity_SVD(T): 
        (MPS, entropys )= get_MPS_SVD(T, tol = 2**(-40))
        T_reconstructed = MPS_to_tensor(MPS)
        for entropy in entropys: 
            print(f"{entropy:.5f}")
        return check_equality_tensor(T, T_reconstructed)
    
    if check_integrity_QR(T):
        print(f'integrity of to MPS using QR Succeed!')
    else: 
        print(f'integrity of to MPS using QR Failed!')
    
    if check_integrity_SVD(T):
        print(f'integrity of to MPS using SVD Succeed!')
    else: 
        print(f'integrity of to MPS using SVD Failed!')
    