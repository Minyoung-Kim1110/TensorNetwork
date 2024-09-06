import numpy as np 
import numpy.linalg as lin 
from typing import List, Tuple

MATLAB_style=False
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
        # rank = len(A.shape)
        A = contract(A, np.transpose(MPS[i], (0, 2, 1)), [len(A.shape)-1], [0])
    
    return A.squeeze()

def get_identity(ket: np.array, 
                idx_ket: int,
                operator=None, # np.array 
                idx_operator=None, 
                ):
    """Get the identity operator that connects the ket and the operator. 
    
    Args:
        ket (np.array): a ket with three legs 
        idx_ket (int) : indicate which leg to connect 
                        0 : left leg, 1: right leg, 2:bottom leg 
        operator (np.array): a operator 
        idx_operator (int) : indicate which leg to connect 
                        order : bottom - up - left - right 
    Returns:
        identity (np.array): an identity operator

    Written by M.Kim (Sep.06 2024)
    """
    dim_ket = ket.shape[idx_ket] # 0 : left leg, 1: right leg, 2: bottom leg
    if operator is None and idx_operator is None: 
        identity = np.eye(dim_ket)
        
    else: 
        dim_operator = operator.shape[idx_operator]
        identity = np.eye(dim_ket * dim_operator)
        identity = identity.reshape(dim_ket, dim_operator, dim_ket*dim_operator, order=order_type)
        identity = np.transpose(identity, (0, 2, 1))
        
    '''if permute == None: 
        return identity
    else: 
        if len(permute)==len(identity.shape):
            return np.transpose(identity, permute)
        elif len(permute)==len(identity.shape)+1:
            if len(identity.shape)==2:
                return np.transpose(identity[:, :, None], permute)
            elif len(identity.shape)==3:
                return np.transpose(identity[:, :,:,  None], permute)
        else: 
            raise ValueError("permute idx is longer than identity tensor")'''
        
    
def updateLeft(Cleft, B, X, A):
    """Update Cleft with ket A, bra B, with operator X 
    Detailed concatenation is described below. 
    Args:
        Cleft (np.array): the left tensor
        B (np.array): a ket to be producted as a bra 
        X (np.array) : an operator 
        A (np.array) : a ket 
        
    Returns:
        Cleft (np.array): updated Cleft with A, B, X 

    Written by M.Kim (Sep.06 2024)
    """
    
    ''' B' : Hermitian conjugate of B (complex conjugate and permute left, right legs) 
            Hence, the left leg (0) of B' = right leg (1) of B  
    * When Cleft is rank-2 and X is rank-2:
                    0         1
            /--------->- A ->--            /---->-- 1
            |            | 2               |
          1 ^            ^                 |
            |            | 1               |      
        Cleft            X         =>    Cleft 
            |            | 0               |
          0 ^            ^                 |
            |            | 2               |
            \---------<- B'-<--            \----<-- 0
                    1        0
        * When Cleft is rank-3 and X is rank-2:
                      0     1
            /--------->- A ->--            /---->-- 1
            |            | 2               |
          1 ^            ^                 |
            |    3       | 1               |      
        Cleft---       X         =>    Cleft ---- 2
            |            | 0               |
          0 ^            ^                 |
            |            | 2               |
            \---------<- B'-<--            \----<-- 0
                    1       0
        * When Cleft is rank-2 and X is rank-3:
                      0      1
            /--------->- A ->--            /---->-- 1
            |            | 2               |
          1 ^            ^                 |
            |          1 |   2             |      
          Cleft          X ----    =>    Cleft ---- 2
            |          0 |                 |
          0 ^            ^                 |
            |            | 2               |
            \---------<- B'-<--            \----<-- 0
                      1     0
        * When both Cleft and X are rank-3:
                      0     1
            /--------->- A ->--            /---->-- 1
            |            | 2               |
          1 ^            ^                 |
            |   2     2  | 1               |      
        Cleft--------- X         =>    Cleft
            |            | 0               |
          0 ^            ^                 |
            |            | 2               |
            \---------<- B'-<--            \----<-- 0
                     1     0
        * When Cleft is rank-3 and X is rank-4:
                      0     1
            /--------->- A ->--            /---->-- 1
            |            | 2               |
          1 ^            ^                 |
            |   2    2   | 1               |      
          Cleft--------- X ---- 2   =>   Cleft ---- 2
            |            | 0               |
          0 ^            ^                 |
            |            | 2               |
            \---------<- B'-<--            \----<-- 0
                      1     0
    Retreived from ssLee's TensorNetwork2022 
    '''    
    
    B = np.conj(B)
    if Cleft is None and X is None: 
        Cleft = contract(B, A, [0, 2], [0,2] )
        
    elif Cleft is not None and X is None: 
        rankC = len(Cleft.shape)
        #contract up_Cleft, left_A
        T = contract(Cleft, A, [1], [0]) # bottom_C, (right_C), right_A, bottom_A
        # contract bottom_C, left_B' 
        # contract up_B', bottom_A
        Cleft = contract(B, T, [0,2], [0,rankC]) # if rankC=2: no right_C, contract with bottom A 
                                                # if rankC=3: contract with bottom_A 
                                                # right_B, (right_C), right_A
        if rankC == 3: 
            Cleft=np.transpose(Cleft, (0, 2, 1))
    elif Cleft is None and X is not None: 
        rankX = len(X.shape)
        if rankX == 4: 
            raise ValueError('Dimension of X has error')
        if rankX == 2: 
            
            T = contract(X, A, [1], [2]) # bottom_X,  left_A, right_A
            # contract leftA, left_B 
            # contract bottom_X, up_B 
            Cleft = contract(B, T, [0, 2], [1, 0])    # right_B, right_A 
            
        if rankX==3:    
            T = contract(X, A, [1], [2]) # bottom_X, right_X, left_A, right_A
            #left_A, left_B' 
            # bottom_X - up_B' 
                                            
            Cleft = contract(B, T, [0, 2], [2, 0]) # right_B, right_X, right_A 
            Cleft = np.transpose(Cleft, (0, 2, 1))                                  
            
    else: 
        rankC = len(Cleft.shape)
        rankX = len(X.shape)
        
        # contract bottom_A, up_X
        T = contract(X, A, [1], [2]) # bottom_X, (leftX, right_X), left_A, right_A
        
        if (rankC, rankX)==(2, 2):
            #contract up_Cleft, left_A
            T = contract(Cleft, T, [1],[1] ) # bottom_C, bottom_X, right_A
            Cleft = contract(B, T, [0, 2], [0, 1]) #left_B'-bottom_C 
                                            # up_B' - bottom_X 
                                            # right_B', right_A  
    
        elif (rankC, rankX)==(2, 3):
            #contract up_Cleft, left_A
            T = contract(Cleft, T, [1],[2] ) # bottom_C, bottom_X, right_X, right_A
            Cleft = contract(B, T, [0, 2], [0, 1]) #left_B'-bottom_C 
                                            # up_B' - bottom_X 
                                            # right_B', right_X, right_A  
            Cleft = np.transpose(Cleft, (0, 2, 1))
        
        elif (rankC, rankX)==(3, 2):
            #contract up_Cleft, left_A
            T = contract(Cleft, T, [1],[1] ) # bottom_C, right_C, bottom_X,  right_A
            Cleft = contract(B, T, [0, 2], [0, 2]) #left_B'-bottom_C 
                                            # up_B' - bottom_X 
                                            # right_B',right_C, right_A  
            Cleft = np.transpose(Cleft, (0, 2, 1))
            
        elif (rankC, rankX)==(3, 3):
            #contract up_Cleft, left_A
            # right_C, left_X                                 
            T = contract(Cleft, T, [1, 2],[2, 1] ) # bottom_C, bottom_X,  right_A
            # contract 
            #left_B'-bottom_C 
            # up_B' - bottom_X 
            Cleft = contract(B, T, [0, 2], [0, 1]) # right_B',right_A  
        elif (rankC, rankX)==(3, 4):
              
            #contract up_Cleft, left_A
            # right_C, left_X                                 
            T = contract(Cleft, T, [1, 2],[3, 1] ) # bottom_C, bottom_X,right_X, right_A
            # contract 
            #left_B'-bottom_C 
            # up_B' - bottom_X 
            Cleft = contract(B, T, [0, 2], [0, 1]) # right_B',right_X, right_A  
            Cleft = np.transpose(Cleft, (0, 2, 1))    
    
    return Cleft 
    
    
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
    