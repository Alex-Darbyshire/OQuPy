# Implementation of transverse contraction for a chain-mapped open quantum system
#Original code modified from https://github.com/emile-cochin/OQuPy/tree/TTIPT

from typing import Text

import numpy as np
from scipy.integrate import dblquad
from scipy.linalg import expm, norm, svd
from scipy.sparse.linalg import eigs
from scipy.special import gamma
from typing import Callable, Optional
from oqupy.util import get_progress
from ncon import ncon  # ncon performs better than np.einsum
from tqdm import tqdm
from itertools import product
from oqupy.bath_correlations import BaseCorrelations


class Chain_TC():
    """ A class to compute and approximate the influence functional using a modified iTEBD-TEMPO in a chain-mapped setup, and compute dynamics. """

    def __init__(self, S: np.ndarray, delta: float, N: int, J_prms: Optional[np.ndarray] = None, chain_prms: Optional[np.ndarray] = None, eps: Optional[float] = 10**-3, progress_type: Optional[Text] = None):
        """
        :param S: System-bath coupling operator.
        :param delta: time step for Trotter splitting.
        :param N: Chain cutoff, i.e. number of chain nodes EXCLUDING system. Should be chosen large enough such that nodes above this value are uniform.
        :param J_prms: Spectral density parameters, handed over as [\alpha : float, s : float, w_cutoff : float, soft : float (takes values 0 or 1)].
        :param chain_prms: Chain map Hamiltonian parameters, handed over as [e : np.ndarray, t : np.ndarray, c0 : float]
        :param eps: Relative error threshold between parameters of final two nodes in chain, used to check uniformity.  
        """

        self.progress_type = progress_type
        self.S = S
        self.S_dim = S.shape[0]
        self.delta = delta
        self.N = N
        self.eps = eps
        
        if chain_prms is None:
            if J_prms is None:
                raise ValueError("Neither chain nor spectral density parameters were specified")     
            else:
                self.J_prms=J_prms
                if len(self.J_prms) != 4:
                    raise ValueError("Incorrect format for spectral density parameters")
                elif self.J_prms[3] != 0 and self.J_prms[3] != 1:
                    raise ValueError("Incorrect cutoff argument given in spectral density parameters")
                else:    
                    self.chain_prms = self.chaincoeffs_ohmic(self.N, self.J_prms[0], self.J_prms[1], self.J_prms[2], self.J_prms[3])
                    
        elif len(chain_prms[0]) != self.N or len(chain_prms[1]) != self.N-1:
            raise ValueError("Chain Hamiltonian parameters not compatible with specified chain length")
        elif len(chain_prms) != 3:
            raise ValueError("Incorrect format for chain Hamiltonian parameters")
        else:
            self.chain_prms = chain_prms
        if np.abs((self.chain_prms[0][-1]-self.chain_prms[0][-2])/self.chain_prms[0][-1])>self.eps or np.abs((self.chain_prms[1][-1]-self.chain_prms[1][-2])/self.chain_prms[1][-1])>self.eps:
            print('Warning: Chain uniformity may not be accurate for given cutoff, the algorithm may become unstable. It is recommended to increase N until this message no longer appears.')
        
        self.f = None
        return

    def chaincoeffs_ohmic(self, nummodes: int, alpha: float, s: float, w_cutoff: float, soft: int):
        
        if soft==1:
            c0 = w_cutoff*np.sqrt(2*alpha*gamma(s+1))
            e = [w_cutoff*(2*n+1+s) for n in range(nummodes)]
            t = [w_cutoff*np.sqrt((n+1)*(n+s+1)) for n in range(nummodes-1)]
        else:
            if s==0:
                c0 = np.sqrt(2*alpha)*w_cutoff
                e = np.repeat(0.5*w_cutoff,nummodes)
                t = [w_cutoff*(1+n)**2/((2+2*n)*(3+2*n))*np.sqrt((3+2*n)/(1+2*n)) for n in range(nummodes-1)]
            else:
                c0 = np.sqrt(2*alpha/(s+1))*w_cutoff
                e = [(w_cutoff/2)*(1 +(s**2)/((s+2*n)*(2+s+2*n))) for n in range(nummodes)]
                t = [w_cutoff*(1+n)*(1+s+n)/((s+2+2*n)*(s+3+2*n))*np.sqrt((3+s+2*n)/(1+s+2*n)) for n in range(nummodes-1)]
            
        return [e, t, c0]
    
    def leading_eig(self, A, num_iterations: int =100):
        n = A.shape[0]
        # Start with a random initial vector
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)
        
        for _ in range(num_iterations):
            # Multiply matrix A by current vector
            av = np.dot(A, v)
            # Normalize the vector
            v = av / np.linalg.norm(av)          
        return v

    def ptrace(self, rho):
        chi=rho.shape[0]/self.S_dim
        rho_S=np.zeros((self.S_dim,self.S_dim),dtype=np.complex128)
        for i,j in product(range(self.S_dim), repeat=2):
            for k in range(int(chi)):            
                bas1=np.zeros(rho.shape[0],dtype=np.complex128)
                bas2=np.zeros(rho.shape[0],dtype=np.complex128)
                bas1[k*self.S_dim+i]=1
                bas2[k*self.S_dim+j]=1
                rho_S[i][j] += bas1 @ rho @ bas2
        return rho_S

        
    
    def compute_chain_f(self, rtol: float, d: int =2, rank: Optional[int] = np.inf, show_info=True):
        """
        Compute the infinite influence functional tensor f using the regular iTEBD algorithm.

        :param rtol: Relative tolerance for svd compression.
        :param d: Hilbert space cutoff for chain nodes.
        :param rank: Maximum allowed rank (bond dimension).
        """
        Ident=np.identity(d)
        b=np.zeros((d,d))
        bdag=np.zeros((d,d))
        number=np.zeros((d,d))
        for n in range(d-1):
            b[n][n+1]=np.sqrt(n+1)
            bdag[n+1][n]=np.sqrt(n+1)
            number[n+1][n+1]=n+1
        numb1=np.kron(number, Ident)
        numb2=np.kron(Ident, number)
        swap2to1=np.kron(bdag,b)
        swap1to2=np.kron(b,bdag)
        
        four_leg_gate=np.zeros((d,d,d,d), dtype=np.complex_)
            
        info_list = []
        A = np.zeros((d, d, d))            
        B = np.zeros((d, d, d))
        for i in range(d):
            A[i][i][i]=1
            B[i][i][i]=1
        sAB = np.repeat(1/np.sqrt(d),d)
        sBA = np.repeat(1/np.sqrt(d),d)

        progress = get_progress(self.progress_type)
        with progress(self.N, '--> Building influence functional:') as prog_bar:
            for k in range(1, self.N + 1):
                
                if k==self.N:
                    gate = expm(- 1j * self.delta * (0.5 * self.chain_prms[0][self.N - k] * numb1 + self.chain_prms[2] * np.kron((bdag + b),self.S)))
                else:
                    gate = expm(- 1j * self.delta * (0.5 * (self.chain_prms[0][self.N - k] * numb1 + self.chain_prms[0][self.N - k - 1] * numb2) + self.chain_prms[1][self.N - k -1] * (swap1to2 + swap2to1)))
                
                for m,n,p,q in product(range(d), repeat=4):
                        four_leg_gate[m][n][p][q]=gate.T[d*m+n][d*q+p]
                    
                if k % 2 == 0:
                    B, sBA, A, sAB, info = iTEBD_apply_gate(four_leg_gate, B, sBA, A, sAB, rank, rtol=rtol)
                else:
                    A, sAB, B, sBA, info = iTEBD_apply_gate(four_leg_gate, A, sAB, B, sBA, rank, rtol=rtol)

                if show_info:
                    info_str = f' d1:{info[0]}, d2:{info[1]}, chi:{info[2]}'
                    prog_bar.info = info_str
                    info_list.append(info)
                    
                prog_bar.update(k)
        print('Contracting to f...')
        self.f = ncon([np.diag(sAB), B, np.diag(sBA), A], [[-1, 1], [1, -2, 2], [2, 3], [3, -3, -4]])
        print('Computing boundary vectors...')
        """if sAB.shape[0] == 1:
            # handle trivial f
            self.f = np.ones((1, self.S_dim, self.S_dim, 1)) """

        # compute f^\inf = v_r * v_l^T using Lanczos
        ftemp = self.f.reshape(self.f.shape[0] * self.f.shape[1], self.f.shape[2] * self.f.shape[3])
        v_l = self.leading_eig(ftemp.T)
        
        #'Lower' index of leading eigenvectors is traced over, to give correct terminations to PT
        v_l = ncon([v_l.reshape([self.f.shape[0], self.f.shape[1]]), np.ones((self.f.shape[1]))], [[-1, 1],[1]])
        self.v_l = v_l / norm(v_l)
      
        
        print('rank ', self.f.shape[0])
        return info_list[:-1]
    
    def evolve(self, h_s: np.ndarray, psi_0: np.ndarray, n: int) -> np.ndarray:
        """
        Compute the time evolution for n time steps.
    
        :param h_s: System Hamiltonian.
        :param rho_0: System initial state.
        :param n: Number of time-steps for the propagation. 
        :return: Time evolution of density matrix.
        """
    
        assert self.f is not None, "the f gate has not yet been computed, run self.compute_f first"
    
        u_s = expm(-1j * h_s * self.delta)
    
        rho_t = np.empty((n + 1, self.S_dim, self.S_dim), dtype=np.complex128)
        rho_t[0] = np.outer(psi_0,psi_0)
    
        evol_tens = ncon([self.f, u_s], [[-1, -2, 1, -4], [1, -3]])
        state = ncon([self.v_l, psi_0], [[-2], [-3]])
        for i in tqdm(range(n), desc='time evolution running'):
            state = ncon([state, evol_tens], [[1, 2], [1, 2, -3, -2]])
            rho_t[i + 1] = self.ptrace(np.outer(state.reshape(evol_tens.shape[3],evol_tens.shape[2]),state.reshape(evol_tens.shape[3],evol_tens.shape[2])))
        return rho_t
    
    """def get_maps(self, h_s: np.ndarray, n: int) -> np.ndarray:
        Compute the (time-nonlocal) dynamical maps for n time steps.
    
        :param h_s: System Hamiltonian in the eigenbasis of the coupling operator.
        :param n: Number of time-steps for the propagation.
        :return: Time evolution of density matrices.
    
        assert self.f is not None, "the influence functional has not yet been computed, run self.compute_f first"
    
        liu_s = np.kron(expm(-1j * h_s * self.delta / 2), expm(1j * h_s * self.delta / 2).T)
        u = np.einsum('ab,bc->abc', liu_s.T, liu_s.T)
    
        evol_tens = ncon([self.f[:, :-1, :], u], [[-1, 2, -3], [-2, 2, -4]])
        temp_map = ncon([self.v_l, evol_tens], [[1], [1,-1,-2,-3]])
       
        dyn_map = np.empty((n, self.s_dim**2, self.s_dim**2), dtype=np.complex128)
        dyn_map[0] = ncon([self.v_r, temp_map], [[1], [-2,1,-1]])
        
        for i in tqdm(range(n-1), desc='time evolution running'):
            temp_map = ncon([temp_map, evol_tens], [[-1, 1, 2], [1, 2, -2, -3]])
            dyn_map[i + 1] = ncon([self.v_r, temp_map], [[1], [-2, 1, -1]])
        return dyn_map"""

def iTEBD_apply_gate(gate: np.ndarray, A: np.ndarray, sAB: np.ndarray, B: np.ndarray, sBA: np.ndarray, rank: int, rtol: float, ctol: Optional[float] = 1e-13):
    """
    single iTEBD step, scheme adapted from https://www.tensors.net/mps
    :param gate: TEBD gate for A-B link
    :param A: A tensor (left)
    :param sAB: weight for A-B link
    :param B: B tensor (right)
    :param sBA: weight for B-A link
    :param rank: maximum rank in svd compression
    :param rtol: relative error for svd compression
    :param ctol: cutoff for weights sBA which need to be inverted
    :return: new tensors and weights A, sAB, B, sBA
    """

    # renormalize weights
    sAB = sAB * norm(sBA)
    sBA = sBA / norm(sBA)

    # ensure weights are above tolerance (needed for inversion)
    sBA[np.abs(sBA) < ctol] = ctol

    # MPS - gate contraction
    d1 = gate.shape[1]
    d2 = gate.shape[2]
    rank_BA = sBA.shape[0]

    u, s_vals, v = svd(np.einsum('a,acd,d,dfg,g,cnmf->anmg', sBA, A, sAB, B, sBA, gate).reshape([d1 * rank_BA, d2 * rank_BA]), full_matrices=False)
    # truncate singular values
    if rtol is None:
        rank_new = min(rank, len(s_vals))
    else:
        s_vals_sum = np.cumsum(s_vals) / np.sum(s_vals)
        rank_rtol = np.searchsorted(s_vals_sum, 1 - rtol) + 1
        rank_new = min(rank, len(s_vals), rank_rtol)
    u = u[:, :rank_new].reshape(rank_BA, d1 * rank_new)
    v = v[:rank_new, :].reshape(rank_new * d2, rank_BA)
    
    # factor out sAB weights from A and B
    A = (np.diag(1 / sBA) @ u).reshape(rank_BA, d1, rank_new)
    B = (v @ np.diag(1 / sBA)).reshape(rank_new, d2, rank_BA)

    # new weights
    sAB = s_vals[:rank_new]

    return A, sAB, B, sBA, (d1, d2, rank_new)
