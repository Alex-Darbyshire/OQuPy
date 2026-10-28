import oqupy
import numpy as np
import pickle
from oqupy import Chain_TC, iTEBD_tempo
import matplotlib.pyplot as plt

sigma_x = np.array([[0,1],[1,0]])
sigma_z = np.array([[1,0],[0,-1]])

#Define full basis of initial density matrices
dm1 = np.array([[1.+0.j, 0.+0.j],[0.+0.j,0.+0.j]])
dm2 = np.array([[0.+0.j,1.+0.j],[0.+0.j,0.+0.j]])
dm3 = np.array([[0.+0.j,0.+0.j],[1.+0.j,0.+0.j]])
dm4 = np.array([[0.+0.j,0.+0.j],[0.+0.j,1.+0.j]])
dm=[dm1,dm2,dm3,dm4]

Omega = 1.0
omega_cutoff = 20
alpha = 0.7
s=1
epsrel=10**(-4)

J_prms=[alpha, s, omega_cutoff, 0]

dt=0.1

N=7

S=0.5*sigma_z

H_s=0.5*Omega*sigma_x

psi_0=np.asarray([0,1])

evals, evecs=np.linalg.eig(S)

tcut=5

end_time=15

correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=1,
                                cutoff=omega_cutoff,
                                cutoff_type='exponential')

func=Chain_TC.Chain_TC(S, dt, N, J_prms)
func.compute_chain_f(epsrel)
states=func.evolve(H_s, psi_0, int(end_time/dt))

"""tempo_parameters = oqupy.TempoParameters(dt=0.1, tcut=3.0, epsrel=10**(-4))

system = oqupy.System(H_s)

bath = oqupy.Bath(0.5 * sigma_z, correlations)

dynamics = oqupy.tempo_compute(system=system,
                               bath=bath,
                               initial_state=dm4,
                               start_time=0.0,
                               end_time=15.0,
                               parameters=tempo_parameters,
                               unique=False)"""
"""func_benchmark=iTEBD_tempo.iTEBD_TEMPO(evals,delta=dt,bath_correlations=correlations, n_c=int(tcut/dt))
func_benchmark.compute_f_enhanced(epsrel,epsrel,epsrel)"""

t=np.arange(0,end_time+dt,0.1)

plt.plot(t, states[:,0,0])
