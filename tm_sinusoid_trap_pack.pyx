import cython
import numpy as np
cimport numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from libc.math cimport cos
from libc.math cimport sin, exp, pi
ctypedef np.float64_t dtype_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef dtype_t phi_prime_1(dtype_t q, dtype_t alpha, dtype_t l, dtype_t pi) : # Potential' 1
    return 2 * pi * alpha/l * cos(2 * pi * q/l)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef dtype_t phi_prime_2(dtype_t q, dtype_t kappa) : # Potential' 2
    return q/kappa

@cython.cdivision(True)
cdef dtype_t phi_1(dtype_t q, dtype_t alpha, dtype_t l, dtype_t pi) :    # Potential 1
    return alpha * sin(2 * pi * q/l)     

@cython.cdivision(True)
cdef dtype_t phi_2(dtype_t q, dtype_t kappa) : # Potential 2
    return q**2 / (2 * kappa)    



def _Peq_projection1D_q1(q1, alpha, l, pi) : # Peq particle
    projection = exp(-(phi_1(q1, alpha, l, pi)))
    return projection

def _Peq_projection1D_q2(q2, kappa) : # Peq wall
    projection = exp(-(phi_2(q2, kappa)))
    return projection

def Peq_projection1D_q1(q1, alpha, l, pi): # Normalised 
    P = np.array([_Peq_projection1D_q1(q1i, alpha, l, pi) for q1i in q1])
    return P / np.trapz(P, q1)

def Peq_projection1D_q2(q2, kappa): # Normalised 
    P = np.array([_Peq_projection1D_q2(q2i, kappa) for q2i in q2])
    return P / np.trapz(P, q2)




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef qi_full_arrays(dtype_t[:] eta_1, dtype_t[:] eta_2,
                    dtype_t gamma_11, dtype_t gamma_12, dtype_t gamma_22,
                    dtype_t kappa, dtype_t alpha, dtype_t l, int N, dtype_t tau):

    cdef dtype_t[:] q1 = np.zeros(N)
    cdef dtype_t[:] q2 = np.zeros(N)
    cdef dtype_t pi = 3.14159265358979323846

    cdef dtype_t gamma_ratio_22 = gamma_12 / gamma_22
    cdef dtype_t gamma_ratio_11 = gamma_12 / gamma_11

    cdef np.ndarray[dtype_t, ndim=1] q1_theo = np.linspace(-l, l, N)
    cdef np.ndarray[dtype_t, ndim=1] q2_theo = np.linspace(-.1, .1, N)


    inverse_cdf_q1 = sample_sin(q1_theo, Peq_projection1D_q1, alpha, l, pi)
    inverse_cdf_q2 = sample_optic(q2_theo, Peq_projection1D_q2, kappa)
    
    q1[0] = inverse_cdf_q1(np.random.uniform(0.000000000000001, (1-0.000000000000001)))
    q2[0] = inverse_cdf_q2(np.random.uniform(0.000000000000001, (1-0.000000000000001)))

    cdef dtype_t _phi_prime_1
    cdef dtype_t _phi_prime_2

    cdef int i
    for i in range(1, N):  
        _phi_prime_1 = phi_prime_1(q1[i-1], alpha, l, pi) 
        _phi_prime_2 = phi_prime_2(q2[i-1], kappa)   
        q1[i], q2[i] = update_q_values(
            q1[i - 1], q2[i - 1], 
            _phi_prime_1, 
            _phi_prime_2, 
            eta_1[i - 1], eta_2[i - 1],
            gamma_11, gamma_12, gamma_22, gamma_ratio_11, gamma_ratio_22, tau)

    return np.asarray(q1), np.asarray(q2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef (dtype_t,dtype_t) update_q_values(
    dtype_t q1_current, dtype_t q2_current, dtype_t phi_prime_1_current, 
    dtype_t phi_prime_2_current, dtype_t eta_1_current, dtype_t eta_2_current, 
    dtype_t gamma_11, dtype_t gamma_12, dtype_t gamma_22, dtype_t gamma_ratio_11,
    dtype_t gamma_ratio_22, dtype_t tau):

    cdef dtype_t q1_next, q2_next

    q1_next = q1_current + (gamma_11 - (gamma_12**2 / gamma_22))**(-1) * (
        tau * (- phi_prime_1_current + gamma_ratio_22 * phi_prime_2_current) +
        eta_1_current - gamma_ratio_22 * eta_2_current)
    
    q2_next = q2_current + (gamma_22 - (gamma_12**2 / gamma_11))**(-1) * (
        tau * (- phi_prime_2_current + gamma_ratio_11 * phi_prime_1_current) +
        eta_2_current - gamma_ratio_11 * eta_1_current)
    
    return q1_next, q2_next


def friction_matrix(g_11, g_12, g_22) :
    gamma_11 = g_11 ** 2 + g_12 ** 2
    gamma_12 = g_12 * (g_11 + g_22)    
    gamma_22 = g_22 ** 2 + g_12 ** 2 
    return gamma_11, gamma_12, gamma_22

def trajectory_sin_trap(N, g_11, g_12, g_22, tau, dtype_t kappa, dtype_t alpha, dtype_t l) :
    cdef dtype_t gamma_11, gamma_12, gamma_22
    gamma_11, gamma_12, gamma_22 = friction_matrix(g_11, g_12, g_22)
    

    cdef dtype_t[:] xi_1 = np.sqrt(2) * np.random.normal(0, np.sqrt(tau), N)

    cdef dtype_t[:] xi_2 = np.sqrt(2) * np.random.normal(0, np.sqrt(tau), N)


    cdef dtype_t[:] eta_1 = g_11 * np.asarray(xi_1) + g_12 * np.asarray(xi_2)

    cdef dtype_t[:] eta_2 = g_12 * np.asarray(xi_1) + g_22 * np.asarray(xi_2)


    q1, q2 = qi_full_arrays(eta_1, eta_2, gamma_11, gamma_12, gamma_22, kappa, alpha, l, N, tau)
    return q1, q2



    

def sample_optic(q_sample, Peq_projection1D_q, kappa) : 
    y_q = Peq_projection1D_q2(q_sample, kappa) 
    cdf_y_q = np.cumsum (y_q)
    cdf_y_q = cdf_y_q/cdf_y_q.max() 
    inverse_cdf_q = interp1d(cdf_y_q,q_sample) 
    return inverse_cdf_q

def sample_sin(q_sample, Peq_projection1D_q, alpha, l, pi) : 
    y_q = Peq_projection1D_q1(q_sample, alpha, l, pi) 
    cdf_y_q = np.cumsum (y_q)
    cdf_y_q = cdf_y_q/cdf_y_q.max() 
    inverse_cdf_q = interp1d(cdf_y_q,q_sample) 
    return inverse_cdf_q