from src.util import *


def delta_euro_call(params): return N(cal_d1(params))

def delta_euro_put(params): return N(cal_d1(params))-1

def theta_euro_call(params): 
    sigma, S0, K, r, T = params
    d1 = cal_d1(params)
    d2 = cal_d2(params)
    
    theta = (-S0*N_prime(d1)*sigma)/(2*tc.sqrt(T)) - r*K*tc.exp(-r*T)*N(d2)
    return theta

def theta_euro_put(params): 
    sigma, S0, K, r, T = params
    d1 = cal_d1(params)
    d2 = cal_d2(params)
    
    theta = (-S0*N_prime(d1)*sigma)/(2*tc.sqrt(T)) + r*K*tc.exp(-r*T)*N(-d2)
    return theta


def gamma_euro(params): 
    sigma, S0, K, r, T = params
    return N_prime(cal_d1(params))/(S0*sigma*tc.sqrt(T))


def vega_euro(params):
    sigma, S0, K, r, T = params
    return S0*tc.sqrt(T)*N_prime(cal_d1(params))

def rho_euro_call(params):
    sigma, S0, K, r, T = params
    d2 = cal_d2(params)
    return K*T*tc.exp(-r*T)*N(d2)

def rho_euro_put(params):
    sigma, S0, K, r, T = params
    d2 = cal_d2(params)
    return -K*T*tc.exp(-r*T)*N(-d2)
