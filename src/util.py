import torch as tc
import torch.distributions as dist


def cal_d1(params):
    sigma, S0, K, r, T = params
    d1 = (tc.log(S0/K) + (r + tc.pow(sigma, 2)/ 2) * T)/(sigma*tc.sqrt(T))
    
    return d1

def cal_d2(params):
    sigma,_,_,_,T = params
    d1 = cal_d1(params)
    d2 = d1 - sigma*tc.sqrt(T)
    return d2


def N_prime(x): return tc.exp(dist.Normal(tc.tensor(0.), tc.tensor(1.)).log_prob(x))

def N(x): return dist.Normal(tc.tensor(0.), tc.tensor(1.)).cdf(x)

