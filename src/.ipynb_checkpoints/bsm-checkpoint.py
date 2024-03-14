from src.util import *


def calculate_bsm_option_price(params, option_type = "call", dividend = None, q = 0):
    '''
    `params` is in the order of [sigma, S0, K, r, T]
    '''
    sigma, S0, K, r, T = params
    d1 = cal_d1(params)
    d2 = cal_d2(params)
    def N(x): return(dist.Normal(tc.tensor(0.), tc.tensor(1.)).cdf(x))
    
    price = 0
    
    match option_type:
        
        case "call":
            price = S0*N(d1) - K*tc.exp(-r*T)*N(d2)
        
        case "put":
            price = K*tc.exp(-r*T)*N(-d2) - S0*N(-d1)
            
    return price