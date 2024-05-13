import torch as tc
import torch.distributions as dist
import matplotlib.pyplot as plt

class Monte_Carlo:
    
    def __init__(self,r, sigma, T, time_steps = 100, S0 = 1):
        self.r = r
        self.sigma = sigma
        self.T = T
        self.time_steps = time_steps
        self.S0 = S0
        return
    
    
    def function_process(self):
        normal  = dist.Normal(loc = tc.tensor(0.0), scale = tc.tensor(1.0))
        delta_T = tc.tensor(self.T/self.time_steps)
        S = tc.empty(self.time_steps)
        S[0] = tc.tensor(self.S0)
        
        for i in range(1, self.time_steps):
            normal_sample = normal.sample()
            S[i] = S[i-1] + self.r*S[i-1]*delta_T + self.sigma*S[i-1]*normal_sample*tc.sqrt(delta_T)
        return S
    
    
    def calculate_option_value(self, K, num_simulation = 10, option_type = "Average Price", call_or_put = "call"):
        
        S_simulations = [self.function_process() for i in range(num_simulation)]
        
        for S in S_simulations:
            plt.plot(S)
        plt.axhline(y = K, linestyle = "dashed")
        plt.title("Simulation")
        plt.xlabel("Time")
        plt.ylabel("Asset Price")
        
        Payoffs = [self.calculate_payoff(S, K, option_type, call_or_put) for S in S_simulations]
        
        mean_payoffs = tc.tensor(Payoffs).mean()
        sd_payoffs = tc.tensor(Payoffs).std()
        discount_factor = tc.exp(tc.tensor(-self.r*self.T))
        
        option_price = discount_factor * mean_payoffs
        
        mc_standard_error = sd_payoffs/tc.sqrt(tc.tensor(num_simulation))
        
        print("Option price:", option_price)
        print("95% Credible Interval is between", option_price - 2*mc_standard_error, "and", option_price + 2*mc_standard_error)

        
        return option_price
    
    
    def calculate_payoff(self, S, K, option_type, call_or_put):
        
        match option_type:
            
            case "European":
                S_final = S[-1]  
            
            case "Average Price":
                S_final = S.mean()
                
            case "Average Strike":
                S_final = S[-1]
                K = S.mean()
                
            case _:
                raise Exception("option type not recognized")
                
        match call_or_put:
            
            case "call":
                return max(S_final - K, 0)
            
            case "put":
                return max(K - S_final, 0)
            
            case _:
                raise Exception("Must be either a `call` or a `put` option")
    