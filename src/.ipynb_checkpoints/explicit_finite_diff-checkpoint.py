import torch as tc



class finite_difference:
    
    def __init__(self, r, q, sigma, K, T, delta_K, delta_T):
        self.r = r
        self.q = q
        self.sigma = sigma
        self.K = K
        self.T = T
        self.delta_K = delta_K
        self.delta_T = delta_T
        
        return
    
    
    def a_star(self, j):
        return 1/(1+self.r * self.delta_T) * (-0.5*(self.r-self.q)*j*self.delta_T + 0.5 * self.sigma**2 * j**2 * self.delta_T)
    
    
    def b_star(self, j):
        
        return 1/(1+self.r * self.delta_T) * (1 - self.sigma**2 * j**2 * self.delta_T)
    
    
    def c_star(self, j):
        
        return 1/(1+self.r * self.delta_T) * (0.5*(self.r-self.q)*j*self.delta_T + 0.5 * self.sigma**2 * j**2 * self.delta_T)
    
    
    def create_grid(self, K_range = tc.tensor([0, 100])):
        self.Ts = tc.arange(0, self.T + self.delta_T, self.delta_T)
        self.Ks = tc.arange(K_range[0], K_range[1] + self.delta_K, self.delta_K)
        
        a = [self.a_star(j) for j in range(len(self.Ks))]
        b = [self.b_star(j) for j in range(len(self.Ks))]
        c = [self.c_star(j) for j in range(len(self.Ks))]
        
        M = len(self.Ks)
        N = len(self.Ts)
        
        f = tc.empty(N, M)
        
        f[-1,:] = tc.tensor([tc.max(tc.tensor(K - S), tc.tensor(0.)) for S in K])
        f[:, 0] = K
        f[:, -1] = 0
        
        
        for i in reversed(range(N - 1)):
            for j in range(M - 1):
                print("##")
                print(i)
                print(j)
        
        return
        
        
if __name__ == "__main__":
    
    fd = finite_difference(0.1, 0, 0.2, 50, 0.4167, 5, 0.0417)
    
    fd.create_grid()