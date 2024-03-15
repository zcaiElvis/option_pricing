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
    
    
    def create_grid(self, S_range = tc.tensor([0, 100])):
        self.Ts = tc.arange(0, self.T + self.delta_T, self.delta_T)
        self.Ss = tc.arange(S_range[0], S_range[1] + self.delta_K, self.delta_K)
        
        a = [self.a_star(j) for j in range(len(self.Ss))]
        b = [self.b_star(j) for j in range(len(self.Ss))]
        c = [self.c_star(j) for j in range(len(self.Ss))]
        
        M = len(self.Ss)
        N = len(self.Ts)
        
        f = tc.empty(N, M)
        
        f[-1,:] = tc.tensor([tc.max(tc.tensor(self.K - S), tc.tensor(0.)) for S in self.Ss])
        f[:, 0] = self.K
        f[:, -1] = 0
        
        
        for i in reversed(range(N - 1)):
            for j in range(1, M - 1):
                f[i, j] = a[j] * f[i+1, j-1] + b[j] * f[i+1, j] + c[j] * f[i+1, j+1]
        
        return f
        
        
if __name__ == "__main__":
    
    fd = finite_difference(0.1, 0, 0.2, 50, 0.4167, 5, 0.0417)
    fd.create_grid()