import torch as tc



class imp_finite_difference:
    
    def __init__(self, r, q, sigma, K, T, delta_S, delta_T, eu_am = "European"):
        self.r = r
        self.q = q
        self.sigma = sigma
        self.K = K
        self.T = T
        self.delta_S = delta_S
        self.delta_T = delta_T
        self.eu_am = eu_am
        
        return
    
    
    def get_a(self, j):
        return (1/2)*(self.r - self.q) * j * self.delta_T - (1/2) * self.sigma**2 * j**2 * self.delta_T
    
    
    def get_b(self, j):
        
        return 1 + self.sigma**2 * j**2 * self.delta_T + self.r * self.delta_T
    
    
    def get_c(self, j):
        
        return -(1/2)*(self.r - self.q) * j * self.delta_T - (1/2) * self.sigma**2 * j**2 * self.delta_T
    
    
    def create_tridiag_matrix(self, a, b, c, M, N):
        tridiag_main = tc.diag(b, 0)
        tridiag_up = tc.diag(c[0:(len(c)-1)], 1)
        tridiag_down = tc.diag(a[1:len(a)], -1)

        tridiag = tridiag_main + tridiag_up + tridiag_down
        
        tridiag_inv = tc.inverse(tridiag)
        print("inveresed")
            
        return tridiag_inv
    
    
    def create_grid(self, S_range = tc.tensor([0, 100])):
        self.Ts = tc.arange(0, self.T + self.delta_T, self.delta_T)
        self.Ss = tc.arange(S_range[0], S_range[1] + self.delta_S, self.delta_S)
        
        a = [self.get_a(j) for j in range(len(self.Ss))]
        b = [self.get_b(j) for j in range(len(self.Ss))]
        c = [self.get_c(j) for j in range(len(self.Ss))]
        
        M = len(self.Ss)
        N = len(self.Ts)
        
        f = tc.empty(N, M)
        tridiag_inv = self.create_tridiag_matrix(tc.tensor(a), tc.tensor(b), tc.tensor(c), M, N)
        
        f[-1,:] = tc.tensor([tc.max(tc.tensor(self.K - S), tc.tensor(0.)) for S in self.Ss])
        f[:, 0] = self.K
        f[:, -1] = 0
        
        for i in reversed(range(N - 1)):
            price = tc.matmul(tridiag_inv, f[i,:])
            print(price)
            payoff = f[i,:] - self.K

            # tf = [payoff[j] > price[j] for j in range(M)]
            # price[tf] = payoff[tf]
            
            f[(i-1),:] = price
            
        return f
            
            
        
        
if __name__ == "__main__":
    
    fd = finite_difference(0.1, 0, 0.2, 50, 0.4167, 5, 0.04167, eu_am = "American")
    print(fd.create_grid())