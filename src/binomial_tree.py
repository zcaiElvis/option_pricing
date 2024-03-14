import torch as tc
import torch.distributions as dist



class binomial_tree:
    
    def __init__(self, name, path, value, params, depth, children, option_type = "call", eu_am = "European"):
        self.name = name
        self.path = path
        self.value = value
        self.params = params
        self.depth = depth
        self.children = children
        
        self.option_type = option_type
        self.eu_am = eu_am
        
        self.get_tree_params()
        

    def get_tree_params(self):
        '''
        Calculate parameters to construct the tree, including
        p - probability the price will go up
        u - the percentage increase of the price if it goes up
        d - the percentage decrease of the price if it goes down
        '''
        self.sigma, self.S0, self.K, self.r, self.q, self.T = self.params
        self.a = tc.exp((self.r - self.q) * self.T)
        self.u = tc.exp(self.sigma * tc.sqrt(self.T))
        self.d = 1/self.u
        self.p = (self.a - self.d)/(self.u - self.d)
        
    def value_option(self, path):
        '''
        Value the option at leaf nodes. Can be overwritten for exotic options
        '''
        
        multiplyer = tc.prod(tc.tensor(path))
            
        self.St = self.S0 * multiplyer


        if self.option_type == "call":

            if self.St < self.K:
                value = 0
            else:
                value = self.St - self.K

        elif self.option_type == "put":

            if self.St > self.K:
                value = 0
            else:
                value = self.K - self.St

        self.value = value

    def build_tree_to_depth(self):
        '''
        Function to create the branches and leaves of the tree
        
        NOTES: if the class is inherited such that `get_tree_params()` is changed,
        this method need to change as well because it recursively build another
        tree
        '''
        
        
        if(self.depth == 0):
            
            self.value_option(self.path)
                

        elif(self.depth > 0):
            
            # Build upper children
            t1_path = self.path + [self.u]
            t1 = binomial_tree(self.name + "_u", t1_path, self.value, self.params, self.depth-1, [],
                              self.option_type, self.eu_am)
            t1.build_tree_to_depth()
            
            # Build lower children
            t2_path = self.path + [self.d]
            t2 = binomial_tree(self.name + "_d", t2_path, self.value, self.params, self.depth-1, [],
                              self.option_type, self.eu_am)
            t2.build_tree_to_depth()

            self.children = [t1, t2]
            children_value = (self.p * t1.value + (1-self.p) * t2.value) * tc.exp(-self.r * self.T)
            
            match self.eu_am:
                
                case "European":
                    self.value = children_value
                
                case "American":
                    multiplyer = tc.prod(tc.tensor(self.path))
                    if self.option_type == "call": current_value = self.S0 * multiplyer - self.K
                    elif self.option_type == "put": current_value = self.K - self.S0 * multiplyer
                    
                    if (current_value > children_value):
                        print("Early exercies at: " + str(self.name))
                    
                    self.value = max(current_value, children_value)
                    
            
        return
    
    
    def simulate(self, steps):
        direction = [self.u, self.d]
        simulated_path = [direction[x] for x in dist.Bernoulli(self.p).sample(tc.tensor([steps])).int().numpy()]
        
        multiplyer = tc.prod(tc.tensor(simulated_path))
        
        self.value_option(simulated_path)
        
        return simulated_path, self.S0*multiplyer, self.value
        
        
        
    def __str__(self):
        
        return str("tree") + str(":") + str(self.name)
        