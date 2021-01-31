import copy
import numpy as np 
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import networkx as nx
import string
import pandas as pd
import json
import random as rd
from scipy.stats import beta
from itertools import product

class DBN(ABC):
    @abstractmethod
    def __init__(self, initial, transition):
        pass
    
    
    @abstractmethod
    def step(self):
        pass

    
class FishBeliefNetwork(DBN):
    def __init__(self, n_agents, edge_p, planet, edges=None, structure=None):  
        self.t = 0
        self.n_agents = n_agents
        self.belief_history = []
        self.sample_history = []
        self.signal_history = []
        self.param_history = []
        self.true_prop_red = planet
        self.structure = structure 
            
        # Agent's prior is uniform, represented as a beta(1,1)
        # The first element of the list is A's belief .
        beliefs = np.array([[1,1] for _ in range(n_agents + 1)], dtype='float64')
        
        # Sample a fish for A 
        sampled_fish = np.random.binomial(1, self.true_prop_red)   
        beliefs[0] += update_beta_bernoulli(1, sampled_fish)
        beliefs[3] += update_beta_bernoulli(1, sampled_fish)
        self.sample_history.append({"agent":0,"step":self.t,"sample":sampled_fish})
        self.beliefs = beliefs
        self.belief_history.append(beliefs)
        
        
        if edges is not None:
            self.edges = edges
            print(edges)
        else:
            # Convention: A non-zero element A_ij indicates an edge from i to j.
            edges = np.zeros((n_agents,n_agents)) 

            # A is has incoming edges from all other nodes, 
            # since A is a special case and integrates information from all other agents.
            edges[:,0] = 1

            # Sample other edges.
            edges[1:n_agents,1:n_agents] = np.random.binomial(1, edge_p, (n_agents-1, n_agents-1))

            # Make sure that self-edges do not exist. 
            np.fill_diagonal(edges, 0)
            self.edges = edges
        
        # Check structure and determine t for sampling new fish. Influencer samples earlier. 
        if self.structure == 'b->c':
            sample_step_1 = 2
            sample_step_2 = 5
        elif self.structure == 'c->b':
            sample_step_1 = 5
            sample_step_2 = 2
        elif self.structure == 'indep':
            sample_step_1 = 2
            sample_step_2 = 2
        elif self.structure == 'b<->c':
            sample_step_1 = 2
            sample_step_2 = 2
        elif self.structure == 'b<->c_echo':
            sample_step_1 = 2
            sample_step_2 = 5
        self.sample_step_1 = sample_step_1
        self.sample_step_2 = sample_step_2
        
    def score_scale_fun(self, X, scores_old_min=0.0, scores_old_max= 0.4206951631858995, scores_new_min=0, scores_new_max=1):
        # TRANSFORMS STRENGTH SCALE INTO 0-1 SCALE 
        X = scores_new_max - ((scores_new_max - scores_new_min) * (scores_old_max - X) / (scores_old_max - scores_old_min))
        return X
        
    def step(self):
        # Beliefs at t are deterministic copies of the beliefs at t-1.
        new_beliefs = copy.deepcopy(self.beliefs)
        
        # Sample new fish based on new sample fish
        if self.t == self.sample_step_1:
            new_beliefs[1] += update_beta_bernoulli(1, 0)
            self.sample_history.append({"agent":1,"step":self.t,"sample":0})
            new_beliefs[3] += update_beta_bernoulli(1, 0)
            self.sample_history.append({"agent":1,"step":self.t,"sample":0})
        if self.t == self.sample_step_2: 
            new_beliefs[2] += update_beta_bernoulli(1, 1)
            self.sample_history.append({"agent":2,"step":self.t,"sample":1})
            new_beliefs[3] += update_beta_bernoulli(1, 1)
            self.sample_history.append({"agent":2,"step":self.t,"sample":1})
        
            
        signals = {}
        parameters = {}
        for i in range(self.n_agents + 1): 
            a = self.beliefs[i,0]
            b = self.beliefs[i,1]
            theta, _ = beta_mean_var(a,b) 
            # NEW: messing around with the first trials to change the beliefs shown to subjects and make things more difficult
            if self.t == 0:
                if theta >= 0.5: # first equal .5 -> red 
                    fish = 1
#                     stresngth = (theta-0.5)*2
                else:
                    fish = 0
#                     strength = np.abs((theta-0.5)*2) 
            elif self.t == 1:
                if theta > 0.5: # then only red if larger than .5 
                    fish = 1
#                     strength = (theta-0.5)*2
                else:
                    fish = 0
#                     strength = np.abs((theta-0.5)*2) 
            else:
                if theta >= 0.5:
                    fish = 1
#                     strength = (theta-0.5)*2
                else:
                    fish = 0
                    strength = np.abs((theta-0.5)*2) 
                    
# #              NEW CALCULATION OF STRENGTH BASED ON DENSITY BELOW OR ABOVE .5 
            betacdf = beta(a,b).cdf 
            if a > b: 
                strength = (betacdf(1) - betacdf(0.5) - .5) 
            elif b > a: 
                strength = (betacdf(0.5) - 0.5)  
            elif a == b:
                strength = (betacdf(1) - betacdf(0.5) - .5) 


            # creating three different discrete strenghts based on three bins for a scale from 0-1
            strength_discrete = 1
            if self.score_scale_fun(strength) >= 1/3 and self.score_scale_fun(strength) < 2/3:
                strength_discrete = 2
            elif self.score_scale_fun(strength) >= 2/3:
                strength_discrete = 3
            
            signals[i] = {"belief":fish,"strength":strength,"strength_discrete": strength_discrete}
            parameters[i] = {'a': a, 'b': b}
            
            for j in range(self.n_agents + 1):
                print(self.edges)
                print(i,j)
                if i == 3 or j == 3:
                    continue
                if self.edges[i,j] != 1:
                    continue
                else:
                    new_beliefs[j][int(np.abs(fish-1))] += strength 
                   
                    
        self.beliefs = new_beliefs
        self.belief_history.append(new_beliefs)
        self.signal_history.append(signals)
        self.param_history.append(parameters)
        self.t += 1   
        
        
        
class FishBeliefNetworkNaive(DBN):
    def __init__(self, n_agents, edge_p, planet=None, edges=None, structure=None):  
        self.t = 0
        self.n_agents = n_agents
        self.belief_history = []
        self.sample_history = []
        self.signal_history = []
        self.param_history = []
        self.structure = structure 
        
        if planet is not None:
            self.true_prop_red = planet
        else:
            if np.random.binomial(1,0.5) == 1:
                self.true_prop_red = 1/4.
            else:
                self.true_prop_red = 3/4.
                
        # Agent's prior is uniform, represented as a beta(1,1)
        # The first element of the list is A's belief .
        beliefs = np.array([[1,1] for _ in range(n_agents + 1)], dtype='float64')
        
        # Sample a fish for A 
        sampled_fish = np.random.binomial(1, self.true_prop_red)   
        beliefs[0] += update_beta_bernoulli(1, sampled_fish)
        self.sample_history.append({"agent":0,"step":self.t,"sample":sampled_fish})
        self.beliefs = beliefs
        self.belief_history.append(beliefs)
        
        
        if edges is not None:
            self.edges = edges
        else:
            # Convention: A non-zero element A_ij indicates an edge from i to j.
            edges = np.zeros((n_agents,n_agents)) 

            # A is has incoming edges from all other nodes, 
            # since A is a special case and integrates information from all other agents.
            edges[:,0] = 1

            # Sample other edges.
            edges[1:n_agents,1:n_agents] = np.random.binomial(1, edge_p, (n_agents-1, n_agents-1))

            # Make sure that self-edges do not exist. 
            np.fill_diagonal(edges, 0)
            self.edges = edges
            
        # Sample time when B and C sample fish
        # Note: Max number of steps hard-coded here
        self.sample_step_1 = rd.randint(1,10)
        self.sample_step_2 = rd.randint(1,10)
        
    def score_scale_fun(self, X, scores_old_min = 0.0, scores_old_max = 0.4206951631858995, scores_new_min=0, scores_new_max=1):
        # TRANSFORMS STRENGTH SCALE INTO 0-1 SCALE 
        X = scores_new_max - ((scores_new_max - scores_new_min) * (scores_old_max - X) / (scores_old_max - scores_old_min))
        return X
        
    def step(self, betacdf):
        # Beliefs at t are deterministic copies of the beliefs at t-1.
        new_beliefs = copy.deepcopy(self.beliefs)
        
        # Sample new fish based on new sample fish
        if self.t == self.sample_step_1:
            sampled_fish = np.random.binomial(1, 0.5)   
            new_beliefs[1] += update_beta_bernoulli(1, sampled_fish)
            new_beliefs[3] += update_beta_bernoulli(1, sampled_fish)
            self.sample_history.append({"agent":1,"step":self.t,"sample":sampled_fish})
<<<<<<< HEAD
            if self.t == self.sample_step_2: 
                sampled_fish = np.random.binomial(1, self.true_prop_red)   
                new_beliefs[2] += update_beta_bernoulli(1, sampled_fish)
                new_beliefs[3] += update_beta_bernoulli(1, sampled_fish)
                self.sample_history.append({"agent":2,"step":self.t,"sample":sampled_fish})
        
        elif self.t == self.sample_step_2:
            sampled_fish = np.random.binomial(1, self.true_prop_red)   
=======
        if self.t == self.sample_step_2: 
            sampled_fish = np.random.binomial(1, 0.5)   
>>>>>>> 323bbde39e997b4fe919bcc3958863283e0f8036
            new_beliefs[2] += update_beta_bernoulli(1, sampled_fish)
            new_beliefs[3] += update_beta_bernoulli(1, sampled_fish)
            self.sample_history.append({"agent":2,"step":self.t,"sample":sampled_fish})
            
        signals = {}
        parameters = {}
        for i in range(self.n_agents): 
            a = self.beliefs[i,0]
            b = self.beliefs[i,1]
            if a > b: 
                fish = 1
                strength = (1 - betacdf.get_cdf(a,b) - .5) 
            elif b > a: 
                fish = 0
                strength = (betacdf.get_cdf(a,b) - 0.5) 
            elif a == b:
                fish = np.random.binomial(1,0.5)
                strength = (1 - betacdf.get_cdf(a,b) - .5)  

                
            # creating three different discrete strenghts based on three bins for a scale from 0-1
            strength_discrete = 1
            if self.score_scale_fun(strength) >= 1/3 and self.score_scale_fun(strength) < 2/3:
                strength_discrete = 2
            elif self.score_scale_fun(strength) >= 2/3:
                strength_discrete = 3
            
            
            signals[i] = {"belief":fish,"strength":strength,"strength_discrete": strength_discrete}
            parameters[i] = {'a': a, 'b': b}
                        
            
            for j in range(self.n_agents):
                if self.edges[i,j] != 1:
                    continue
                else:
                    new_beliefs[j][int(np.abs(fish-1))] += strength 
                   
                    
        self.beliefs = new_beliefs
        self.belief_history.append(new_beliefs)
        self.signal_history.append(signals)
        self.param_history.append(parameters)
        self.t += 1   
        
        

class BetaCDF():
    def __init__(self):  
        self.values = {}
    def get_cdf(self, a,b):
        entry = (a,b)
        if entry in self.values:
            result = self.values[entry]
        else: 
            result = beta(a,b).cdf(0.5) 
            self.values[entry] = result
        return result
            
    
def update_beta_bernoulli(n, hits, mult_factor=1):
    n = n * mult_factor
    hits = hits * mult_factor
    return (hits, (n-hits))

        

def beta_mean_var(a,b):
    mean = a / (a+b)
    var = (a*b)/((a+b)**2 * (a+b+1))
    return mean, var

# STRUCTURES FOR NAIVE LEARNER 
structures = {"indep": np.array([[0., 0., 0.],
                                       [1., 0., 0.],
                                       [1., 0., 0.]]),
              "b->c": np.array([[0., 0., 0.],
                                         [1., 0., 1.],
                                         [1., 0., 0.]]),
              "c->b": np.array([[0., 0., 0.],
                                         [1., 0., 0.],
                                         [1., 1., 0.]]),
              "b<->c": np.array([[0., 0., 0.],
                                         [1., 0., 1.],
                                         [1., 1., 0.]]) 
    
}

# STRUCTURES FOR ``LISTENING TO INFLUENCED AGENT ONLY''
structures_influenced = {"indep": np.array([[0., 0., 0.],
                                       [1., 0., 0.],
                                       [1., 0., 0.]]),
              "b->c": np.array([[0., 0., 0.],
                                         [0., 0., 1.],
                                         [1., 0., 0.]]),
              "c->b": np.array([[0., 0., 0.],
                                         [1., 0., 0.],
                                         [0., 1., 0.]]),
              "b<->c": np.array([[0., 0., 0.],
                                         [1., 0., 1.],
                                         [1., 1., 0.]]) 
    
}

# STRUCTURES FOR ``LISTENING TO INFLUENCER ONLY''
structures_influencer = {"indep": np.array([[0., 0., 0.],
                                       [1., 0., 0.],
                                       [1., 0., 0.]]),
              "b->c": np.array([[0., 0., 0.],
                                         [1., 0., 1.],
                                         [0., 0., 0.]]),
              "c->b": np.array([[0., 0., 0.],
                                         [0., 0., 0.],
                                         [1., 1., 0.]]),
              "b<->c": np.array([[0., 0., 0.],
                                         [1., 0., 1.],
                                         [1., 1., 0.]]) 
    
}

def structure_names(array):
    if np.array_equal(array, np.array([[0., 0., 0.],
                                       [1., 0., 0.],
                                       [1., 0., 0.]])):
        struc = "indep"
        
    elif np.array_equal(array, np.array([[0., 0., 0.],
                                         [1., 0., 1.],
                                         [1., 0., 0.]])):
        struc = "b->c"
        
    elif np.array_equal(array, np.array([[0., 0., 0.],
                                         [1., 0., 0.],
                                         [1., 1., 0.]])):
        struc = "c->b"
        
    elif np.array_equal(array, np.array([[0., 0., 0.],
                                         [1., 0., 1.],
                                         [1., 1., 0.]])):
        struc = "b<->c"
        
    else:
        raise ValueError
    return struc

