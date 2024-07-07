"""
Probabilities of Causation
"""

# Author: Leo Guelman <leo.guelman@gmail.com>


import numpy as np
import pandas as pd
import seaborn as sns
import sys
from sklearn import linear_model
import pickle
import plotly.express as px
from plotly.offline import plot
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings 
import os


class PnsSim:
    """
    Probabilities of Causation: Adequate Size of Experimental and Observational Samples
    (Ang Li et al.)
     
    Parameters
    ----------
    B : Number of runs. 
    N_o : Number of observational samples per run 
    N_e : Number of experimental samples per run 
    N_i : (Large) number of observational and experimental sampes to compute true values.
    Z_dim : Dimensionality of confounders. 
    seed : Random seed.              
    """

    def __init__(self, B: int = 2, N_o:int = 100, N_e:int = 100, N_i = 1000, Z_dim: int = 20, seed: int = 42):
                
        np.random.seed(seed)
        
        self.p = np.random.uniform(size=(Z_dim+2,B))
        self.M_X_coef = np.random.uniform(-1, 1, size=(Z_dim, B))
        self.M_Y_coef = np.random.uniform(-1, 1, size=(Z_dim, B))
        self.C = np.random.uniform(-1, 1, size=(1, B))
        
        self.B = B 
        self.N_o = N_o
        self.N_e = N_e
        self.N_i = N_i
        self.Z_dim = Z_dim 
        self.seed = seed

    #def __repr__(self):
    #        
    #        items = ("%s = %r" % (k, v) for k, v in self.__dict__.items())
    #        return "<%s: {%s}>" % (self.__class__.__name__, ', '.join(items))
    #       pass
    
    def _generate_samples(self, N=None, experimental=None):
           
        data = []
        
        for b in range(self.B):
            Z = []
            for _ in range(N):
                Z.append(np.random.binomial(1, p=self.p[0:self.Z_dim, b], size=self.Z_dim))
            Z = np.array(Z)

            M_X = np.dot(Z, self.M_X_coef[:,b])
            M_Y = np.dot(Z, self.M_Y_coef[:,b])
    
            U_X = np.random.binomial(1, self.p[self.Z_dim, b], size=N)
            U_Y = np.random.binomial(1, self.p[self.Z_dim+1, b], size=N)
        
            if experimental:
                X = np.random.binomial(1, 0.5, size=N)
            else: 
                X = 1 * ((M_X + U_X) > 0.5)
            Y_ = self.C[:,b] * X + M_Y + U_Y
            Y = 1 * (np.logical_or(np.logical_and(Y_ > 0, Y_ < 1), np.logical_and(Y_ > 1, Y_ < 2)))
            
            Y = 1 * (np.logical_or(np.logical_and(Y_ > 0, Y_ < 1), np.logical_and(Y_ > 1, Y_ < 2)))
            
            Y_1 = self.C[:,b] * 1 + M_Y + U_Y
            Y_0 = self.C[:,b] * 0 + M_Y + U_Y
            Y1 = 1 * (np.logical_or(np.logical_and(Y_1 > 0, Y_1 < 1), np.logical_and(Y_1 > 1, Y_1 < 2)))
            Y0 = 1 * (np.logical_or(np.logical_and(Y_0 > 0, Y_0 < 1), np.logical_and(Y_0 > 1, Y_0 < 2)))
           
            
            data_dict = {
                'Z':Z,
                'X':X,
                'Y':Y,
                'Y1': Y1,
                'Y0': Y0,
                'experimental': experimental,
                }
        
            data.append(data_dict)
            
        return data
    
    def generate_samples(self):
       
        self.train_obs_data = self._generate_samples(N= self.N_o, experimental = False)         
        self.train_exp_data = self._generate_samples(N= self.N_e, experimental = True)      
       
        return self 
                    
    def fit(self, obs_data=None, exp_data=None):
        
        ATE = []
        true_P_benefit = []
        true_P_harm = []
        
        dist_check_pass = []
        PNS_l = []
        PNS_u = []
        
        if obs_data is None and exp_data is None:
            
            obs_data = self.train_obs_data           
            exp_data = self.train_exp_data
            
        for b in range(self.B):
            
            # Experimental distributions
            phat_y1_do_x1 = sum(np.logical_and(exp_data[b]['X'] == 1, exp_data[b]['Y'] ==1)) / sum(exp_data[b]['X'] == 1)
            phat_y1_do_x0 = sum(np.logical_and(exp_data[b]['X'] == 0, exp_data[b]['Y'] ==1)) / sum(exp_data[b]['X'] == 0)
            phat_y0_do_x0 = sum(np.logical_and(exp_data[b]['X'] == 0, exp_data[b]['Y'] ==0)) / sum(exp_data[b]['X'] == 0)

            # Observational distributions
            phat_x1y1 = sum(np.logical_and(obs_data[b]['X'] == 1, obs_data[b]['Y'] ==1)) / obs_data[b]['Y'].shape[0]
            phat_x1y0 = sum(np.logical_and(obs_data[b]['X'] == 1, obs_data[b]['Y'] ==0)) / obs_data[b]['Y'].shape[0]
            phat_x0y1 = sum(np.logical_and(obs_data[b]['X'] == 0, obs_data[b]['Y'] ==1)) / obs_data[b]['Y'].shape[0]
            phat_x0y0 = sum(np.logical_and(obs_data[b]['X'] == 0, obs_data[b]['Y'] ==0)) / obs_data[b]['Y'].shape[0]
            phat_y1 = sum(obs_data[b]['Y'] ==1) / obs_data[b]['Y'].shape[0]
            
            # ATE , P(Benefit) and P(Harm)
            ATE_ = np.mean(exp_data[b]['Y'][exp_data[b]['X'] == 1]) - np.mean(exp_data[b]['Y'][exp_data[b]['X'] == 0])
            true_P_benefit_ = (sum((obs_data[b]['Y1'] - obs_data[b]['Y0']) > 0) + 
                              sum((exp_data[b]['Y1'] - exp_data[b]['Y0']) > 0)) / (len(exp_data[0]['Y'])+ len(obs_data[0]['Y']))
            true_P_harm_ = (sum((obs_data[b]['Y1'] - obs_data[b]['Y0']) < 0) + 
                              sum((exp_data[b]['Y1'] - exp_data[b]['Y0']) < 0)) / (len(exp_data[0]['Y'])+ len(obs_data[0]['Y']))
            
            #Check violations between experimental and observational distributions 
            #P(x,y) ⩽ P(yₓ) ⩽ 1 - P(x,y'),
            #P(x',y) ⩽ P(yₓ') ⩽ 1 - P(x',y'),
            
            ATE.append(ATE_)
            true_P_benefit.append(true_P_benefit_)
            true_P_harm.append(true_P_harm_)
            
            c1 = phat_x1y1 <= phat_y1_do_x1 <= 1- phat_x1y0
            c2 = phat_x0y1 <= phat_y1_do_x0 <= 1 - phat_x0y0
            
            dist_check_pass.append(
                np.logical_and(c1, c2)
                )
            
            PNS_l.append(
                max(
                0, 
                phat_y1_do_x1 - phat_y1_do_x0,
                phat_y1 - phat_y1_do_x0,
                phat_y1_do_x1 - phat_y1
                )
            )

            PNS_u.append(
                min(
                phat_y1_do_x1,
                phat_y0_do_x0,
                phat_x1y1 + phat_x0y0,
                phat_y1_do_x1 - phat_y1_do_x0 + phat_x1y0  + phat_x0y1 
                )
            )
              
        return PNS_l, PNS_u, dist_check_pass, ATE, true_P_benefit, true_P_harm
    
    def get_informer_data(self):
        
        obs_data = self._generate_samples(N = self.N_i, experimental = False)         
        exp_data = self._generate_samples(N = self.N_i, experimental = True)   
        true_PNS_l, true_PNS_u, true_dist_check_pass, ATE, true_P_benefit, true_P_harm = self.fit(obs_data=obs_data, exp_data=exp_data)
          
        return true_PNS_l, true_PNS_u, true_dist_check_pass, ATE, true_P_benefit, true_P_harm
        

