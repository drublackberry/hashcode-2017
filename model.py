#!/usr/bin/env python

"""
Intentionally left blank.
"""
import iolib as iolib
import pandas as pd
import numpy as np
import time

class Model:
    
    def __init__(self):
        self.storage = pd.DataFrame([])
        pass
    
    def load_scenario(self, scenario_name):
        ''' Returns the dataframes for computation
        '''
        self.scenario_name = scenario_name
        dict_in = iolib.read_scenario(scenario_name)               
        self.L = dict_in['L']      
        self.L_D = dict_in['L_d']
        self.Rn = dict_in['R_n']
        self.X = dict_in['X']
        self.v_size = dict_in['v_size']
        self.storage = pd.DataFrame(0, index=self.Rn.columns, columns=self.L.columns)
        pass
    
    def cache_servers_available(self):
        '''L loses a column when server c is used
        '''
        return not(self.L.empty)
    
    def compute_J_cv(self):
        ''' Computes the score of server c and video v
        '''
        # extend L_D for all cache servers
        L_D_ext = pd.DataFrame(index=self.L.index, columns=self.L.columns)        
        for c in self.L.columns:            
            L_D_ext.loc[:,c] = self.L_D.values
        # matrix of deltas in latency
        dL = - self.L + L_D_ext                
        # compute J_cv        
        J_cv_val = np.matmul(self.Rn.transpose().fillna(0).values, dL.fillna(0).values)  
        J_cv = pd.DataFrame(index=self.Rn.columns, columns=self.L.columns, data=J_cv_val)
        return J_cv
    
    
    def compute_J_c(self, J_cv):
        ''' Computes the accummulated score for server c
        '''
        return J_cv.dropna().sum()        
    
    def sort_cache_server_by_J_c(self, J_c):
        ''' Ranks from high to low the servers according to score J_c
        '''        
        return J_c.sort_values(ascending=False).index
        
    
    def store_video_in_cache_server(self, J_cv, c):
        ''' Ranks video according to J_cv , 
        questions the rules module to see if it fits
        updates the storage matrix
        returns v, e
        '''
        # Rank the videos on the given server by J_cv
        video_order = J_cv[c].sort_values(ascending=False).index        
        for v in video_order:            
            # Check if the video fits in the cache server
            if self.available_storage(c,v):
                # Store the video on the                
                self.storage.loc[v,c] = self.v_size.loc[v].values                
            else:
                # server is full, remove the server from list and update Rn
                self.remove_server(c)
        pass
    
    def available_storage (self, c, v):
        ''' Checks if there is enough storage available on the server c
        for video v
        '''         
        if (self.storage[c].dropna().sum() + v) <= self.X:
            return True
        else:
            return False
    
    def remove_server (self, c):
        ''' Removes a cache server from the available servers
        '''
        #self.L.drop(c, axis=1, inplace=True)        
        self.L = self.L[[col for col in self.L.columns if col != c]]        
    
    def update_Rn (self, c, v):
        ''' Updates Rn by putting zero requestst in (v,e)
        '''
        for e in self.get_e_connected_to_c(c):
            self.Rn.loc[e,v] = np.nan
    
    def get_e_connected_to_c (self, c):
        ''' Returns a list of endpoints connected to the server c
        '''
        return self.L[c].dropna()     
        
    def write_storage(self, out_dir, c):
        out = iolib.OutputBuffer(self.scenario_name, out_dir)
        out.generate_output(self.storage)
        out.write_to_file(str(c)+'.txt')
        