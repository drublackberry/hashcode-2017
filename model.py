#!/usr/bin/env python

"""
Intentionally left blank.
"""
import iolib as iolib
import pandas as pd
import numpy as np

class Model:
    
    def __init__(self):
        self.storage = pd.DataFrame([])
        pass
    
    def load_scenario(self, scenario_name):
        ''' Returns the dataframes for computation
        '''
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
            L_D_ext[c] = self.L_D
        # matrix of deltas in latency
        dL = self.L - L_D_ext
        # compute J_cv
        J_cv_val = np.matmul(self.Rn.values, dL.values)
        J_cv = pd.DataFrame(index=self.Rn.columns, colums=self.L.columns, data=J_cv_val)
        return J_cv
    
    
    def compute_J_c(self, J_cv):
        ''' Computes the accummulated score for server c
        '''
        return J_cv.sum()        
    
    def rank_cache_server_by_J_c(self, J_c):
        ''' Ranks from high to low the servers according to score J_c
        '''
        return J_c.rank().index
    
    def store_video_in_cache_server(self, J_cv, c):
        ''' Ranks video according to J_cv , 
        questions the rules module to see if it fits
        updates the storage matrix
        returns v, e
        '''
        # Rank the videos on the given server by J_cv
        video_order = J_cv[c].rank().index
        for v in video_order:
            # Check if the video fits in the cache server
            if self.available_storage(c,v):
                # Store the video on the server
                self.storage.loc[v,c] = 1
            else:
                # server is full, remove the server from list and update Rn
                self.remove_server(c)
        pass
    
    def available_storage (self, c, v):
        ''' Checks if there is enough storage available on the server c
        for video v
        '''
        if (self.storage[c].sum() + v) <= self.C:
            return True
        else:
            return False
    
    def remove_server (self, c):
        ''' Removes a cache server from the available servers
        '''
        self.L.drop(c)
    
    def update_Rn (self, c, v):
        ''' Updates Rn by putting zero requestst in (v,e)
        '''
        for e in self.get_e_connected_to_c(c):
            self.Rn.loc[e,v] = np.nan
    
    def get_e_connected_to_c (self, c):
        ''' Returns a list of endpoints connected to the server c
        '''
        return self.L[c].dropna()
        
        
    


    
    
    
    
        