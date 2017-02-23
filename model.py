#!/usr/bin/env python

"""
Intentionally left blank.
"""
import iolib as iolib
import pandas as pd

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
        pass
    
    
    def compute_J_c(self, J_cv):
        ''' Computes the accummulated score for server c
        '''
        pass
    
    def rank_cache_server_by_J_c(self, J_c):
        ''' Ranks from high to low the servers according to score J_c
        '''
        pass
    
    def store_video_in_cache_server(self, J_cv, c):
        ''' Ranks video according to J_cv , 
        questions the rules module to see if it fits
        updates the storage matrix
        returns v, e
        '''
        pass
    
    def update_Rn (self, v, e):
        ''' Updates Rn by putting zero requestst in (v,e)
        '''
        pass
    
    def update_L (self, c):
        ''' Updates Lec by removing the column c
        '''
        pass
        
    


    
    
    
    
        