#!/usr/bin/env python

"""
Intentionally left blank.
"""
import iolib as iolib
import pandas as pd
import numpy as np
import time
import scipy.sparse as sp


class ModelInterface(object):


    def __init__(self, *args, **kw):
        raise NotImplementedError()

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
        self.C = dict_in['C']
        self.V = dict_in['V']
        self.E = dict_in['E']



class SparseModel(ModelInterface):

    def __init__(self, scenario):
        self.load_scenario(scenario)

        # forget pandas stuff
        for x in ['L', 'L_D', 'Rn', 'v_size']:
            a = getattr(self, x)
            if hasattr(a, 'columns'):
                setattr(self, x, a.as_matrix())

        self.Rn_T = sp.csc_matrix(self.Rn.T)
        #print(self.Rn_T.shape, self.V, self.E)
        #exit(1)
        self.S = sp.csc_matrix(np.zeros((self.V, self.C), dtype=np.int32))
        #print(self.S)
        self.caches = np.arange(self.C)
        self.videos = np.arange(self.V)

    def cache_servers_available(self):
        return len(self.caches)

    def compute_J_cv(self):
        dL = self.L_D - self.L
        dL[np.isnan(dL)] = 0
        #print(dL)
        dL = sp.csc_matrix(dL)
        J_cv = sp.csc_matrix(self.Rn_T.dot(dL))
        return J_cv

    def compute_J_c(self, J_cv):
        return np.asarray(J_cv.sum(axis=0)).reshape(-1)

    def sort_cache_server_by_J_c(self, J_c):
        ind = np.argsort(J_c)[::-1]
        #print(ind, self.caches[ind], J_c[ind])
        return self.caches[ind]

    def store_video_in_cache_server(self, J_cv, c):
        ''' Ranks video according to J_cv ,
        questions the rules module to see if it fits
        updates the storage matrix
        returns v, e
        '''
        # Rank the videos on the given server by J_cv
        ic = np.argwhere(self.caches == c)
        try:
            ic = ic[0][0]
        except:
            raise StopIteration()

        #print(J_cv.shape)
        ind = np.argsort(np.asarray(J_cv[:, ic].todense()).reshape(-1))[::-1]
        ind = [i for i in ind if J_cv[i, ic] > 0]
        video_order = self.videos[ind]

        # for v in video_order:
        #     # Check if the video fits in the cache server
        #     if self.available_storage(c, v):
        #         # Store the video on the
        #         self.S[v, c] = self.v_size[v]
        #     else:
        #         # server is full, remove the server from list and update Rn
        #         self.remove_server(c)
        #         if self.cache_servers_available():
        #             self.update_Rn(c, v)

        space = self.X - self.S[:, c].sum()
        full = False
        for v in video_order:
            if space >= self.v_size[v]:
                self.S[v, c] = self.v_size[v]
                for e in self.get_e_connected_to_c(c):
                    self.Rn_T[v, e] = 0
                break
            full = True

        if full or len(video_order) == 0:
            self.remove_server(c)
            raise StopIteration()


    def available_storage(self, c, v):
        #print(self.S.shape)
        not_full = (self.S[:, c].sum() + self.v_size[v] <= self.X)
        has_all_vids = (self.S[:, c].nnz < self.V)
        return (not_full and not(has_all_vids))

    def remove_server (self, c):
        ''' Removes a cache server from the available servers
        '''
        #self.L = self.L[[col for col in self.L.columns if col != c]]
        print("KILL: ", c)
        ic = np.argwhere(self.caches == c)
        #print(ic, c)
        #print("ic", ic)
        try:
            ic = ic[0][0]
            self.L = np.delete(self.L, ic, axis=1)
            self.caches = np.delete(self.caches, ic)
        except:
            self.L = []
            self.caches = []

    def update_Rn (self, c, v):
        ''' Updates Rn by putting zero requestst in (v,e)
        '''
        for e in self.get_e_connected_to_c(c):
            self.Rn_T[v, e] = 0

    def get_e_connected_to_c (self, c):
        ''' Returns a list of endpoints connected to the server c
        '''
        ic = np.argwhere(self.caches == c)
        if len(ic):
            ic = ic[0][0]
        else:
            return []
        Lc = self.L[:, ic]
        return np.arange(Lc.shape[0])[np.isfinite(Lc)]

    def write_storage(self, out_dir, i):
        out = iolib.OutputBuffer(self.scenario_name, out_dir)
        out.generate_output(self.S.todense())
        out.write_to_file('-%04d' % i)






class DenseModel(ModelInterface):

    def __init__(self, scenario):
        self.load_scenario(scenario)
        self.storage = pd.DataFrame(0, index=self.Rn.columns, columns=self.L.columns)


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
        if (self.storage[c].dropna().sum() + self.v_size.loc[v].values) <= self.X:
            return True
        else:
            return False

    def remove_server (self, c):
        ''' Removes a cache server from the available servers
        '''
        self.L = self.L[[col for col in self.L.columns if col != c]]

    def update_Rn (self, c, v):
        ''' Updates Rn by putting zero requestst in (v,e)
        '''
        for e in self.get_e_connected_to_c(c):
            self.Rn.loc[e,v] = np.nan

    def get_e_connected_to_c (self, c):
        ''' Returns a list of endpoints connected to the server c
        '''
        return self.L[c].dropna().index

    def write_storage(self, out_dir, i):
        out = iolib.OutputBuffer(self.scenario_name, out_dir)
        out.generate_output(self.storage)
        out.write_to_file('-%04d' % i)
