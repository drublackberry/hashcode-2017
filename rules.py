#!/usr/bin/env python

"""
Scoring functions etc.
"""

import numpy as np
import scipy.sparse as sp


class Judge(object):

    def __init__(self, mod):
        #self.mod = mod
        dL = mod.L_D - mod.L
        dL[np.isnan(dL)] = 0

        # dL: (E x C)
        self.dL = sp.csc_matrix(dL)

        # Rn: (E x V)
        self.Rn = sp.csc_matrix(mod.Rn)
        self.total_reqs = int(self.Rn.sum())


        # v_size: V
        self.v_size = mod.v_size

        # dimensions for convenience
        self.E = mod.E
        self.X = mod.X
        self.C = mod.C
        self.V = mod.V


    def overflow(self, S, fill_rate=None):
        if fill_rate is None:
            fill_rate = (S.multiply(self.v_size)).sum(axis=0) / self.X

        return np.any(np.greater(np.asarray(fill_rate), 1))


    def score(self, S, ignore_overflow=False, fill_rate=None):
        """
        Return score for state S.
        """

        if not ignore_overflow:
            if self.overflow(S, fill_rate):
                return 0

        dL_tot = 0
        for e in range(self.E):
            #print(self.dL[e].shape, S.shape)
            dL_per_video = self.dL[e].multiply(S).toarray().max(axis=-1)
            #print(dL_per_video.shape)
            # print(type(self.Rn[e]))

            delta = float(self.Rn[e].dot(dL_per_video))
            dL_tot += delta

        s = dL_tot * 1000 / self.total_reqs
        return s
