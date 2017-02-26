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
        self.dL = sp.csr_matrix(dL)

        # Rn: (E x V)
        self.Rn = sp.csc_matrix(mod.Rn, dtype=np.uint32)
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
                print("THIS SHOULD NOT HAPPEN")
                return 0

        mx = np.empty((self.E, self.V), np.float64)

        dL = self.dL
        for e, dL_row in enumerate(dL):
            mx[e, :] = dL_row.multiply(S).max(axis=-1).toarray()[:, 0]
        dL_tot = self.Rn.multiply(mx).sum()

        return dL_tot * 1000 / self.total_reqs
