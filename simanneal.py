#!/usr/bin/env python

"""
Simulated annealing optimizer for the video caching problem.
"""

import numpy as np
import scipy.sparse as sp
import rules


def sim_anneal(mod, judge=None, cooling_step=0.1, T0=1.0, cooling_rate=0.9, Q_callback=None):
    """
    Construct basic SimAnneal instance using exponential cooling schedule.
    """
    S_shape = (mod.V, mod.C)
    S_size = mod.V * mod.C
    S_dtype = np.bool

    if judge is None:
        judge = rules.Judge(mod)

    fn_temp = lambda k: T0 * (cooling_rate ** np.int(k / cooling_rate))
    fn_energy = lambda S: -judge.score(S, ignore_overflow=False)
    fn_evo = Propagator(mod)

    return SimAnneal(fn_temp, fn_energy, fn_evo, Q_callback=Q_callback)


class Propagator(object):
    """
    Basic move class.
    """

    def __init__(self, mod):
        self.mod = mod
        wsum = (1 / self.mod.v_size).sum()
        wvid = (1 / self.mod.v_size / wsum).flatten()
        judge = rules.Judge(mod)
        pot_dL = (judge.Rn.T.dot(judge.dL)).toarray()
        pot_dL = sp.csc_matrix(np.greater(pot_dL, 0), dtype=np.int8)
        pot_dL.eliminate_zeros()
        W = pot_dL.multiply(wvid[:, None])
        W /= W.sum(axis=0)
        self.weights = np.asarray(W)

    def __call__(self, S):
        return self.neighbor(S)

    def prune(self, S):
        return sp.csc_matrix(S.multiply(np.greater(self.weights > 0)))

    def neighbor(self, S):
        found = False
        S_new = S.copy()
        while not found:
            c = np.random.choice(self.mod.C)
            v = np.random.choice(self.mod.V, p=self.weights[:, c])
            if S_new[v, c] == 1:
                S_new[v, c] = 0
                found = True
            else:
                if S[:, c].multiply(self.mod.v_size).sum() + self.mod.v_size[v] <= self.mod.X:
                    S_new[v, c] = 1
                    found = True
        return S_new


class Acceptance(object):
    """
    Boltzmann acceptance function with optional callback to
    log attempts (calculate Q/P matrix).
    """

    def __init__(self, callback=None):
        self.callback = callback

    def __call__(self, E, E_next, temp):
        dE = E_next - E
        p = np.random.rand()
        boltz = 0
        if temp > 0:
            boltz = np.exp(-dE / temp)

        if self.callback:
            self.callback(E, E_next)

        return (dE < 0) or (p < boltz)




class CanonicalEnsemble(object):

    def __init__(self, lumps):
        self.n = len(lumps) - 1
        self.lumps = lumps
        self.centroids = 0.5 * (lumps[:-1] + lumps[1:])
        self._Q = np.zeros((self.n, self.n))
        self.dE = self.centroids[:, None] - self.centroids

    def reset(self):
        self._Q = np.zeros((self.n, self.n))

    def find_lump(self, E):
        return np.argmin(abs(E - self.centroids))

    def __call__(self, E_0, E_1):
        i, j = self.find_lump(E_0), self.find_lump(E_1)
        self._Q[i, j] += 1

    def compute_P(self):
        P = self._Q.copy()
        P /= P.sum(axis=0)
        return P

    def compute_G(self, T):
        P = self.compute_P()
        P[np.isnan(P)] = 0
        G = P * np.exp(+self.dE / T)
        return G


class SimAnneal(object):


    def __init__(self, fn_temp, fn_energy, fn_evo, Q_callback=None):
        """
        fn_temp - callable [0, 1] -> T
        fn_energy - callable S -> E(S)
        fn_evo - callable yielding time evolution operator
        """

        self.fn_temp = fn_temp
        self.fn_energy = fn_energy
        self.fn_evo = fn_evo
        self.accept = Acceptance(Q_callback)

    def get_next_state(self, S, x):
        """
        S - current state (boolean array)
        """

        temp = self.fn_temp(x)
        E = self.fn_energy(S)

        T = self.fn_evo(temp)
        S_next = T(S)
        E_next = self.fn_energy(S_next)

        S_next_true = S
        E_next_true = E
        if self.accept(E, E_next, temp):
            E_next_true = E_next
            S_next_true = S_next

        return S_next_true, E_next_true


    def __call__(self, S_0, num_steps):
        return SimAnnealIterator(self, S_0, num_steps)



class SimAnnealIterator(object):

    def __init__(self, sim_anneal, S_0, num_steps):
        self.sim_anneal = sim_anneal
        self.num_steps = num_steps
        self.current_step = 0
        self.S = S_0

    def __iter__(self):
        return self

    def __next__(self):
        #x = self.current_step / float(self.num_steps)
        self.S, E = self.sim_anneal.get_next_state(self.S, self.current_step)
        #self.current_step += 1
        self.current_step += 1
        if self.current_step >= self.num_steps:
            raise StopIteration()
        return self.S, E
