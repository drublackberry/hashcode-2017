#!/usr/bin/env python

"""
Simulated annealing optimizer for the video caching problem.
"""

import numpy as np
import scipy.sparse as sp
import rules


def sim_anneal(mod, k_B=1.0, k_fill_rate=1000, n_fill_rate=2, evo_damp=0.1, judge=None,
               allow_zero=True, cooling_step=0.1, T0=1.0, cooling_rate=0.9):
    S_shape = (mod.V, mod.C)
    S_size = mod.V * mod.C
    S_dtype = np.bool

    if judge is None:
        judge = rules.Judge(mod)

    def fn_temp(x):
        t = np.int(x / cooling_step)
        return T0 * (cooling_rate ** t)

    def fn_energy(S):
        fill_rate = (S.multiply(mod.v_size)).sum(axis=0) / mod.X

        score = judge.score(S, ignore_overflow=False, fill_rate=fill_rate)
        #return - score + k_fill_rate * (fill_rate ** (n_fill_rate)).sum()
        return -score

    def fn_invalid(S):
        print("Repair solution...")
        while True:
            F = np.asarray(S.multiply(mod.v_size) / mod.X)
            #print(F.shape)
            fill_rate = np.sum(F, axis=0)
            #print(fill_rate.shape)
            full = np.greater(fill_rate, 1)
            #print(full)
            #exit(1)
            if not np.any(full):
                print("DONE")
                return S
            caches = np.argwhere(full)
            #print(len(caches))
            for c in caches:
                i = F[:, c].argmax()
                S[i, c] = False

    w = (mod.v_size ** 2).sum()
    weights = mod.v_size ** 2/ w
    print(weights)
    def fn_evo(temp):
        T_ = (weights * np.random.rand(*S_shape) < evo_damp * temp / w)
        #return lambda S: np.logical_xor(T_, S)
        return lambda S: sp.csc_matrix(np.logical_xor(T_, S.toarray()))

    return SimAnneal(fn_temp, fn_energy, fn_evo, k=k_B, allow_zero=allow_zero, fn_invalid=fn_invalid)


class SimAnneal(object):


    def __init__(self, fn_temp, fn_energy, fn_evo, k=1.0, allow_zero=True,
                 fn_invalid=None):
        """
        fn_temp - callable [0, 1] -> T
        fn_energy - callable S -> E(S)
        fn_evo - callable yielding time evolution operator
        k - Boltzmann constant analog (default 1.0)
        """

        self.fn_temp = fn_temp
        self.fn_energy = fn_energy
        self.fn_evo = fn_evo
        self.k = k
        self.allow_zero = allow_zero
        self.fn_invalid = fn_invalid


    def get_next_state(self, S, x):
        """
        S - current state (boolean array)
        """

        temp = self.fn_temp(x)
        E = self.fn_energy(S)

        T = self.fn_evo(temp)
        S_next = T(S)
        E_next = self.fn_energy(S_next)

        # If we're already non-compliant, accept only descent
        if E >= 0:
            dE = E_next - E
            while dE > 0:
                T = self.fn_evo(temp)
                S_next = T(S)
                E_next = self.fn_energy(S_next)
                dE = E_next - E

        if not self.allow_zero:
            if self.fn_invalid is not None:
                if E_next >= 0:
                    S_next = self.fn_invalid(S_next)
                    E_next = self.fn_energy(S_next)
            else:
                while E_next >= 0:
                    T = self.fn_evo(temp)
                    S_next = T(S)
                    E_next = self.fn_energy(S_next)

        S_next_true = S
        E_next_true = E
        dE = E_next - E


        p = np.random.rand()
        if (dE < 0) or (p > np.exp(-dE / (self.k * temp))):
            S_next_true = S_next
            E_next_true = E_next

        #print(type(S_next_true))
        #print(S_next_true.shape)

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
        x = self.current_step / float(self.num_steps)
        self.S, E = self.sim_anneal.get_next_state(self.S, x)
        self.current_step += 1
        if self.current_step >= self.num_steps:
            raise StopIteration()
        return self.S, E
