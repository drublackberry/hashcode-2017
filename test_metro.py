#!/usr/bin/env python

import matplotlib
matplotlib.use("TkAgg")

import model
import simanneal
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import pickle


mod = model.SparseModel("me_at_the_zoo")



def do_temp(T):

    stats = simanneal.RunningStats()
    algo = simanneal.sim_anneal(mod, T0=T, cooling_rate=1, Q_callback=stats)

    S_0 = sp.csc_matrix(np.greater(mod.storage.toarray(), 0), dtype=np.int16)
    energy_t = []

    # Burn-in
    for i, (S, E) in enumerate(algo(S_0, 2000)):
        pass
    stats.reset()

    for i, (S, E) in enumerate(algo(S_0, 10000)):
        if i % 100 == 0:
            print(i, E)
        energy_t.append(E)

    with open('stats.pickle', 'wb') as fp:
        pickle.dump(stats, fp, pickle.HIGHEST_PROTOCOL)

    # vals, vecs = spla.eigs(P, k=1, which="LR")
    # Z = np.sum(vecs[:, 0] * np.exp(-E / T))

    return np.asarray(energy_t)


for T in [1000000]:
    print(T)
    plt.figure()
    E = do_temp(T)
    Z = np.mean(np.exp(E / T))
    plt.hist(E, bins=100)
    plt.savefig("mchist-%d.png" % T)
    plt.close()
    print("Z(%d) = %.5e" % (T, Z))
    print("Unique energies: %d" % len(np.unique(E)))
