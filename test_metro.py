#!/usr/bin/env python

import matplotlib
matplotlib.use("TkAgg")

import model
import simanneal
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


mod = model.SparseModel("me_at_the_zoo")



def do_temp(T):

    lumps = -np.linspace(0, 500000, 1001)
    print(lumps)
    stats = simanneal.CanonicalEnsemble(lumps)

    algo = simanneal.sim_anneal(mod, T0=T, cooling_rate=1, Q_callback=stats)

    S_0 = sp.csc_matrix(np.greater(mod.storage.toarray(), 0), dtype=np.int16)

    energy_t = []

    for i, (S, E) in enumerate(algo(S_0, 2000)):
        pass

    stats.reset()
    for i, (S, E) in enumerate(algo(S_0, 10000)):
        if i % 1000 == 0:
            print(i)
        energy_t.append(E)

    np.save("G.npy", stats.compute_G(T))
    return np.asarray(energy_t)


for T in [100000]:
    print(T)
    plt.figure()
    E = do_temp(T)
    Z = np.mean(np.exp(E / T))
    plt.hist(E, bins=100)
    plt.savefig("mchist-%d.png" % T)
    plt.close()
    print("Z(%d) = %.5e" % (T, Z))
    print("Unique energies: %d" % len(np.unique(E)))
