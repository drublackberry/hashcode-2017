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
import progressbar as pb
import rules
import os


mod = model.SparseModel("me_at_the_zoo")
judge = rules.Judge(mod)

N = 100
V = 3e-4
T0 = 12000
STEPS = 5000


if not os.path.exists("S0.pickle"):

    S_0 = sp.csc_matrix(np.greater(mod.storage.toarray(), 0), dtype=np.int16)

    energy_t = []

    # Burn-in
    metro = simanneal.sim_anneal(mod, T0=np.inf, cooling_rate=1)

    print("Burn in...")
    for i, (S, E) in enumerate(metro(S_0, 1000)):
        pass

    print("Setting up ensemble...")
    for i, (S, E) in enumerate(metro(S, 5000)):
        energy_t.append(E)

    with open("S0.pickle", 'wb') as fp:
        pickle.dump(S, fp)
    with open("E.pickle", 'wb') as fp:
        pickle.dump(energy_t, fp)

else:
    with open("S0.pickle", 'rb') as fp:
        S = pickle.load(fp)
    with open("E.pickle", 'rb') as fp:
        energy_t = pickle.load(fp)

stats = simanneal.Canonical(N)
stats.setup_macrostates(energy_t)

prop = simanneal.Propagator(mod)
prop.dist = 30
schedule = simanneal.EntropyDescentScheduler(V, T0, 0.0001, stats, 500)
algo = simanneal.SimAnneal(schedule,
                           lambda S: -judge.score(S),
                           prop,
                           Q_callback=stats)

energy_t = []
temps = []
for i, (S, E) in enumerate(algo(S, STEPS)):
    if i % 10 == 0 and i > 0:
        props = stats.get_equilibrium_properties(schedule.T[-1])
        print(i, E, schedule.T[-1], schedule.rate, props["expected_energy"], props["entropy"])
    energy_t.append(E)
    temps.append(schedule.T[-1])

temps = np.asarray(temps)
energy_t = np.asarray(energy_t)

np.save("T.npy", temps)
np.save("E.npy", energy_t)


plt.plot(np.arange(len(temps)), temps)
plt.grid()
plt.savefig("cooling.png")
