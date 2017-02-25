#!/usr/bin/env python

import numpy as np
import model
import simanneal
import rules
import iolib

SCENARIO = "me_at_the_zoo"
STEPS = 10000
K_B = 1
K_FR = 1
N_FR = 2
DAMP = 1


def main():
    mod = model.SparseModel(SCENARIO)
    algo = simanneal.sim_anneal(mod, K_B, K_FR, N_FR, DAMP)

    S_0 = np.greater(mod.storage.toarray(), 0)
    #print(S_0)
    buf = iolib.OutputBuffer(SCENARIO, "foo")
    for i, (S, E) in enumerate(algo(S_0, STEPS)):
        if i % 100 == 0:
            print(i, "%.3e" % E)
            buf.generate_output(S)
            buf.write_to_file("-%05d" % i)

if __name__ == "__main__":
    main()
