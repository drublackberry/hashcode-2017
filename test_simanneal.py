#!/usr/bin/env python

import numpy as np
import model
import simanneal
import rules
import iolib
import argparse
import zipfile
import os
import glob
import tempfile
import scipy.sparse as sp


STEPS = 100000
K_B = 1
K_FR = 0
N_FR = 0
CSTEP = 0.001
T0 = 50000
DAMP = 5 / T0
CRATE = 0.9

def main(args, outpath):
    mod = model.SparseModel(args.scenario)
    print("Model loaded.")
    judge = rules.Judge(mod)

    algo = simanneal.sim_anneal(mod, K_B, K_FR, N_FR, DAMP, judge=judge,
                                allow_zero=False, T0=T0, cooling_step=CSTEP,
                                cooling_rate=CRATE)

    #S_0 = np.greater(mod.storage.toarray(), 0)

    best = 0
    for i in range(5):
        S_ = np.random.rand(mod.V, mod.C) < 1e-1
        S_ = sp.csc_matrix(S_)
        S_ = algo.fn_invalid(S_)
        score = judge.score(S_)
        if score > best:
            print(score)
            best = score
            S_0 = S_



    buf = iolib.OutputBuffer(args.scenario, outpath)
    print("START")
    for i, (S, E) in enumerate(algo(S_0, STEPS)):
        if i % 1 == 0:
            print(i, "E=%.3e" % E, "T=%.3f" % algo.fn_temp(i / STEPS))
        if i % 1000 == 0:
            buf.generate_output(S)
            buf.write_to_file("-%05d" % (i / 1000))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=int)
    parser.add_argument("scenario", type=str)
    args = parser.parse_args()

    outpath = os.path.join(".", "outputs", "%03d" % args.run_id)
    print("Preparing output directory %s" % outpath)

    if not os.path.exists(os.path.join(".", "outputs")):
        os.mkdir(os.path.join(".", "outputs"))


    if os.path.exists(outpath):
        bak = tempfile.mktemp(prefix="bak-%03d_" % args.run_id, dir=os.path.join(".", "outputs"))
        os.rename(outpath, bak)
    os.mkdir(outpath)

    sources = glob.glob("*.py")
    zip_dst = os.path.join(outpath, "sources.zip")

    print("Writing %s to %s" % (sources, zip_dst))
    with zipfile.ZipFile(zip_dst, 'w') as zip:
        for filename in sources:
            zip.write(filename)
    print("Setup done, launching main function.")

    main(args, outpath)
