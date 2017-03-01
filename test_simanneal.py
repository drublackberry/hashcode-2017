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

STEPS = 1000
CSTEP = 100
T0 = 20000
CRATE = 0.8


#IG_FILE = "./outputs/017/trending_today-00099.out"


def main(args, outpath):
    mod = model.SparseModel(args.scenario)
    print("Model loaded.")
    judge = rules.Judge(mod)

    algo = simanneal.sim_anneal(mod, judge=judge, T0=T0,
                                cooling_step=CSTEP,
                                cooling_rate=CRATE)
    algo.fn_evo.dist = 10
    S_0 = sp.csc_matrix(np.greater(mod.storage.toarray(), 0), dtype=np.int16)

    # best = 0
    # for i in range(10):
    #     S_ = np.random.rand(mod.V, mod.C) < 1e-1
    #     S_ = sp.csc_matrix(S_)
    #     S_ = algo.fn_invalid(S_)
    #     score = judge.score(S_)
    #     if score >= best:
    #         print(score)
    #         best = score
    #         S_0 = S_

    #S_0 = iolib.read_output(IG_FILE, mod.C, mod.V)

    # S_0 = algo.fn_invalid(S_0)
    # S_0 = simanneal.Propagator(mod).prune(S_0)



    print(judge.score(S_0))

    # Thermalize
    prop = simanneal.Propagator(mod)

    buf = iolib.OutputBuffer(args.scenario, outpath)

    print("START")
    for i, (S, E) in enumerate(algo(S_0, STEPS)):
        if i % 100 == 0:
            print("%05d" % i, "E=%d" % E, "T=%.3f" % algo.fn_temp(i))
        if i % 1000 == 0:
            buf.generate_output(S)
            buf.write_to_file("-%05d" % (i / 100))
        # if i % 10 == 0:
        #     print("SCORE=%d" % judge.score(S))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=int)
    parser.add_argument("scenario", type=str)
    #parser.add_argument("--profile", action="store_true")
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
