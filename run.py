#!/usr/bin/env python

import argparse
import os
import tempfile
import zipfile
import glob
from solver import solve


def main(args, outpath):
    solve(args.scenario, outpath, model=args.model)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=int)
    parser.add_argument("scenario", type=str)
    parser.add_argument("--model", type=str, default="dense")
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
