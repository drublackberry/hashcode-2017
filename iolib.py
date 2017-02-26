#!/usr/bin/env python

"""
I/O functions for Hashcode 2017.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import os

INPUT_DIR = os.path.join(".", "input")

def read_scenario(scen_name):
    fname = os.path.join(INPUT_DIR, scen_name + ".in")
    with open(fname, 'r') as fp:
        V, E, R, C, X = [int(x) for x in fp.readline().strip().split()]
        v_sizes = [int(x) for x in fp.readline().strip().split()]

    # for each endnode
    start = 2
    endpoint_cache_lats = []

    # L_d is dense
    vec_L_d = np.empty(E, dtype=np.int16)

    for e in range(E):
        L_d, K = np.genfromtxt(fname, skip_header=start, max_rows=1, dtype=np.int64)
        vec_L_d[e] = L_d
        if K == 0:
            foo = []
        else:
            foo = np.genfromtxt(fname, skip_header=start+1, max_rows=K, dtype=np.int16)
            if len(foo.shape) == 1:
                foo = foo.reshape(1, len(foo))
        endpoint_cache_lats.append(foo)
        start += 1 + K
    assert(len(endpoint_cache_lats) == E)

    # Now read requests
    requests = np.genfromtxt(fname, skip_header=start, max_rows=R, dtype=np.int16)

    L = np.ones([E, C]) * np.nan

    for e in range(E):
        lats = endpoint_cache_lats[e]
        if len(lats) > 0:
            L[e, lats[:, 0]] = lats[:, 1]

    R_n = np.zeros([E, V], dtype=np.int32)
    req = requests.astype(np.int16)

    #R_n[req[:, 1], req[:, 0]] = requests[:, 2]
    for r, (v, e, n) in enumerate(requests):
        R_n[e, v] += n

    data = {}
    data['R_n'] = pd.DataFrame(R_n)
    data['L_d'] = pd.DataFrame(vec_L_d)
    data['L'] = pd.DataFrame(L)
    data['V'] = V
    data['E'] = E
    data['R'] = R
    data['C'] = C
    data['X'] = X
    data['v_size'] = pd.DataFrame(v_sizes)
    return data


class OutputBuffer(object):

    def __init__(self, prefix, output_dir):
        self.prefix = prefix
        self.output_dir = output_dir
        self.result = None

    def generate_output(self, storage):
        # storage is a V X C matrix
        stuff = {}

        try:
            for c in storage.columns:
                storage_c = storage[c]
                stuff[c] = list(storage_c[storage_c>0].index)
        except:
            # not a DataFrame
            n = storage.shape[1]
            storage = storage.toarray()
            #print(storage.shape)
            for c in range(n):
                inds = np.argwhere(storage[:, c] > 0)
                #print(inds)
                if len(inds) > 0:
                    inds = inds.flatten()
                stuff[c] = inds
        self.result = dict([(k, v) for (k, v) in stuff.items() if len(v) > 0])

    def write_to_file(self, fname=None):
        if fname is None:
            fname = os.path.join(self.output_dir, self.prefix + ".out")
        else:
            fname = os.path.join(self.output_dir, self.prefix + fname + ".out")
        with open(fname, 'w') as fp:
            fp.write("%d\n" % len(self.result))
            for c, line in self.result.items():
                fp.write("%d " % c)
                fp.write(" ".join(["%d" % n for n in line]))
                fp.write("\n")


def read_output(fname, C, V):
    S = sp.lil_matrix((V, C), dtype=np.int8)
    with open(fname, 'r') as fp:
        fp.readline()
        for line in fp.readlines():
            data = np.asarray([int(x) for x in line.strip().split()])
            S[data[1:], data[0]] = True
    return S.tocsc()


if __name__ == "__main__":
    # test
    data = read_scenario("me_at_the_zoo")
    print(data.keys())
    for k in data.keys():
        print(data[k])
