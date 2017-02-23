#!/usr/bin/env python

"""
I/O functions for Hashcode 2017.
"""

import numpy as np
import pandas as pd
import os

INPUT_DIR = os.path.join(".", "input")

def read_scenario(scen_name):
    fname = os.path.join(INPUT_DIR, scen_name + ".in")
    with open(fname, 'r') as fp:
        V, E, R, C, X = [int(x) for x in fp.readline().strip().split()]
        v_sizes = [int(x) for x in fp.readline().strip().split()]
    #print(V, E, R, C, X)
    #print(len(v_sizes))

    # for each endnode
    start = 2
    endpoint_cache_lats = []
    vec_L_d = np.ones(E, dtype=np.float64) * np.nan
    for e in range(E):
        L_d, K = np.genfromtxt(fname, skip_header=start, max_rows=1, dtype=np.int64)
        vec_L_d[e] = L_d
        #print(L_d, K, e)
        foo = np.genfromtxt(fname, skip_header=start+1, max_rows=K, dtype=np.int64)
        if len(foo.shape) == 1:
            foo = foo.reshape(1, len(foo))
        endpoint_cache_lats.append(foo)
        #print("****\n", foo)
        start += 1 + K
    assert(len(endpoint_cache_lats) == E)

    # Now read requests
    requests = np.genfromtxt(fname, skip_header=start, max_rows=R)

    L = np.ones([E, C]) * np.nan
    for e in range(E):
        lats = endpoint_cache_lats[e]
        # lats[:, 0] has the cache IDs!
        #print(lats.shape, L.shape)
        L[e, lats[:, 0]] = lats[:, 1]

    R_n = np.ones([E, V]) * np.nan
    req = requests.astype(np.int64)

    R_n[req[:, 1], req[:, 0]] = requests[:, 2]

    data = {}
    data['R_n'] = pd.DataFrame(R_n)#, columns=['v'], rows=['e'])
    data['L_d'] = pd.DataFrame(vec_L_d)#, rows=['e'])
    data['L'] = pd.DataFrame(L)#, rows)=['e'], columns=['c'])
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
        for c in storage.shape[1]:
            stuff[c] = [vid for vid in storage.shape[0]
                        if storage[vid, c]]
        self.result = stuff

    def write_to_file(self, fname=None):
        if fname is None:
            fname = os.path.join(self.output_dir, prefix + ".out")
        with open(fname, 'w') as fp:
            for line in self.result:
                fp.write(" ".join(["%d" % n for n in line]))
                fp.write("\n")

if __name__ == "__main__":
    # test
    data = read_scenario("me_at_the_zoo")
    print(data.keys())
    for k in data.keys():
        print(data[k])
