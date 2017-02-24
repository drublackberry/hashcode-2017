#!/usr/bin/env python

"""
Intentionally left blank.
"""


from model import DenseModel, SparseModel
import sys


def solve(scenario, output_dir, model="dense"):
    # Load the scenario, it populates the Rn, L_ec, LD_e and Rn_ev matrices

    if model == "dense":
        my_model = DenseModel(scenario)
    elif model == "sparse":
        my_model = SparseModel(scenario)
    # Loop until all servers are used
    global_step = 0
    while my_model.cache_servers_available():
        print(global_step, len(my_model.caches))
        J_cv = my_model.compute_J_cv()
        J_c = my_model.compute_J_c(J_cv)
        c = my_model.sort_cache_server_by_J_c(J_c)[0]
        sys.stdout.flush()
        try:
            my_model.store_video_in_cache_server(J_cv, c)
        except StopIteration:
            continue
        my_model.write_storage(output_dir, global_step)
        global_step += 1
