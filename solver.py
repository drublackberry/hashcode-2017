#!/usr/bin/env python

"""
Intentionally left blank.
"""


from model import DenseModel, SparseModel


def solve(scenario, output_dir, model="dense"):
    # Load the scenario, it populates the Rn, L_ec, LD_e and Rn_ev matrices

    if model == "dense":
        my_model = DenseModel(scenario)
    elif model == "sparse":
        my_model = SparseModel(scenario)
    # Loop until all servers are used
    while my_model.cache_servers_available():
        # Compute the costs of servers J_cv
        print(my_model.caches)
        J_cv = my_model.compute_J_cv()
        J_c = my_model.compute_J_c(J_cv)
        # Rank the servers by cost
        #print(J_c)
        cache_order = my_model.sort_cache_server_by_J_c(J_c)
        #print(cache_order)
        for i, c in enumerate(cache_order):
            print('Sorting server' + str(c))
            my_model.store_video_in_cache_server(J_cv, c)
            # After my_modelving for the server dump the storage matrix
            my_model.write_storage(output_dir, i)
