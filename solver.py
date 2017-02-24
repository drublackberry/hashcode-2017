#!/usr/bin/env python

"""
Intentionally left blank.
"""


from model import Model
import os


def solve(scenario, output_dir):
    i = 0
    # Load the scenario, it populates the Rn, L_ec, LD_e and Rn_ev matrices
    my_model = Model()
    my_model.load_scenario(scenario)
    # Loop until all servers are used
    while my_model.cache_servers_available():
        # Compute the costs of servers J_cv
        i = i+1
        J_cv = my_model.compute_J_cv()    
        J_c = my_model.compute_J_c(J_cv)
        # Rank the servers by cost
        cache_order = my_model.sort_cache_server_by_J_c(J_c)
        c = cache_order[0]
        #J_cv.to_excel(os.path.join(output_dir, str(c)+'_Jcv.xlsx'))        
        print("Sorting server {}".format(c))
        my_model.store_video_in_cache_server(J_cv, c)
        # After my_modelving for the server dump the storage matrix
        my_model.write_storage(output_dir, i)
