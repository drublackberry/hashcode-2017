#!/usr/bin/env python

"""
Intentionally left blank.
"""


from model import Model

scenario = 'kittens'

def solve (scenario):
    # Load the scenario, it populates the Rn, L_ec, LD_e and Rn_ev matrices
    solver = Model()
    solver.load_scenario(scenario)
    # Loop until all servers are used
    while solver.cache_servers_available():
        # Compute the costs of servers J_cv
        J_cv = solver.compute_J_cv()
        J_c = solver.compute_J_c()
        # Rank the servers by cost
        cache_order = solver.rank_cache_server_by_J_c(J_c)
        for c in cache_order:
            solver.store_video_in_cache_server(J_cv, c)                
        # After solving for the server dump the storage matrix
        solver.write_storage()
    
        
    
