import numpy as np

def get_stationary_dist(transition, iters=50):
    initial = np.zeros(transition.shape[0])
    initial[0] = 1

    for _ in range(iters):
        updated_initial = initial @ transition
        initial = updated_initial
        
    return initial