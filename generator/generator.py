import numpy as np

gen = lambda a, b: np.random.default_rng().uniform(a, b)

def generate_model(N, M, epsilon=0.01):
    t_start = 1 / N
    o_start = 1 / M

    distribution = [0] * N
    transition = np.matrix(np.zeros((N, N)))
    output = np.matrix(np.zeros((N, M)))

    for i in range(N):
        for j in range(N - 1):
            distribution[j] = round(gen(t_start - epsilon, 
                                                t_start), 3)
            transition[i, j] = round(gen(t_start - epsilon, 
                                                t_start), 3)

        for k in range(M - 1):
            output[i, k] = round(gen(o_start - epsilon, 
                                                o_start), 3)

    distribution[-1] = 1 - sum(distribution[:-1])
    for i in range(N):
        transition[i, -1] = 1 - transition.sum(axis=1).item(i)
        output[i, -1] = 1 - output.sum(axis=1).item(i)
    
    return distribution, transition, output