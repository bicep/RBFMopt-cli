import pygmo as pg


# Calculates the hypervolume with a changing ref point
def calculate_hv(hv_pop):

    # Have the same ref point at the beginning, and compute the starting hypervolume
    new_hv = pg.hypervolume(hv_pop)
    ref = new_hv.refpoint(offset=4.0)
    return new_hv.compute(ref)
# import math


# Stores the f and x values for each generation of the evolutionary algo,
# Then calculate the hypervolume per function evaluation
def evolve_pygmo_algo(algo, pop_size, seed, problem):

    # max_gen = math.ceil(max_fevals/pop_size)

    # same (random) starting population for algo
    pop = pg.population(problem, pop_size, seed)

    # for i in range(max_gen):
    while(True):
        pop = algo.evolve(pop)
