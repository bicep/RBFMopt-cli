import pygmo as pg
import matplotlib.pyplot as plt
import math as math
import numpy as numpy
from scipy.interpolate import UnivariateSpline
from utils.pygmo_utils import calculate_mean, get_hv_for_algo, reconstruct_hv_per_feval, plot_spline
from classes.RbfmoptWrapper import RbfmoptWrapper

# where n is the number of times the meta-heuristic algorithms are run to get the mean
n = 10
max_fevals = 100
# To account for the fact that the zero index array in util functions
# are actually the 1st feval
working_fevals = max_fevals-1
pop_size = 24
seed = 33
problem = pg.problem(pg.problems.dtlz(2, dim=6, fdim=2))
algo_moead = pg.algorithm(pg.moead(gen=1))
algo_nsga2 = pg.algorithm(pg.nsga2(gen=1))
algo_rbfmopt = RbfmoptWrapper(max_fevals, problem)

# RBFMopt hypervolume calculations
algo_rbfmopt.evolve()
empty_pop = pg.population(prob=problem, seed=seed)
hv_rbfmopt_plot = reconstruct_hv_per_feval(working_fevals, algo_rbfmopt.get_x_list(), algo_rbfmopt.get_f_list(), empty_pop) 

# Meta-heuristics hypervolume calculations, mean taken over n number of times
hv_moead_plot = calculate_mean(n, algo_moead, working_fevals, pop_size, seed, problem)
hv_nsga2_plot = calculate_mean(n, algo_nsga2, working_fevals, pop_size, seed, problem)
fevals_plot = range(0, max_fevals)

fig, ax = plt.subplots()

plot_spline(plt, fevals_plot, hv_rbfmopt_plot, max_fevals, "rbfmopt")
plot_spline(plt, fevals_plot, hv_moead_plot, max_fevals, "moead")
plot_spline(plt, fevals_plot, hv_nsga2_plot, max_fevals, "nsga2")

plt.legend(loc='best')
plt.xlabel('Function evaluations')
plt.ylabel('Mean hypervolume over '+str(n)+' runs')
plt.grid()
plt.savefig("/Users/rogerko/dev/Opossum/pygmo/graphics/Benchmark_Multi3.png")
plt.show()
