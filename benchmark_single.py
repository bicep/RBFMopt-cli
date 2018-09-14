import pygmo as pg
import matplotlib.pyplot as plt
import math as math
import numpy as numpy
import rbfopt as rbfopt
from scipy.interpolate import UnivariateSpline
from utils.pygmo_utils import calculate_mean, get_champion_for_algo
from classes.RbfoptWrapper import RbfoptWrapper

# where n is the number of times the algorithm is run to get the mean
dimensions = 2
n = 10
max_fevals = 100
pop_size = 10
seed = 5
problem = pg.problem(pg.problems.rosenbrock(dimensions))
algo_cmaes = pg.algorithm(pg.cmaes(gen=1))
algo_pso = pg.algorithm(pg.pso(gen=1))


settings = rbfopt.RbfoptSettings(minlp_solver_path='/Users/rogerko/dev/Opossum/pygmo/solvers/bonmin-osx/bonmin', nlp_solver_path='/Users/rogerko/dev/Opossum/pygmo/solvers/ipopt-osx/ipopt', max_evaluations=max_fevals)
algo_rbfopt = RbfoptWrapper(settings, problem)
val, x, itercount, evalcount, fast_evalcount = algo_rbfopt.evolve()
champion_rbfopt_plot = algo_rbfopt.get_champions()


champion_cmaes_plot = calculate_mean(n, algo_cmaes, max_fevals, pop_size, seed, problem, get_champion_for_algo)
champion_pso_plot = calculate_mean(n, algo_pso, max_fevals, pop_size, seed, problem, get_champion_for_algo)

fevals_plot = range(0, max_fevals)
# plt.scatter(fevals_plot, champion_pso_plot, label="cmaes", marker='x')
# plt.scatter(fevals_plot, champion_cmaes_plot, label="pso", marker='x')

fig, ax = plt.subplots()

spline_cmaes = UnivariateSpline(fevals_plot, champion_cmaes_plot, s=5)
spline_pso = UnivariateSpline(fevals_plot, champion_pso_plot, s=5)
spline_rbfopt = UnivariateSpline(fevals_plot, champion_rbfopt_plot, s=5)


spline_x = numpy.linspace(0, max_fevals, 20)
spline_cmaes = spline_cmaes(spline_x)
spline_pso = spline_pso(spline_x)
spline_rbfopt = spline_rbfopt(spline_x)

plt.plot(spline_x, spline_cmaes, label="cmaes")
plt.plot(spline_x, spline_pso, label="pso")
plt.plot(spline_x, spline_rbfopt, label="rbfopt")

plt.ylim(ymax=500, ymin=0)
plt.legend(loc='best')
plt.xlabel('Function evaluations')
plt.ylabel('Mean champion over '+str(n)+' runs')
plt.grid()
plt.savefig("/Users/rogerko/dev/Opossum/pygmo/graphics/Benchmark_Single.png")
plt.show()
