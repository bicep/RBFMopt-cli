import sys
import csv
import numpy as np
import pygmo as pg
from classes.PygmoUDP import PygmoUDP


def construct_pygmo_problem(parbfopt_algm_list, n_obj, obj_funct):
    # there should be a sequence of length 3*dimension of the form 
    # (lower_bound_i, upper_bound_i, integer_i)
    # separbfopt_algted by comma. If integer_i=1 integer variable, otherwise 
    # continuous
    # r_init = list(csv.reader(parbfopt_algm_list, delimiter=','))
    r_init = csv.reader([parbfopt_algm_list], delimiter=';')
    r_init = list(r_init)
    r_init = r_init[0]
    params = r_init[1:]

    # Settings
    dimension = int(r_init[0])
    var_lower = np.array([None] * dimension)
    var_upper = np.array([None] * dimension)
    var_type = np.array(['R'] * dimension)

    # Check if correct length of parbfopt_algmeters
    assert(len(params) / 3 == dimension)

    # Set Variables
    for j in range(dimension):
        var_lower[j] = float(params[3 * j])
        var_upper[j] = float(params[3 * j + 1])
        if params[3 * j + 2] == '1':
            var_type[j] = 'I'

    # Check Variables
    assert(len(var_lower) == dimension)
    assert(len(var_upper) == dimension)
    assert(len(var_type) == dimension)

    udp = PygmoUDP(dimension, var_lower, var_upper, var_type, n_obj, obj_funct)
    return pg.problem(udp)


# Read write "problem" evaluate
def read_write_obj_fun(x):
    # prepare input for simulator: list of variables values separbfopt_alg
    # seperated by comma
    dimension = len(x)
    newline = ''
    for h in range(dimension - 1):
        var_value = x[h]
        newline = newline + '%.6f' % var_value + ','
    newline = newline + '%.6f' % x[dimension - 1]

    # write line
    sys.stdout.write(newline + "\n")
    sys.stdout.flush()
    # wait for reply
    func_value = sys.stdin.readline()

    obj_values = [float(x.strip()) for x in func_value.split(',')]

    return obj_values


# Weighted Objective Function (Augmented Chebyshev)
def calculate_weighted_objective(weights, values, rho):
    weighted_vals = [value * weight for value, weight in zip(values, weights)] 
    aug_tcheby = max(weighted_vals) + rho * sum(weighted_vals)
    return aug_tcheby
