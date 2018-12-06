import sys
import csv
import numpy as np
import pygmo as pg
from classes.PygmoUDP import PygmoUDP
from utils.rbfopt_utils import parse_variable_string
import utils.hv_record as hv_record


def construct_pygmo_problem(parbfopt_algm_list, n_obj, obj_funct):

    dimension, var_lower, var_upper, var_type = parse_variable_string(parbfopt_algm_list)
    udp = PygmoUDP(dimension, var_lower, var_upper, var_type, n_obj, obj_funct)

    return (pg.problem(udp), var_type)


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

    # hv_bool and hv_array are our global records
    if (hv_record.hv_bool):
        sys.stdout.write('%.6f' % hv_record.hv_array[-1] + ',')

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
