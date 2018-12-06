import csv
import sys
import numpy as np
from rbfopt import RbfoptUserBlackBox

def open_output_stream(algo_args):
    output_stream = None

    if(algo_args.output_stream is not None):
        try:
            output_stream = open(algo_args.output_stream, 'w')
        except IOError as e:
            print('Error while opening log file', file=sys.stderr)
            print(e, file=sys.stderr)

    return output_stream

# Opossum Evaluate
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

    return float(func_value)

# Parse Variable String
def parse_variable_string(parbfopt_algm_list):
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

    return(dimension, var_lower, var_upper, var_type)


# Opossum Black Box
def construct_black_box(parbfopt_algm_list, obj_funct):
    
    dimension, var_lower, var_upper, var_type = parse_variable_string(parbfopt_algm_list)

    return RbfoptUserBlackBox(dimension,
                              var_lower, var_upper, var_type, obj_funct)
