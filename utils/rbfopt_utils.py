import csv
import sys
import numpy as np
from rbfopt import RbfoptUserBlackBox


def add_additional_nodes(algo_args):
    points = None
    values = None
    # Add Additional points before evaluation
    if algo_args.addNodes:
        assert algo_args.path is not None, "Missing path parameters!"
        assert algo_args.addPointsFile is not None, "Missing addPoints file parameter!"
        assert algo_args.addValuesFile is not None, "Missing addValues file parameter!"

        points, values = readPoints(
            algo_args.path + algo_args.addPointsFile,
            algo_args.path + algo_args.addValueFile)
    return (points, values)


def open_output_stream(algo_args):
    output_stream = None

    if(algo_args.output_stream is not None):
        try:
            output_stream = open(algo_args.output_stream, 'w')
        except IOError as e:
            print('Error while opening log file', file=sys.stderr)
            print(e, file=sys.stderr)

    return output_stream


def readPoints(pointFilePath, valueFilePath):
    """Read the points and values from a file
    Parameters
    ----------
    pointFilePath : str
        File Path to read points from.
    valueFilePath : str
        File Path to read values from.
    """
    points = np.loadtxt(pointFilePath, delimiter=" ", ndmin=2)
    values = np.loadtxt(valueFilePath, delimiter=" ", ndmin=2)

    # check that list is complete
    assert len(points) == len(values), "%d points, but %d values" % (len(points), len(values))

    # sort according to objective (minimization)
    # values, points = zip(*sorted(zip(values, points)),
    # key=lambda pair: pair[0])

    points = np.array(points)
    values = np.array(values)

    return points, values


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


# Opossum Black Box
def construct_black_box(parbfopt_algm_list, obj_funct):
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

    return RbfoptUserBlackBox(dimension,
                              var_lower, var_upper, var_type, obj_funct)
