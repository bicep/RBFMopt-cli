#######################################################################
#  File:      sumo_evaluate.py
#  Author(s): Thomas Wortmann
#             Singapore University of Techonology and Design
#             thomas_wortmann@mymail.sutd.edu.sg
#  Date:      05/15/16
#
#  (C) Copyright Singapore University of Technology and Design 2016.
#  You should have received a copy of the license with this code.
#  Research supported by the SUTD-MIT International Design Center.
#######################################################################
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import csv
import argparse
import ast
import unittest
import math

from multiprocessing import Pool
from multiprocessing import cpu_count
from multiprocessing import freeze_support

import numpy as np
import pygmo as pg

import time
import random

from rbfopt import RbfoptSettings
import rbfopt.rbfopt_utils as rbfopt_utils
from classes.RbfmoptWrapper import RbfmoptWrapper
from classes.PygmoUDP import PygmoUDP

# Read write "problem" evaluate
def read_write_obj_fun(x):
    #prepare input for simulator: list of variables values separbfopt_alg seperated by comma
    dimension = len(x)
    newline = ''
    for h in range(dimension - 1):
        var_value = x[h]
        newline = newline + '%.6f' % var_value + ','
    newline = newline + '%.6f' % x[dimension - 1]

    #write line
    sys.stdout.write(newline + "\n")
    sys.stdout.flush()
    #wait for reply
    func_value = sys.stdin.readline()

    obj_values = [float(x.strip()) for x in func_value.split(',')]

    return obj_values

def construct_pygmo_problem(parbfopt_algm_list, n_obj, obj_funct):
    #there should be a sequence of length 3*dimension of the form (lower_bound_i, upper_bound_i, integer_i)
    #separbfopt_algted by comma. If integer_i=1 integer variable, otherwise continuous
    #r_init = list(csv.reader(parbfopt_algm_list, delimiter=','))
    r_init = csv.reader([parbfopt_algm_list], delimiter=';')
    r_init = list(r_init)
    r_init = r_init[0] 
    params = r_init [1:]
    
    #Settings
    dimension = int(r_init [0])
    var_lower = np.array([None] * dimension)
    var_upper = np.array([None] * dimension)
    var_type = np.array(['R'] * dimension)
    obj_funct_fast = None
    
    #Check if correct length of parbfopt_algmeters
    assert(len(params) / 3 == dimension)
        
    #Set Variables
    for j in range(dimension):
        var_lower[j] = float(params[3 * j])
        var_upper[j] = float(params[3 * j + 1])
        if params[3 * j + 2] == '1': var_type[j] = 'I'
        
    #Check Variables
    assert(len(var_lower) == dimension)
    assert(len(var_upper) == dimension)
    assert(len(var_type) == dimension)
    
    udp = PygmoUDP(dimension, var_lower, var_upper, var_type, n_obj, obj_funct)
    return pg.problem(udp)

def register_options(parser):
    """Add options to the command line parser.

    Register all the options for the optimization algorithm.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser.

    See also
    --------   
    :class:`rbfopt_settings.RbfoptSettings` for a detailed description of
    all the command line options.
    """
    # Algorithmic settings
    algset = parser.add_argument_group('Algorithmic settings')
    # Get default values from here
    default = RbfoptSettings()
    attrs = vars(default)
    docstring = default.__doc__
    param_docstring = docstring[docstring.find('Parameters'):
                                docstring.find('Attributes')].split(' : ')
    param_name = [val.split(' ')[-1].strip() for val in param_docstring[:-1]]
    param_type = [val.split('\n')[0].strip() for val in param_docstring[1:]]
    param_help = [' '.join(line.strip() for line in val.split('\n')[1:-2])
                  for val in param_docstring[1:]]
    # We extract the default from the docstring in case it is
    # necessary, but we use the actual default from the object above.
    param_default = [val.split(' ')[-1].rstrip('.').strip('\'') 
                     for val in param_help]
    for i in range(len(param_name)):
        if (param_type[i] == 'float'):
            type_fun = float
        elif (param_type[i] == 'int'):
            type_fun = int
        elif (param_type[i] == 'bool'):
            type_fun = ast.literal_eval
        else:
            type_fun = str
        algset.add_argument('--' + param_name[i], action = 'store',
                            dest = param_name[i],
                            type = type_fun,
                            help = param_help[i],
                            default = getattr(default, param_name[i]))

#Get the model's bounds, evaluated points and values from a surrogate model
#and write them to a file
def getPoints(pointFilePath, valueFilePath, model):
        points = model.node_pos
        lowerBounds = model.l_lower
        upperBounds = model.l_upper
        boundsAndPoints = [lowerBounds, upperBounds]
        boundsAndPoints.extend(model.node_pos)
        writeFile(pointFilePath, boundsAndPoints)
        writeFile(valueFilePath, model.node_val)

#Read points from a file,
#and adds them to the surrogate model
def addPoints(pointFilePath, valueFilePath, model):
    #Read the points and values to add
    points = np.loadtxt(pointFilePath, delimiter = " ", ndmin = 2)
    values = np.loadtxt(valueFilePath, delimiter = " ", ndmin = 2)
    
    #Check that points have correct dimensionality
    assert len(points[0]) == len(model.l_lower), "Model has %d dimensions, but point file has %d" % (len(model.l_lower), len(points[0]))
    
    #Check that list is complete
    assert len(points) == len(values), "%d points, but %d values" % (len(points), len(values))
    
    #Sort according to objective (minimization)
    values, points = zip(*sorted(zip(values, points)))
    
    points = np.array(points)
    values = np.array(values)
    
    #Scale points
    points_scaled = rbfopt_utils.bulk_transform_domain(model.l_settings, model.l_lower, model.l_upper, points)
    
    #Add the points (last entry is for fast evals)
    #add_node(self, point, orig_point, value, is_fast)
    for i in range(len(points)):
        #Distance test
        if (rbfopt_utils.get_min_distance(points[i], model.all_node_pos) > model.l_settings.min_dist):
            model.add_node(points_scaled[i], points[i], values[i])
        else:
            print ("Point too close to add to model!")

#Read points from a file, 

#evaluate them based one the surrogate model (in RbfoptAlgorithm.evaluateRBF),
#and write the results to a file
def evaluatePoints(pointFilePath, valueFilePath, model):
    #Get the points to evaluate
    points = np.loadtxt(pointFilePath, delimiter = " ", ndmin = 2)
    
    #Check that points have correct dimensionality
    assert len(points[0]) == len(model.l_lower), "Model has %d dimensions, but point file has %d" % (len(model.l_lower), len(points[0]))
    
    #size = 1000000
    #var1 = [random.uniform(-5, 0) for x in xrange(size)]
    #var2 = [random.uniform(10, 15) for x in xrange(size)]
    #points = zip(var1, var2)
    
    #timer = time.clock    
    #start = timer()
   
    values = evaluateRBF((points, model))

    #end = timer()
    #print(str(end - start))
    
    #Combine the results and write them to a file
    
    writeFile(valueFilePath, values)

#Evaluates a set points based one a given surrogate model
#(with a single input object for multiprocessing)
def evaluateRBF(input):
    #Single input for parallel processing
    points, model = input

    # Write error message
    if len(points) == 0:
        sys.stdout.write("No points to evaluate!\n")  
        sys.stdout.flush()
    
    # Number of nodes at current iterbfopt_algtion
    k = len(model.node_pos)

    # Compute indices of fast node evaluations (sparse format)
    fast_node_index = (np.nonzero(model.node_is_fast)[0] if model.two_phase_optimization else np.array([]))
            
    # Rescale nodes if necessary
    tfv = rbfopt_utils.transform_function_values(model.l_settings, np.array(model.node_val), model.fmin, model.fmax, fast_node_index)
    (scaled_node_val, scaled_fmin, scaled_fmax, node_err_bounds, rescale_function) = tfv

    # Compute the matrices necessary for the algorithm
    Amat = rbfopt_utils.get_rbf_matrix(model.l_settings, model.n, k, np.array(model.node_pos))

    # Get coefficients for the exact RBF
    rc = rbfopt_utils.get_rbf_coefficients(model.l_settings, model.n, k, Amat, scaled_node_val)
    if (fast_node_index):
        # RBF with some fast function evaluations
        rc = aux.get_noisy_rbf_coefficients(model.l_settings, model.n, k, Amat[:k, :k], Amat[:k, k:], scaled_node_val, fast_node_index, node_err_bounds, rc[0], rc[1])
    (rbf_l, rbf_h) = rc
    
    # Evaluate RBF
    if len(points) <= 1:
        values = []    
        for point in points: values.append(rbfopt_util.evaluate_rbf(model.l_settings, point, model.n, k, np.array(model.node_pos), rbf_l, rbf_h))
        return values
    else: 
        return rbfopt_utils.bulk_evaluate_rbf(model.l_settings, np.array(points), model.n, k, np.array(model.node_pos), rbf_l, rbf_h)
    
#Write a list, or list of list, to a file
def writeFile(file, elements):
    with open(file, 'w') as file:
        for element in elements:
            if isinstance(element, (list, np.ndarray)):
                string = " ".join(str(value) for value in element)
                file.write(string + "\n")
            elif isinstance(element, (float, np.float64)):
                string = str(element)
                file.write(string + "\n")
            else: 
                print("Element " + str(type(element)) + " not recognized!")

#Check if an executable is on the path (for solvers)
def which(progrbfopt_algm):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(progrbfopt_algm)
    if fpath:
        if is_exe(progrbfopt_algm):
            return progrbfopt_algm
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, progrbfopt_algm)
            if is_exe(exe_file):
                return exe_file

    return None

#Run unit tests
def run_unit_test():
    #os.chdir("..")
    test_loader = unittest.TestLoader()
    
    test_suite = test_loader.discover('tests/', pattern='test_*.py')
    unittest.TextTestRunner().run(test_suite)
    
if(__name__ == "__main__"):
    #Needed to make py2exe work
    freeze_support()

    #Create rbfopt_cl line parsers
    desc = ('rbfopt_utiln RBFOpt, or get the current RBFOpt optimization state and evalueate the RBF surrogate model.')
    parser = argparse.ArgumentParser(description=desc)

    #Add Rbfopt options to parser
    register_options(parser)

    #Algorithm flags as mutually exclusive groups
    algo_action = parser.add_mutually_exclusive_group(required=True)
    algo_action.add_argument('--optimizeRBFWeightedSum', action = 'store_true', help = 'RBF Multi-Objective Optimization')
    algo_action.add_argument('--optimizeRBF', action='store_true', help = 'RBF Single-Objective Optimization')
    algo_action.add_argument('--optimizePygmo', action='store_true', help = 'Pygmo Multi-Objective Optimization')

    #Add additional options
    parser.add_argument('--objective_n', '--objectiveN', action = 'store', dest = 'objective_n', metavar = 'OBJECTIVE_N', type = int, default = 1, help = 'number of objectives')
	
    parser.add_argument('--param_list', '--param', action = 'store', dest = 'param_list', metavar = 'PARAM_LIST', type = str, help = 'list of parameters for initialization')
    parser.add_argument('--addNodes', action = 'store_true', help = 'add the points from the addPointsFile and addValuesFile to the model')
    parser.add_argument('--path', action = 'store', type = str, help = 'path for files, default is script directory')
    parser.add_argument('--addPointsFile', action = 'store', type = str, help = 'file name for points to add to the model')
    parser.add_argument('--addValuesFile', action = 'store', type = str, help = 'file name for objective values to add to the model')
    parser.add_argument('--log', '-o', action = 'store', metavar = 'LOG_FILE_NAME', type = str, dest = 'output_stream', help = 'Name of log file for output redirection')



    #Get arguments
    args = parser.parse_args()
	
    #Run RBFOpt
    if args.optimizeRBFWeightedSum:
        assert args.param_list is not None, "Missing variable list parameters!"

        #Open output stream if necessary
        output_stream = None

        if(args.output_stream is not None):
            try:
                output_stream = open(args.output_stream, 'w')
            except IOError as e:
                print('Error while opening log file', file = sys.stderr)
                print(e, file = sys.stderr)

        #Add Additional points before evaluation
        if args.addNodes:
            assert args.path is not None, "Missing path parameters!"
            assert args.addPointsFile is not None, "Missing addPoints file parameter!"
            assert args.addValuesFile is not None, "Missing addValues file parameter!"

            points, values = readPoints(args.path + args.addPointsFile, args.path + args.addValueFile)

            pts = points.copy
            objs = values.copy
        else:
            points = None
            values = None
        
        #Create dictionary from parser
        dict_args = vars(args)

        #Remove non-RBFOpt arguments from directory
        dict_args.pop('output_stream')
        dict_args.pop('optimizeRBFWeightedSum')
        dict_args.pop('addNodes')
        dict_args.pop('path')
        dict_args.pop('addPointsFile')
        dict_args.pop('addValuesFile')

        parameters = dict_args.pop('param_list')
        objectiveN = dict_args.pop('objective_n')
		
        pygmo_read_write_problem = construct_pygmo_problem(parameters, objectiveN, read_write_obj_fun)
        
        alg = RbfmoptWrapper(dict_args, pygmo_read_write_problem)
        
        if(output_stream is not None): alg.set_output_stream(output_stream)
	
        alg.evolve()
	
    # elif args.optimizeRBF: #This is for the single objective rbfopt 
    # elif args.optimizePygmo:
        # I need a way to turn a string into a call for the pygmo algorithm
    else:
        #objs = [[0,0],[1,1],[2,2],[3,3]]
        #transposed = transpose(objs)
        #reverse = transpose(transposed)
		
        #print(*objs)
        #print(*transposed)
        #print(*reverse)
        
        str = "123456";  # Only digit in this string
        print (str.isdigit())