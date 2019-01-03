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

import argparse
import pygmo as pg
import rbfmopt
import cma

from multiprocessing import freeze_support
from rbfopt import RbfoptSettings
from rbfopt import RbfoptAlgorithm

# My inbuilt utils for rbfopt
import utils.rbfopt_utils as my_rbfopt_utils
import utils.rbfmopt_utils as rbfmopt_utils
import utils.rbfopt_model_utils as model_utils
import utils.cli_utils as cli_utils
import utils.global_record as global_record
import utils.pygmo_utils as pygmo_utils

import random

if(__name__ == "__main__"):
    # Was needed to make py2exe work
    # freeze_support()

    # Create rbfopt_cl line parsers
    desc = ('rbfopt_utiln RBFOpt, or get the current RBFOpt optimization state and evaluate the RBF surrogate model.')

    parser = argparse.ArgumentParser(description=desc)

    subparsers = parser.add_subparsers(help='Algorithms', dest='mode')

    # Create subparsers
    rbfmopt_subparser = subparsers.add_parser(
        'RBFOptWeightedSum', help='Rbfmopt algorithm settings')
    rbfopt_subparser = subparsers.add_parser(
        'RBFOpt', help='Rbfopt algorithm setttings')
    model_subparser = subparsers.add_parser(
        'RBFOptModel', help='Evaluate RBFOpt Surrogate Model and add points to it')
    nsgaii_subparser = subparsers.add_parser(
        'NSGAII', help='NSGAII algorithm settings')
    cmaes_subparser = subparsers.add_parser(
        'CMAES', help = 'CMA-ES algorithm settings')

    # Add Rbfopt options to rbfmopt and rbfopt sub parsers
    cli_utils.register_rbfmopt_options(rbfmopt_subparser)
    cli_utils.register_rbfopt_options(rbfopt_subparser)
    cli_utils.register_rbfopt_model(model_subparser)
    cli_utils.register_pygmo_options(nsgaii_subparser)
    cli_utils.register_cmaes_options(cmaes_subparser)

    # Add problem options
    cli_utils.register_problem_options(parser)

    # Get arguments
    args = parser.parse_args()

    # Run single-objective RBFOpt
    if args.mode == "RBFOpt":

        assert args.param_list is not None, "Parameter string is missing!"
        parameters = args.param_list

        # parse_known_args returns a tuple with the known parsed args
        # and the unknown args
        algo_args = rbfopt_subparser.parse_known_args()[0]

        # Open output stream if necessary
        output_stream = my_rbfopt_utils.open_output_stream(algo_args)

        # Create dictionary from subparser
        dict_args = vars(algo_args)

        # Remove non-RBFOpt arguments from algo_arguments because we want to
        # pass the dict_args with all the rbfopt setting arguments to
        # construct an Rbfopt settings object
        dict_args.pop('output_stream')

        # Use parameters to create Black Box and remove them from dictonary
        black_box = my_rbfopt_utils.construct_black_box(parameters,
                                                        my_rbfopt_utils.
                                                        read_write_obj_fun)

        settings = RbfoptSettings.from_dictionary(dict_args)

        # Run the interface
        alg = RbfoptAlgorithm(settings, black_box)

        if (output_stream is not None):
            alg.set_output_stream(output_stream)

        alg.optimize()

    # Run single-objective CMA-ES
    elif args.mode == "CMAES":

        assert args.param_list is not None, "Parameter string is missing!"
        parameters = args.param_list

        # Use parameters to create Black Box and remove them from dictonary
        black_box = my_rbfopt_utils.construct_black_box(parameters,
                                                        my_rbfopt_utils.
                                                        read_write_obj_fun)

        # parse_known_args returns a tuple with the known parsed args
        # and the unknown args
        algo_args = cmaes_subparser.parse_known_args()[0]

        # Create dictionary from subparser
        dict_args = vars(algo_args)
        
        #Check if inital solutions (i.e., starting point) is provided
        if algo_args.initialSolution is not None:
            initalSolutionStr = dict_args.pop('initialSolution')
            initalSolution = list(map(float, initalSolutionStr.split(';')))
            assert len(initalSolution) == black_box.get_dimension(), "Initial Solution for CMA-ES has incorrect dimension"
        #Else generate random starting point
        else:
            initalSolution = black_box.get_dimension() * [random.random()]
            
        #With random initial solution and initial sigma = 0.5
        #Integer Variables?
        #Max Evaluations?
        es = cma.CMAEvolutionStrategy(initalSolution, 0.5, {'bounds': [0,1], 'verb_log': 0})
        es.optimize(black_box.evaluate)
    
    # Run multi-objective RBFMopt
    elif args.mode == "RBFOptWeightedSum":
        
        assert args.param_list is not None, "Parameter string is missing!"
        parameters = args.param_list

        assert args.objective_n is not None, "Missing number of objectives!"
        objectiveN = args.objective_n

        # parse_known_args returns a tuple with the known parsed args
        # and the unknown args
        algo_args = rbfmopt_subparser.parse_known_args()[0]

        # set decomp_method global setting
        assert (algo_args.decomp_method == 'tchebycheff' or algo_args.decomp_method == 'weighted' or algo_args.decomp_method == 'bi'), "Invalid decomposition method!"
        weight_method = algo_args.decomp_method

        # Set hypervolume global setting
        if (algo_args.hyper):
            global_record.hv_bool = True

        # Set cycle number
        cycle = algo_args.cycle

        # Open output stream if necessary
        output_stream = my_rbfopt_utils.open_output_stream(algo_args)

        # Create dictionary from subparser
        dict_args = vars(algo_args)

        # Remove non-RBFOpt arguments from algo_arguments because we want to
        # pass the dict_args with all the rbfopt setting arguments to
        # construct an Rbfopt settings object
        dict_args.pop('output_stream')
        dict_args.pop('hyper')
        dict_args.pop('decomp_method')
        dict_args.pop('cycle')

        pygmo_read_write_problem, var_types = rbfmopt_utils.construct_pygmo_problem(
            parameters, objectiveN, rbfmopt_utils.read_write_obj_fun)

        alg = rbfmopt.RbfmoptWrapper(
            dict_settings=dict_args,
            problem=pygmo_read_write_problem,
            var_types=var_types,
            output_stream=output_stream,
            weight_method=weight_method,
            cycle=cycle,
            hv_array=global_record.hv_array)

        alg.evolve()

    # Evaluate RBFOpt Surrogate Model and add points to it
    elif args.mode == "RBFOptModel":

        assert args.path is not None, "Missing path parameter!"
        assert args.pointFile is not None, "Missing point file parameter!"
        assert args.valueFile is not None, "Missing value file parameter!"
        assert args.stateFile is not None, "Missing state file parameter!"

        # Read algorithm state (R prefix to accomadate spaces in the path)
        stateFilePath = args.path + args.stateFile        
        model = RbfoptAlgorithm.load_from_file(stateFilePath)
        
        # Add additional points before evaluation
        if args.addNodes is True:
            assert args.addPointsFile is not None, "Missing addPoints file parameter!"
            assert args.addValuesFile is not None, "Missing addValues file parameter!"
            
            model_utils.addPoints(args.path + args.addPointsFile, args.path + args.addValuesFile, model)

        # One point or value per line, point coordinates delimited by " ")
        if args.approximate is True:
            model_utils.evaluatePoints(args.path + args.pointFile, args.path + args.valueFile, model)
        else:
            model_utils.getPoints(args.path + args.pointFile, args.path + args.valueFile, model)

    # Run multi-objective NSGA-II
    elif args.mode == "NSGAII":

        assert args.param_list is not None, "Parameter string is missing!"
        parameters = args.param_list

        assert args.objective_n is not None, "Missing number of objectives!"
        objectiveN = args.objective_n

        algo_args = nsgaii_subparser.parse_known_args()[0]

        # Set hypervolume global setting
        if (algo_args.hyper):
            global_record.hv_bool = True

        seed = algo_args.seed
        pop_size = algo_args.pop_size

        # I need a way to turn a string into a call for the pygmo algorithm
        # But it is okay first do nsga II
        pygmo_read_write_problem, var_types = rbfmopt_utils.construct_pygmo_problem(
            parameters, objectiveN, rbfmopt_utils.read_write_obj_fun)

        algo_nsga2 = pg.algorithm(pg.nsga2(gen=1))

        pygmo_utils.evolve_pygmo_algo(algo_nsga2, pop_size, seed, pygmo_read_write_problem)

    else:
        # objs = [[0,0],[1,1],[2,2],[3,3]]
        # transposed = transpose(objs)
        # reverse = transpose(transposed)

        # print(*objs)
        # print(*transposed)
        # print(*reverse)

        str = "123456"  # Only digit in this string
        print(str.isdigit())
