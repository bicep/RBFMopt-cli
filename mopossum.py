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

from multiprocessing import freeze_support
from rbfopt import RbfoptSettings
from rbfopt import RbfoptAlgorithm

# My inbuilt utils for rbfopt
import utils.rbfopt_utils as my_rbfopt_utils
import utils.rbfmopt_utils as rbfmopt_utils
import utils.rbfopt_model_utils as model_utils
import utils.cli_utils as cli_utils
from classes.RbfmoptWrapper import RbfmoptWrapper

if(__name__ == "__main__"):
    # Needed to make py2exe work
    freeze_support()

    # Create rbfopt_cl line parsers
    desc = ('rbfopt_utiln RBFOpt, or get the current RBFOpt optimization state and evalueate the RBF surrogate model.')

    parser = argparse.ArgumentParser(description=desc)

    subparsers = parser.add_subparsers(help='Algorithms', dest='mode')

    # Create subparsers
    rbfmopt_subparser = subparsers.add_parser(
        'RBFOptWeightedSum', help='Rbfmopt algorithm settings')
    rbfopt_subparser = subparsers.add_parser(
        'RBFOpt', help='Rbfopt algorithm setttings')
    model_subparser = subparsers.add_parser(
        'RBFOptModel', help='Evaluate RBFOpt Surrogate Model and add points to it')

    # Add Rbfopt options to rbfmopt and rbfopt sub parsers
    cli_utils.register_rbfopt_options(rbfmopt_subparser)
    cli_utils.register_rbfopt_options(rbfopt_subparser)
    cli_utils.register_rbfopt_model(model_subparser)

    # Add problem options
    cli_utils.register_problem_options(parser)

    # Get arguments
    args = parser.parse_args()

    # Run RBFOpt
    if args.mode == "RBFOpt":  # This is for the single objective rbfopt

        assert args.param_list is not None, "Parameter strign is missing!"
        parameters = args.param_list

        # parse_known_args returns a tuple with the known parsed args
        # and the unknown args
        algo_args = rbfmopt_subparser.parse_known_args()[0]

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
    
    elif args.mode == "RBFOptWeightedSum":
        
        assert args.param_list is not None, "Parameter string is missing!"
        parameters = args.param_list

        assert args.objective_n is not None, "Missing number of objectives!"
        objectiveN = args.objective_n

        # parse_known_args returns a tuple with the known parsed args
        # and the unknown args
        algo_args = rbfmopt_subparser.parse_known_args()[0]

        # Open output stream if necessary
        output_stream = my_rbfopt_utils.open_output_stream(algo_args)

        # Create dictionary from subparser
        dict_args = vars(algo_args)

        # Remove non-RBFOpt arguments from algo_arguments because we want to
        # pass the dict_args with all the rbfopt setting arguments to
        # construct an Rbfopt settings object
        dict_args.pop('output_stream')

        pygmo_read_write_problem, var_types = rbfmopt_utils.construct_pygmo_problem(
            parameters, objectiveN, rbfmopt_utils.read_write_obj_fun)

        alg = RbfmoptWrapper(dict_args, pygmo_read_write_problem, var_types, output_stream)

        alg.evolve()

    elif args.mode == "RBFOptModel": # This is to evaluate RBFOpt Surrogate Model and add points to it

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
            assert args.addValuesFile is not None, "Missing addCalues file parameter!"
            
            model_utils.addPoints(args.path + args.addPointsFile, args.path + args.addValuesFile, model)

        # One point or value per line, point coordinates delimited by " ")
        if args.approximate is True:
            model_utils.evaluatePoints(args.path + args.pointFile, args.path + args.valueFile, model)
        else:
            model_utils.getPoints(args.path + args.pointFile, args.path + args.valueFile, model)

    # elif args.optimizePygmo:
        # I need a way to turn a string into a call for the pygmo algorithm

    else:
        # objs = [[0,0],[1,1],[2,2],[3,3]]
        # transposed = transpose(objs)
        # reverse = transpose(transposed)

        # print(*objs)
        # print(*transposed)
        # print(*reverse)

        str = "123456"  # Only digit in this string
        print(str.isdigit())
