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
import utils.cli_utils as cli_utils
from classes.RbfmoptWrapper import RbfmoptWrapper

if(__name__ == "__main__"):
    # Needed to make py2exe work
    freeze_support()

    # Create rbfopt_cl line parsers
    desc = ('rbfopt_utiln RBFOpt, or get the current RBFOpt optimization state and evalueate the RBF surrogate model.')

    parser = argparse.ArgumentParser(description=desc)

    subparsers = parser.add_subparsers(help='Algorithms', dest='mode')

    # Three subparsers for the three categories of algorithms
    rbfmopt_subparser = subparsers.add_parser(
        'RbfWeightedSum', help='Rbfmopt algorithm settings')
    rbfopt_subparser = subparsers.add_parser(
        'Rbfopt', help='Rbfopt algorithm setttings')

    # Add Rbfopt options to rbfmopt and rbfopt sub parsers
    cli_utils.register_rbfopt_options(rbfmopt_subparser)
    cli_utils.register_rbfopt_options(rbfopt_subparser)

    # Add problem options
    cli_utils.register_problem_options(parser)

    # Get arguments
    args = parser.parse_args()

    # Run RBFOpt
    if args.mode == "RbfWeightedSum":

        # param_list is definitely filled already because the parser settings
        # require it
        assert args.objective_n is not None, "Missing number of objectives!"

        parameters = args.param_list
        objectiveN = args.objective_n

        # parse_known_args returns a tuple with the known parsed args
        # and the unknown args
        algo_args = rbfmopt_subparser.parse_known_args()[0]

        # Open output stream if necessary
        output_stream = my_rbfopt_utils.open_output_stream(algo_args)

        points, values = my_rbfopt_utils.add_additional_nodes(algo_args)

        # Create dictionary from subparser
        dict_args = vars(algo_args)

        # Remove non-RBFOpt arguments from algo_arguments because we want to
        # pass the dict_args with all the rbfopt setting arguments to
        # construct an Rbfopt settings object
        dict_args.pop('output_stream')
        dict_args.pop('addNodes')
        dict_args.pop('path')
        dict_args.pop('addPointsFile')
        dict_args.pop('addValuesFile')

        pygmo_read_write_problem = rbfmopt_utils.construct_pygmo_problem(
            parameters, objectiveN, rbfmopt_utils.read_write_obj_fun)

        alg = RbfmoptWrapper(dict_args, pygmo_read_write_problem)

        if(output_stream is not None):
            alg.set_output_stream(output_stream)

        alg.evolve()

    elif args.mode == "Rbfopt":  # This is for the single objective rbfopt

        parameters = args.param_list

        # parse_known_args returns a tuple with the known parsed args
        # and the unknown args
        algo_args = rbfmopt_subparser.parse_known_args()[0]

        # Open output stream if necessary
        output_stream = my_rbfopt_utils.open_output_stream(algo_args)

        points, values = my_rbfopt_utils.add_additional_nodes(algo_args)

        # Create dictionary from subparser
        dict_args = vars(algo_args)

        # Remove non-RBFOpt arguments from algo_arguments because we want to
        # pass the dict_args with all the rbfopt setting arguments to
        # construct an Rbfopt settings object
        dict_args.pop('output_stream')
        dict_args.pop('addNodes')
        dict_args.pop('path')
        dict_args.pop('addPointsFile')
        dict_args.pop('addValuesFile')

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
