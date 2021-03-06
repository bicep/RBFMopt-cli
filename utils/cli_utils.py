import ast
from rbfopt import RbfoptSettings
import argparse

#Problem Parser
def register_problem_options(parser):
    algset = parser.add_argument_group('Problem settings')
    algset.add_argument('--objective_n', '--objectiveN', action='store', dest='objective_n',
                        metavar='OBJECTIVE_N', type=int, default=1, help='number of objectives')
    algset.add_argument('--param_list', '--param', action='store', dest='param_list',
                        metavar='PARAM_LIST', type=str, help='list of parameters for initialization')

#Pygmo Parser (e.g, NSGA-II)
def register_pygmo_options(parser):
    algset = parser.add_argument_group('Algorithmic settings')
    algset.add_argument('--pop_size', action='store', dest='pop_size', type=int, default=24, help='Population size for evolutionary algorithm to work on')
    algset.add_argument('--seed', action='store', dest='seed', type=int, default=30, help='Seed used to generate the random population')
    register_hyper(algset)

#RBFOpt Parser
def register_rbfopt_options(parser):
    """Add options to the command line parser.

    Register options for RBFOpt optimization algorithm.

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
    docstring = default.__doc__
    param_docstring = docstring[docstring.find('Parameters'):
                                docstring.find('Attributes')].split(' : ')
    param_name = [val.split(' ')[-1].strip() for val in param_docstring[:-1]]
    param_type = [val.split('\n')[0].strip() for val in param_docstring[1:]]
    param_help = [' '.join(line.strip() for line in val.split('\n')[1:-2])
                  for val in param_docstring[1:]]
    # We extract the default from the docstring in case it is
    # necessary, but we use the actual default from the object above.
    # param_default = [val.split(' ')[-1].rstrip('.').strip('\'') for val in
    # param_help]
    for i in range(len(param_name)):
        if (param_type[i] == 'float'):
            type_fun = float
        elif (param_type[i] == 'int'):
            type_fun = int
        elif (param_type[i] == 'bool'):
            type_fun = ast.literal_eval
        else:
            type_fun = str
        algset.add_argument('--' + param_name[i], action='store',
                            dest=param_name[i],
                            type=type_fun,
                            help=param_help[i],
                            default=getattr(default, param_name[i]))

    algset.add_argument('--log', '-o', action='store', metavar='LOG_FILE_NAME',
                        type=str, dest='output_stream', help='Name of log file for output redirection')
    return algset

#RBFMopt Parser
def register_rbfmopt_options(parser):
    algset = register_rbfopt_options(parser)
    register_hyper(algset)
    algset.add_argument('--decomp_method',
                        action='store',
                        type=str,
                        default="tchebycheff",
                        help='Decomposition objectives method. Choose between \'tchebycheff\', \'weighted\', \'bi\'(boundary interception). Default tchebycheff.')

    algset.add_argument('--cycle',
                        action='store',
                        type=int,
                        default=9,
                        help='Cycle number for the algorithm: default is 9.')

    algset.add_argument('--max_filter_mult',
                        action='store',
                        type=int,
                        default=9,
                        help='Multiplier for maximum number of points to re-feed into RBFMopt. So if 9 then max number of points will be 9 * dim of problem. Default is 9')

#RBFOpt Model Parser
def register_rbfopt_model(parser):
    """Add options to the command line parser.

    Register options for reading points from and adding points to RBFOpt's optimization algorithm.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser.

    """
    # Algorithmic settings
    algset = parser.add_argument_group('Model settings')   

    algset.add_argument('--path', action='store', type=str, help='path for files, default is script directory')
    algset.add_argument('--stateFile', action='store', type = str, help = 'file name for algorithm state')
    algset.add_argument('--approximate', action = 'store_true', help = 'approximate the points in the pointFile, otherwise return evaluated points')
    algset.add_argument('--addNodes', action='store_true', help='add the points from the addPointsFile and addValuesFile to the model')
    algset.add_argument('--pointFile', action='store', type = str, help = 'file name for points')
    algset.add_argument('--valueFile', action='store', type = str, help = 'file name for objective values')
    algset.add_argument('--addPointsFile', action='store', type=str, help='file name for points to add to the model')
    algset.add_argument('--addValuesFile', action='store', type=str, help='file name for objective values to add to the model')

#CMA-ES Parser
def register_cmaes_options(parser):
    """Add options to the command line parser.

    Register options for CMA-ES optimization algorithm.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser."""

    # Algorithmic settings
    algset = parser.add_argument_group('Algorithmic settings')

    #Add options for CMA-ES
    algset.add_argument('--initialSolution', action='store', dest = 'initialSolution', type = str, help = 'starting point as list')

    return algset

#Helper Functions
def register_hyper(parser):
    parser.add_argument('--hyper',
                        action='store',
                        type=str2bool,
                        default=False,
                        help='Should hypervolume be written to stream? Default false.')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')