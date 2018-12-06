import pygmo as pg
import rbfopt as rbfopt
from classes.PygmoProblemWrapper import PygmoProblemWrapper
from classes.Rbfmopt import Rbfmopt


class RbfmoptWrapper(pg.algorithm):
    # So the settings tell us how many function evaluations to do, and the 
    # problem is instantiated together with the algorithm
    def __init__(self, dict_settings, problem, var_types, output_stream):

        self.settings = rbfopt.RbfoptSettings.from_dictionary(dict_settings)
        self.problem = PygmoProblemWrapper(problem, var_types)
        self.rbfmoptClass = Rbfmopt(self.settings, self.problem, output_stream)

    def evolve(self):

        self.rbfmoptClass.optimize()

    def get_f_list(self):

        return self.problem.get_f_list()

    def get_x_list(self):

        return self.problem.get_x_list()