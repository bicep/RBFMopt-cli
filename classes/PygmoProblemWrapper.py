import rbfopt as rbfopt
from utils.rbfmopt_utils import calculate_weighted_objective


class PygmoProblemWrapper(rbfopt.RbfoptBlackBox):

    def __init__(self, pygmoProblem, var_types):
        self.seed = 33
        self.rho = 0.05

        self.pygmoProblem = pygmoProblem
        self.x_list = []
        self.f_list = []
        self.current_weights = []
        self.var_types = var_types

    def set_current_weights(self, wts):
        self.current_weights = wts

    # In this case x is the value of the decision variables (some kind 
    # of array)
    def evaluate(self, x):

        # fitness returns the fitness vector as an iterable python object, so 
        # we get the zero index
        fitnessValue = self.pygmoProblem.fitness(x)
        # Store the population values so that we can back calculate the 
        # hypervolume later
        self.x_list.append(x)
        self.f_list.append(fitnessValue)

        # weighted value to get single fitness value
        # Have to make sure that current weights is set first!!
        weightedSingleFitnessValue = calculate_weighted_objective(
                                                          self.current_weights,
                                                          fitnessValue, 
                                                          self.rho)

        return weightedSingleFitnessValue

    def get_n_obj(self):
        return self.pygmoProblem.get_nobj()

    def get_f_list(self):
        return self.f_list

    def get_x_list(self):
        return self.x_list

    def get_dimension(self):
        return self.pygmoProblem.get_nx()

    def get_var_lower(self):
        return self.pygmoProblem.get_bounds()[0]

    # This is the problematic thing, I don't know how to get the decision 
    # vector
    # decision vector = population?
    def get_var_type(self):
        return self.var_types

    def get_var_upper(self):
        return self.pygmoProblem.get_bounds()[1]

    def evaluate_noisy(self, x):
        return

    def has_evaluate_noisy(self):
        return
