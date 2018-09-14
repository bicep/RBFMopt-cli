import pygmo as pg
import rbfopt as rbfopt
from classes.PygmoSingleProblemWrapper import PygmoSingleProblemWrapper

class RbfoptWrapper(pg.algorithm):

  # So the settings tell us how many function evaluations to do, and the problem is instantiated together with the
  # algorithm
  def __init__(self, settings, problem):
    self.settings = settings
    self.problem = PygmoSingleProblemWrapper(problem)
    self.algorithm = rbfopt.RbfoptAlgorithm(settings, self.problem)

  def evolve(self):
    val, x, itercount, evalcount, fast_evalcount = self.algorithm.optimize()
    return val, x, itercount, evalcount, fast_evalcount

  def get_champions(self):
    return self.problem.get_champions()