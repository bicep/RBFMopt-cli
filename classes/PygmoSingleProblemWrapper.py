import pygmo as pg
import rbfopt as rbfopt
import math as math
from classes.PygmoProblemWrapper import PygmoProblemWrapper

class PygmoSingleProblemWrapper(PygmoProblemWrapper):

  def __init__(self, pygmoProblem):
    PygmoProblemWrapper.__init__(self, pygmoProblem)
    self.pygmoProblem = pygmoProblem
    self.champion = math.inf
    self.champions = []
  
  # In this case x is the value of the decision variables (some kind of array)
  def evaluate(self, x):
    # fitness returns the fitness vector as an iterable python object, so we get the zero index
    fitnessValue = self.pygmoProblem.fitness(x)[0]

    # Store the population values so that we can back calculate the hypervolume later
    # x_list and f_list inherited from PygmoProblemWrapper
    self.x_list.append(x)
    self.f_list.append(fitnessValue)

    # if there is a new champion we change this champion
    if (fitnessValue < self.champion):
      self.champion = fitnessValue
    
    # add this to the champions array
    self.champions.append(self.champion)

    return fitnessValue

  def get_champions(self):
    return self.champions
  
