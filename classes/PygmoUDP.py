import numpy as np

class PygmoUDP:
  # So the settings tell us how many function evaluations to do, and the problem is instantiated together with the
  # algorithm
  def __init__(self, dim, var_lower, var_upper, var_type, n_obj, obj_funct):
      self.dim = dim
      self.var_lower = np.ndarray.tolist(var_lower)
      self.var_upper = np.ndarray.tolist(var_upper)
      self.var_type = var_type
      self.obj_funct = obj_funct
      self.n_obj = n_obj

  def fitness(self, x):
    return self.obj_funct(x)

  def get_bounds(self):
    return (self.var_lower, self.var_upper)
  
  def get_nx(self):
    return self.dim
  
  def get_nobj(self):
    return self.n_obj
  
