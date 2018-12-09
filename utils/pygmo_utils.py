import pygmo as pg


# Calculates the hypervolume with a changing ref point
def calculate_hv(hv_pop):

    # Have the same ref point at the beginning, and compute the starting hypervolume
    new_hv = pg.hypervolume(hv_pop)
    ref = new_hv.refpoint(offset=4.0)
    return new_hv.compute(ref)
