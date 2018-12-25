# Why is a global_record necessary? 
# PygmoProblemWrapper object calculates the hypervolume and then adds it to array
# Ideally it would pass the hypervolume (as parameters) onto read_write_obj_func to display on output stream
# However read_write_obj_func only accepts one argument (x, the decision variable parameter)
hv_array = []
hv_bool = False
