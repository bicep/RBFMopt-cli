import rbfopt as rbfopt
import rbfopt.rbfopt_utils as rbfopt_utils
import numpy as np


#Get the model's bounds, evaluated points and values from a surrogate model
#and write them to a file
def getPoints(pointFilePath, valueFilePath, model):

        points = model.node_pos
        lowerBounds = model.l_lower
        upperBounds = model.l_upper
        boundsAndPoints = [lowerBounds, upperBounds]
        boundsAndPoints.extend(model.node_pos)
        writeFile(pointFilePath, boundsAndPoints)
        writeFile(valueFilePath, model.node_val)

#Read points and values from a file
def readPoints(pointFilePath, valueFilePath):

	#Read the points and values to add
    points = np.loadtxt(pointFilePath, delimiter = " ", ndmin = 2)
    values = np.loadtxt(valueFilePath, delimiter = " ", ndmin = 2)
    
    #Check that list is complete
    assert len(points) == len(values), "%d points, but %d values" % (len(points), len(values))
    
    #Sort according to objective (minimization)
    #values, points = zip(*sorted(zip(values, points)), key=lambda pair: pair[0])
    
    points = np.array(points)
    values = np.array(values)
    
    return points, values

#Adds points and values to the surrogate model
def addPoints(pointFilePath, valueFilePath, model):
    points, values = readPoints(pointFilePath, valueFilePath)
    
	#Check that points have correct dimensionality
    assert len(points[0]) == len(model.l_lower), "Model has %d dimensions, but point file has %d" % (len(model.l_lower), len(points[0]))

    #Scale points
    points_scaled = rbfopt_utils.bulk_transform_domain(model.l_settings, model.l_lower, model.l_upper, points)
    
    #Add the points (last entry is for fast evals)
    #add_node(self, point, orig_point, value, is_fast)
    for i in range(len(points)):
        #Distance test
        if (rbfopt_utils.get_min_distance(points[i], model.all_node_pos) > model.l_settings.min_dist):
            model.add_node(points_scaled[i], points[i], values[i])
        else:
            print ("Point too close to add to model!")

#Read points from a file, 
#evaluate them based one the surrogate model (in RbfoptAlgorithm.evaluateRBF),
#and write the results to a file
def evaluatePoints(pointFilePath, valueFilePath, model):
    #Get the points to evaluate
    points = np.loadtxt(pointFilePath, delimiter = " ", ndmin = 2)
    
    #Check that points have correct dimensionality
    assert len(points[0]) == len(model.l_lower), "Model has %d dimensions, but point file has %d" % (len(model.l_lower), len(points[0]))
    
    #size = 1000000
    #var1 = [random.uniform(-5, 0) for x in xrange(size)]
    #var2 = [random.uniform(10, 15) for x in xrange(size)]
    #points = zip(var1, var2)
    
    #timer = time.clock    
    #start = timer()
   
    values = evaluateRBF((points, model))

    #end = timer()
    #print(str(end - start))
    
    #Combine the results and write them to a file
    
    writeFile(valueFilePath, values)

#Evaluates a set points based one a given surrogate model
#(with a single input object for multiprocessing)
def evaluateRBF(input):
    #Single input for parallel processing
    points, model = input

    # Write error message
    if len(points) == 0:
        sys.stdout.write("No points to evaluate!\n")  
        sys.stdout.flush()
    
    # Number of nodes at current iterbfopt_algtion
    k = len(model.node_pos)

    # Compute indices of fast node evaluations (sparse format)
    fast_node_index = (np.nonzero(model.node_is_fast)[0] if model.two_phase_optimization else np.array([]))
            
    # Rescale nodes if necessary
    tfv = rbfopt_utils.transform_function_values(model.l_settings, np.array(model.node_val), model.fmin, model.fmax, fast_node_index)
    (scaled_node_val, scaled_fmin, scaled_fmax, node_err_bounds, rescale_function) = tfv

    # Compute the matrices necessary for the algorithm
    Amat = rbfopt_utils.get_rbf_matrix(model.l_settings, model.n, k, np.array(model.node_pos))

    # Get coefficients for the exact RBF
    rc = rbfopt_utils.get_rbf_coefficients(model.l_settings, model.n, k, Amat, scaled_node_val)
    if (fast_node_index):
        # RBF with some fast function evaluations
        rc = aux.get_noisy_rbf_coefficients(model.l_settings, model.n, k, Amat[:k, :k], Amat[:k, k:], scaled_node_val, fast_node_index, node_err_bounds, rc[0], rc[1])
    (rbf_l, rbf_h) = rc
    
    # Evaluate RBF
    if len(points) <= 1:
        values = []    
        for point in points: values.append(rbfopt_util.evaluate_rbf(model.l_settings, point, model.n, k, np.array(model.node_pos), rbf_l, rbf_h))
        return values
    else: 
        return rbfopt_utils.bulk_evaluate_rbf(model.l_settings, np.array(points), model.n, k, np.array(model.node_pos), rbf_l, rbf_h)
    
#Write a list, or list of list, to a file
def writeFile(file, elements):

    with open(file, 'w') as file:
        for element in elements:
            if isinstance(element, (list, np.ndarray)):
                string = " ".join(str(value) for value in element)
                file.write(string + "\n")
            elif isinstance(element, (float, np.float64)):
                string = str(element)
                file.write(string + "\n")
            else: 
                print("Element " + str(type(element)) + " not recognized!")