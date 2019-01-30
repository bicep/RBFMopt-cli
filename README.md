## What is RBFMopt?
RBFMopt is a novel multiobjective version of a blackbox optimization algorithm called RBFOpt (https://github.com/coin-or/rbfopt).

The RBFMopt algorithm is used in parametric design software Grasshopper to find the multiobjective optimal (along a pareto front) of a blackbox.  A blackbox here refers to a function that we do not know the mathematical definition of. Lots of Grasshopper functions that calculate say glare, windflow are essentially so complex they are treated as blackboxes. RBFMopt helps to find optimal options quickly.

RBFMOpt-cli is the command line interface that allows C# in Grasshopper to call the the optimization algorithm.

## Setting up
Using Miniconda, create a new environment:

    conda create --name <env> --file requirements.txt

Using Pip, install rbfopt:

    pip install rbfopt

Get hold of the RBFMopt (private repository for now) module, and within that directory, install the module:

    pip install -e .
