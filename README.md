## What is RBFMopt?
RBFMopt is a novel multiobjective version of a blackbox optimization algorithm called RBFOpt (https://github.com/coin-or/rbfopt).

It is used in parametric design software Grasshopper to find the pareto curve of a blackbox, or a function that we do not know the mathematical definition of. 

RBFMOpt is the command line interface that allows C# in Grasshopper to call the the optimization algorithm.

## Setting up
Using Miniconda, create a new environment:

    conda create --name <env> --file requirements.txt

Using Pip, install rbfopt:

    pip install rbfopt

Get hold of the RBFMopt (private repository for now) module, and within that directory, install the module:

    pip install -e .
