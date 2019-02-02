#!/bin/python
"""

This is a dummy optimizer, designed to be executed in a independend python process.

Notable interactions:

loads the file "parameters.toml", located at its same folder.

outputs "results.txt" file, containing an array that can be sourced inside python.

Those interactions are defined in batch_sequencer.py and should be conserved among all optimizers.


"""
from scipy import optimize
import toml
import os
import sys
import pandas as pd
def optimizerFunction(x):
    x1, x2, x3, x4 = x
    delta = x1 ** 2 * x4
    omega = -500 + delta * x1 + x2

    return omega


def execute():
    ii = int(sys.argv[1])
    
    os.chdir(os.path.dirname(os.path.realpath(__file__)))


    HyperParameterSet = pd.read_csv("hyperparams.csv")
    params_dict = HyperParameterSet.iloc[ii].to_dict()

    res = optimize.minimize(optimizerFunction, [
        params_dict["x1"],
        params_dict["x2"],
        params_dict["x3"],
        params_dict["x4"]
    ])

    print("Result: %s" % res.x)

    with open("batch_results/result%i.txt" % ii, 'a') as ResultOutput:
        result = str(list(res.x))
        ResultOutput.write(result + "\n")

    print("Finished.", file=sys.stderr)


if __name__ == "__main__":
    execute()
