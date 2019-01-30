#!/bin/python
"""

Test to run _batch module:

might need arguments to run this:

$python ztest_batch.py --hyperparam hyperparams.csv --optimizer optimizer.py --directory _batch



"""
import _batch

_batch.execute_batch(krepeat=1)
