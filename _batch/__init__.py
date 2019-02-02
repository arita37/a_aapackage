#!/bin/python

from . import batch_sequencer
from functools import partial

options = batch_sequencer.load_arguments()

mandatoryArguments = (
    options.HyperParametersFile,
    options.WorkingDirectory,
    options.OptimizerName,
    options.AdditionalFiles
)



execute_batch = partial(
    batch_sequencer.build_execute_batch,
    *mandatoryArguments)
