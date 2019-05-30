"""
from . import batch_sequencer
from functools import partial

options = batch_sequencer.load_arguments()

mandatoryArguments = (
    options.HyperParametersFile,
    options.WorkingDirectory,
    options.OptimizerName,
    options.AdditionalFiles
)


build_execute_batch = batch_sequencer.build_execute_batch

execute_batch = partial(
    build_execute_batch,
    *mandatoryArguments)

"""


import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())
