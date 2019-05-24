import os

from src import ast_analyzer


def findVariablesInFile(filepath):
    """Lists variables parsed from the given file

  Args:
    filepath: A path to a file containing Python code to be parsed

  Returns:
    A set of 4-tuples describing found variables
    {(str file,
      ast.stmt function_or_class,
      str variable,
      bool is_local), ...}
  """
    ret = set()

    with open(filepath, "r") as source_file:
        source = source_file.read()
        variables = ast_analyzer.analyzeSource(source)

        for ((function_or_class, variable_name), is_local) in variables.items():
            ret.add((filepath, function_or_class, variable_name, is_local))

    return ret


def findVariablesInDir(directory):
    """Lists variables parsed from the given directory

  Args:
    directory: A path to a directory containing Python code to be parsed

  Returns:
    A set of 4-tuples describing found variables
    {(str file,
      ast.stmt function_or_class,
      str variable,
      bool is_local), ...}
  """
    ret = set()

    for root, _, files in _walk(directory, onerror=_onerror_reraise, include_hidden=False):
        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(root, filename)
                ret |= findVariablesInFile(filepath)

    return ret


def _onerror_reraise(e):
    raise e


def _walk(*args, include_hidden=None, **kwargs):
    """A thin wrapper over os.walk which lists only non-hidden files on demand"""
    if include_hidden is None:
        raise ValueError("include_hidden must be specified")

    exclude_prefixes = () if include_hidden else (".")

    for root, dirs, files in os.walk(*args, **kwargs):
        files = [f for f in files if not f.startswith(exclude_prefixes)]
        dirs[:] = [d for d in dirs if not d.startswith(exclude_prefixes)]

        yield root, dirs, files
