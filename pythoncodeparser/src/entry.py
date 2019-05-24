import os
import sys

from src import file_finder
from src import output


def usage(message):
    """Prints the usage description"""
    print(message)
    print(f"""
Usage:
{sys.argv[0]} filepath|directory_path
""")


def main():
    """Entry point"""
    try:
        path = sys.argv[1]
    except IndexError:
        usage("Either a filepath or a directory path must be provided")
        return

    if os.path.isdir(path):
        variables = file_finder.findVariablesInDir(path)
    elif os.path.isfile(path):
        variables = file_finder.findVariablesInFile(path)
    else:
        usage(f"The given path is neither a file nor a directory: {path}")
        return

    output.writeCSV(variables)
