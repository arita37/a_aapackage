#!/usr/bin/env python3

"""This Python code parser fetches variable information from the given source

The source may be given either as a .py filepath or as a directory path
containing multiple .py source files

The output is written to standard output in CSV format with the following
columns; it starts with a header row:
filepath,function_or_class_name,variable_name,is_local

Function_or_class_name defaults to (global) when there's no enclosing function
nor class
"""


from src import entry


if __name__ == "__main__":
  entry.main()

