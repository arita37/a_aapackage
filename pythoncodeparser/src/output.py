import sys


def writeCSV(variables, file_handle=sys.stdout):
    """Writes found variables' data to file_handle in CSV format

  The output is written to standard output by default

  The output starts with a header row:
  filepath,function_or_class_name,variable_name,is_local

  Function_or_class_name defaults to (global) when there's no enclosing function
  nor class

  Args:
    variables: A set of 4-tuples in format provided by the file_finder module
  """
    file_handle.write("filepath,function_or_class_name,variable_name,is_local\n")

    for filepath, function_or_class, variable_name, is_local in variables:
        function_or_class_name = (function_or_class.name if function_or_class is not None else '(global)')

        file_handle.write("%s,%s,%s,%s\n" % (filepath, function_or_class_name, variable_name, is_local))
