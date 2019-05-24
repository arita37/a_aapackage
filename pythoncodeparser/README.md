# pythoncodeparser

This Python code parser fetches variable information from the given source

The source may be given either as a .py filepath or as a directory path
containing multiple .py source files

The output is written to standard output in CSV format with the following
columns; it starts with a header row:
filepath,function\_or\_class\_name,variable\_name,is\_local

Function\_or\_class\_name defaults to (global) when there's no enclosing
function nor class

### Example:

**pythoncodeparser$ ./main.py src/file_finder.py**

```
filepath,function_or_class_name,variable_name,is_local
src/file_finder.py,findVariablesInFile,filepath,False
src/file_finder.py,findVariablesInDir,findVariablesInFile,False
src/file_finder.py,_walk,args,False
src/file_finder.py,_walk,f,True
src/file_finder.py,_walk,root,True
src/file_finder.py,_walk,files,True
src/file_finder.py,findVariablesInDir,filepath,True
src/file_finder.py,_walk,os,False
src/file_finder.py,_walk,dirs,True
src/file_finder.py,_walk,include_hidden,False
src/file_finder.py,findVariablesInFile,source,True
src/file_finder.py,findVariablesInDir,ret,True
src/file_finder.py,findVariablesInDir,_,True
src/file_finder.py,findVariablesInDir,_walk,False
src/file_finder.py,_walk,d,True
src/file_finder.py,findVariablesInDir,filename,True
src/file_finder.py,findVariablesInFile,is_local,True
src/file_finder.py,findVariablesInDir,set,False
src/file_finder.py,findVariablesInFile,variable_name,True
src/file_finder.py,findVariablesInFile,set,False
src/file_finder.py,findVariablesInDir,directory,False
src/file_finder.py,_walk,ValueError,False
src/file_finder.py,findVariablesInFile,open,False
src/file_finder.py,findVariablesInDir,root,True
src/file_finder.py,findVariablesInDir,files,True
src/file_finder.py,_walk,exclude_prefixes,True
src/file_finder.py,_walk,kwargs,False
src/file_finder.py,findVariablesInFile,ret,True
src/file_finder.py,findVariablesInFile,ast_analyzer,False
src/file_finder.py,findVariablesInDir,os,False
src/file_finder.py,findVariablesInFile,source_file,True
src/file_finder.py,findVariablesInFile,function_or_class,True
src/file_finder.py,findVariablesInFile,variables,True
```

### Run all unittests
**pythoncodeparser$ python3 -m unittest discover -p '*_test.py'**
