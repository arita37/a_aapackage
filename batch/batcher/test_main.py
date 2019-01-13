import task_parallel as tpar
import os

#todo_folder= "c:/My_todoFolder/"
todo_folder="testing" + os.sep
tpar.execute(todo_folder, startime="2017-01-02-15:25")


'''
Launch an independant sub-process :
   sub-process wait 2017-01-02 15:25
   and execute the list of tasks in parallel as task_parallel


'''








