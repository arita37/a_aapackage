# -*- coding: utf-8 -*-
"""
Confusion comes there 3 levels of Batchs in my initial code

meta_batch_task.py :
   ...task_folder/task1/mybatch_optim.py  + hyperparamoptim.csv +  myscript.py ...
   ...task_folder/task2/mybatch_optim.py  + hyperparamoptim.csv + .myscript.py ...


meta_batch_task.py :
   for all "task*" folders in task_working_dir  :
       run subprocess taskXXX/mybatch_optim.py



mybatch_optim.py :
  for all rows ii of hyperparams.csv
     run subprocess  myscript.py  ii
     check CPU suage with psutil.   CPU usage < 90 %   mem usage < 90%


### not needed from you
aws_batch_script.py :
   for all tasks folder in LOCAL PC :
       transfer by SFTP to REMOTE Task folder (by zip)

   subprocess  meta=batch_task.py on REMOTE PC.    


##### Warning :
  Please check my initial code to make sure most of functionnalities are here in your code


####################################################################
batch_sequencer.py :
  Launch 1 subprcoess per hyperparam row.


"""

import numpy as np
import os
import shutil
import sys
import random

import toml
import subprocess
import optparse
import pandas as pd


################### Argument catching ########################################################
print('Start Args')
'''
print sys.argv, "\n\n", sys.argv[0], "\n\n", sys.argv[1], "\n\n"
'''


def load_arguments():
    parser = optparse.OptionParser()
    parser.add_option("--hyperparam", dest="HyperParametersFile",
                      help="Select the path for a .csv containing batch optimization parameters.\n One row per optimization.")

    parser.add_option("--directory", dest="WorkingDirectory", default=".",
                      help="Absolute or relative path to the working directory.")

    parser.add_option("--optimizer", dest="OptimizerName",
                      default="optimizer.py",
                      help="Name of the optimizer script. Should be located at WorkingDirectory")

    parser.add_option("--file", dest="AdditionalFiles",
                      help="A file or comma-separated list of files to be provided for the optimizer function.")

    parser.add_option("--aws", dest="ExecuteAws", action="store_true",
                      help="Wether optimization will run locally or on a dedicated AWS instance.")


    options, args = parser.parse_args()
    return options


################### Batch Params ############################################################
"""
date0 = util.date_now()
isverbose = 0
batchname = task_folder
DIRBATCH = DIRCWD + '/linux/batch/task/' + task_folder + '/'

batch_script = DIRBATCH + '/elvis_pygmo_optim.py'
batch_in_data1 = DIRBATCH + '/aafolio_storage_ref.pkl'
filedata = DIRBATCH + '/close_15assets_2012_dec216.spydata'


dir_batch_main = DIRCWD + '/linux/batch/'
# str(np.random.randint(1000, 99999999))
batch_out_name = 'batch_' + util.date_nowtime('stamp')
batch_out = dir_batch_main + '/output/'+date0 + '/' + batch_out_name
os.makedirs(batch_out)

batch_out_log = batch_out + '/output_result.txt'
batch_out_data = batch_out + '/aafolio_storage_' + date0 + '.pkl'
util.os_print_tofile('\n\n'+title1, batch_out_log)
"""

def checkAdditionalFiles(WorkingDirectory, AdditionalFiles):
    if not AdditionalFiles:
        return []

    AdditionalFiles = [f.strip(" ")
                       for f in AdditionalFiles.split(",")]

    for File in AdditionalFiles:
        ExpectedFilePath = os.path.isfile(os.path.join(WorkingDirectory, File))
        if not ExpectedFilePath:
            print("Additional file <%s> %s not found. Aborting!" % File)
            exit(1)

    return AdditionalFiles



################  Output Data table ######################################################
"""
if util.os_file_exist(batch_out_data):
    aux3_cols, aafolio_storage = util.py_load_obj(
        batch_out_data, isabsolutpath=1)
else:
    aux3_cols, aafolio_storage = util.py_load_obj(
        batch_in_data1, isabsolutpath=1)


"""
def logs() :
        with open(batch_log_file, 'a') as batch_log:
            batch_log.write("Executing index %i at %s." % (ii, TaskDirectory))
            batch_log.write("\n\n")




############### Loop on each parameters sets #############################################
def execute_batch(HyperParametersFile, WorkingDirectory,
                  ScriptPath, AdditionalFilesArg, krepeat=1):

    HyperParameters = pd.read_csv(os.path.join(WorkingDirectory, HyperParametersFile))

    batch_label = "%s_%.3i" % (OptimizerName, random.randint(0, 10e5))

    batch_log_file = os.path.join(WorkingDirectory, "log_batch_%s.txt" % batch_label)

    ChildProcesses = []
    for ii in range(HyperParameters.shape[0]):

        # Extract parameters for single run from batch_parameters data.
        params_dict = HyperParameters.iloc[ii].to_dict()


        """
        No Need t 
        # build path & create task directory;
        TaskDirectoryName = "task_%s_%i" % (batch_label, ii)
        TaskDirectory = os.path.join(WorkingDirectory, TaskDirectoryName)

        os.mkdir(TaskDirectory)

        # Copy optimization script to Task Directory;
        shutil.copy(os.path.join(WorkingDirectory, OptimizerName),
                    os.path.join(TaskDirectory, OptimizerName))

        # Save Parameters inside Task Directory;
        with open(os.path.join(TaskDirectory, "parameters.toml"), "w") as task_parameters:
            toml.dump(params_dict, task_parameters)

        
        # Copy additional files to Task Directory;
        if AdditionalFilesArg:
            AdditionalFiles = checkAdditionalFiles(WorkingDirectory, AdditionalFilesArg)
            for File in AdditionalFiles:
                shutil.copy(os.path.join(WorkingDirectory, File),
                            os.path.join(TaskDirectory, File))
        """


        python_path = os.python_path

        # Random Start loop  krepeat to have many random start....
        for rr in range(krepeat):
            # optimizerScriptPath = os.path.join(TaskDirectory, OptimizerName)
              

            proc = subprocess.Popen([python_path, ScriptPath, ii],
                                    stdout=subprocess.PIPE)
            ChildProcesses.append(proc)
            wait( waitseconds)




'''  Manual test
  ii= 2; kkk=1
  params_dict= dict(input_params_list[ii,1])
  globals().update(params_dict)

  util.os_print_tofile('\n task_' + str(ii) +'\n' , batch_out_log)
  for kkk in xrange(0, krepeat) : # Random Start loop
      execfile(batch_script)

'''


#####################################################################################

''' Create Storage file :
np.concatenate((    ))

aafolioref= aafolio_storage[0,:].reshape(1,20)
util.py_save_obj( (aux3_cols, aafolioref) ,  'aafolio_storage_ref' ) 

aux3_cols, aafolio_storage=  util.py_load_obj( dir_batch_main+  '/batch_20161228_96627894/aafolio_storage_20161227',
                                                   isabsolutpath=1 )

'''

#####################################################################################

'''
for arg in sys.argv:
    print arg
Each command-line argument passed to the program will be in sys.argv, which is just a list. Here you are printing each argument on a separate line.
Example 10.21. The contents of sys.argv
[you@localhost py]$ python argecho.py             1
argecho.py
'''

'''
import argparse
parser.add_argument('-i','--input', help='Script File Name', required=False)
parser.add_argument('-o','--output',help='Script ID', required=False)
args = parser.parse_args()

## show values ##
print ("Input file: %s" % args.input )
print ("Output file: %s" % args.output )
'''
