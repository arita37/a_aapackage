# -*- coding: utf-8 -*-
'''Daemon monitoring batch '''
import os, sys

import logging
import socket
from time import sleep

###############################################################################
import util_log
import util_batch

############### Variable definition ###########################################
logging.basicConfig(level=logging.INFO)
# APP_ID   = __file__ + ',' + str(os.getpid()) + ',' + str(socket.gethostname())
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
WORKING_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

util_log.APP_ID = __file__ + ',' + str(os.getpid()) + ',' + str(socket.gethostname())
util_log.LOG_FILE = "logfile.log"


######### Logging ##############################################################
def load_arguments():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--hyperparam", help="Select path for a .csv  HyperParameters ")
    parser.add_argument("--directory", default=".", help="Absolute or relative path to the working directory.")
    parser.add_argument("--subprocess_script", default="optimizer.py", help="Name of the optimizer script")

    parser.add_argument("--file_log", default="ztest/logfile_batchdaemon.log", help=".")
    parser.add_argument("--file_error_log", default="ztest/log_batchdaemon_error.log", help=".")

    options = parser.parse_args()
    return options


def log(message):
    util_log.printlog(s1=message)


def get_list_of_valid_directories():
    valid_directoties = []
    for root, dirs, files in os.walk(DIR_PATH):
        for filename in files:
            root_splits = root.split("/")
            if root_splits[-2] == "tasks" and not root_splits[-1].endswith("_qstart") and \
                    not root_splits[-1].endswith("_qdone") and filename == "main.py":
                valid_directoties.append(root)
    return valid_directoties


def run_files_in_directory(valid_directoties):
    sub_process_list = []
    for each_dir in valid_directoties:
        foldername = each_dir + "_qstart"
        util_batch.rename_directory(old_directory=each_dir, new_directory=foldername)
        main_file = os.path.join(foldername, "main.py")
        log("running file: %s" % main_file)
        ps = util_batch.execute_script(script=main_file)
        sub_process_list.append(ps.pid)
        log("running process: %s" % str(ps.pid))
        sleep(5)
    util_batch.wait_for_completion(sub_process_list)

if __name__ == '__main__':
    args = load_arguments()
    util_log.LOG_FILE = args.file_log

    log("Current Process Id: %s" % (str(os.getpid())))
    valid_directoties = get_list_of_valid_directories()
    if len(valid_directoties) > 0:
        log("valid direcoties: %s" % str(valid_directoties))
        run_files_in_directory(valid_directoties=valid_directoties)

    log("Process Completed")

"""
################### Argument catching ########################################################
print('Start Args')
xprint sys.argv, "\n\n", sys.argv[0], "\n\n", sys.argv[1], "\n\n"
title1= sys.argv[0]
task_folder= title1.split('/')[-2]            #  print 'Task Folder', task_folder

#Task data, str  --> Tuple of arguments
args1=  eval(sys.argv[1])
input_param_file, itask0, itask1= args1[0]

#Other data
args2= args1[1]

print(title1, input_param_file, (itask0, itask1), args2 )


################### Batch Params #############################################################
date0= util.date_now()
isverbose =0
batchname=        task_folder
DIRBATCH=         DIRCWD + '/linux/batch/task/'+ task_folder + '/'

batch_script=     DIRBATCH + '/elvis_pygmo_optim.py'
batch_in_data1=   DIRBATCH + '/aafolio_storage_ref.pkl'
filedata=         DIRBATCH + '/close_15assets_2012_dec216.spydata'



dir_batch_main=   DIRCWD + '/linux/batch/'
batch_out_name=   'batch_' + util.date_nowtime('stamp')   #str(np.random.randint(1000, 99999999))
batch_out=        dir_batch_main + '/output/'+date0 + '/'+ batch_out_name
os.makedirs(batch_out)

batch_out_log=    batch_out + '/output_result.txt'
batch_out_data=   batch_out + '/aafolio_storage_' + date0 + '.pkl'
util.os_print_tofile( '\n\n'+title1, batch_out_log) 



################  Output Data table ######################################################
if util.os_file_exist(batch_out_data):          
  aux3_cols, aafolio_storage=  util.py_load_obj( batch_out_data, isabsolutpath=1 )
else :
  aux3_cols, aafolio_storage=  util.py_load_obj( batch_in_data1, isabsolutpath=1 )  



################## Model Parameters ######################################################
# input_param_file=    DIRBATCH_local+'input_params_all.pkl'
input_params_list=   util.py_load_obj(input_param_file, isabsolutpath=1)
input_params_list=   input_params_list[itask0:itask1,:]    #Reduce the size



############### Loop on each parameters sets #############################################
iibatch=0; krepeat= 1;ii=0
for ii in xrange(itask0-itask0, itask1-itask0):
  iibatch+=1
  params_dict= dict(input_params_list[ii,1])
  globals().update(params_dict)     #Update in the interpreter memory

  util.os_print_tofile('\n ' + task_folder + ' , task_' + str(ii+itask0) + '\n', batch_out_log)
  for kkk in xrange(0, krepeat) :   # Random Start loop  krepeat to have many random start....
      execfile(batch_script)


"""

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
