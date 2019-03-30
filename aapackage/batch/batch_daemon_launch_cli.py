# -*- coding: utf-8 -*-
'''
  Daemon monitoring batch

  scans sub-folder in /tasks/
  and execute  /tasks/taskXXXX/main.py in that folder  (if task is not qdone)
  When task is finished, taskXXX is rename to taskXXX_qdone



batch_daemon_launch_cli.py  --task_folder  tasks  --log_file   zlog/batchdaemong.log  --mode nodaemon  --waitsec 10  &



batch_daemon_launch_cli.py  --task_folder  tasks  --log_file   zlog/batchdaemong.log  --mode nodaemon  --waitsec 10  & batch_daemon_monitor_cli.py --monitor_log_folder   tasks_out/   --monitor_log_file monitor_log_file.log   --log_file   zlog/batchdaemon_monitor.log    --mode daemon     





'''
import os
import sys
from time import sleep
import argparse
import logging

################################################################################
from aapackage import util_log
from aapackage.batch import util_batch
from aapackage.batch import util_cpu


############### Variable definition ############################################
#DIR_PATH = os.path.dirname(os.path.realpath(__file__))
TASK_FOLDER_DEFAULT = os.path.dirname(os.path.realpath(__file__)) + "/ztestasks/"
#TASK_FOLDER_DEFAULT = os.getcwd()
logger = logging.basicConfig()


################################################################################
def log(*argv):
  logger.info(",".join([str(x) for x in argv]))


def load_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task_folder", default=TASK_FOLDER_DEFAULT, help="Absolute or relative path to the working folder.")
  parser.add_argument("--log_file", default="logfile_batchdaemon.log", help=".")
  parser.add_argument("--log_file_task", default="logfile_batchdaemon_task.log", help=".")
  parser.add_argument("--mode", default="nodaemon", help="daemon/ .")
  parser.add_argument("--waitsec", type=int, default=30, help="wait sec")

  options = parser.parse_args()
  return options


def get_list_valid_task_folder(folder, script_name="main.py"):
  if not os.path.isdir(folder):
    return []
  valid_folders = []
  for root, dirs, files in os.walk(folder):
    root_splits = root.split("/")
    for filename in files:
      if filename == script_name and \
          "_qstart" not in root_splits[-1] and \
          "_qdone" not in root_splits[-1] and \
          "_ignore" not in root_splits[-1]:
        valid_folders.append(root)

  return valid_folders










################################################################################
################################################################################
if __name__ == '__main__':
  args = load_arguments()
  logger = util_log.logger_setup(__name__,
                                 log_file=args.log_file,
                                 formatter=util_log.FORMATTER_4,
                                 isrotate=True)

  log("Daemon","start ", os.getpid())
  while True:
    log("Daemon new loop", args.task_folder)
    folders = get_list_valid_task_folder(args.task_folder)

    if len(folders) > 0:
      log("task folder:", folders)
      pid_list = util_batch.batch_run_infolder(task_folders=folders,
                                               log_file= args.log_file)
      log("task folder started:", pid_list)

    if args.mode != "daemon":
      log("Daemon","terminated", os.getpid())
      break

    sleep(args.waitsec)


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
