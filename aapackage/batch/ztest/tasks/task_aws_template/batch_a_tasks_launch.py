# -*- coding: utf-8 -*-
"""Portfolio Simulation details / simulation """
# %load_ext autoreload
# %autoreload 2
import os
import sys

import numpy as np

import util

##############################################################################################


#### run in AWS servers


# for foldername in working_dir_folder :
#    if "_qdone" not in foldername and "_qstart" not in foldername  :
#
#        os.rename( working_dir + foldername, working_dir + foldername +  "_qstart" )
#        foldername = foldername + "_qstart"
#
#        startime = now()
#        subprocess( working_dir + foldername + / + "main.py" , starttime)


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


"""  Manual test
  ii= 2; kkk=1
  params_dict= dict(input_params_list[ii,1])
  globals().update(params_dict)

  util.os_print_tofile('\n task_' + str(ii) +'\n' , batch_out_log)
  for kkk in xrange(0, krepeat) : # Random Start loop
      execfile(batch_script)

"""


#####################################################################################

""" Create Storage file :
np.concatenate((    ))

aafolioref= aafolio_storage[0,:].reshape(1,20)
util.py_save_obj( (aux3_cols, aafolioref) ,  'aafolio_storage_ref' ) 

aux3_cols, aafolio_storage=  util.py_load_obj( dir_batch_main+  '/batch_20161228_96627894/aafolio_storage_20161227',
                                                   isabsolutpath=1 )

"""

#####################################################################################

"""
for arg in sys.argv:
    print arg
Each command-line argument passed to the program will be in sys.argv, which is just a list. Here you are printing each argument on a separate line.
Example 10.21. The contents of sys.argv
[you@localhost py]$ python argecho.py             1
argecho.py
"""

"""
import argparse
parser.add_argument('-i','--input', help='Script File Name', required=False)
parser.add_argument('-o','--output',help='Script ID', required=False)
args = parser.parse_args()

## show values ##
print ("Input file: %s" % args.input )
print ("Output file: %s" % args.output )
"""
