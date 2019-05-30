# -*- coding: utf-8 -*-
import os
import sys

import numpy as np

import util

DIRCWD=  'D:/_devs/Python01/project27/' if sys.platform.find('win')> -1   else  '/home/ubuntu/notebook/' if os.environ['HOME'].find('ubuntu')>-1 else '/media/sf_project27/'
os.chdir(DIRCWD); sys.path.append(DIRCWD+'/aapackage');  sys.path.append(DIRCWD+'/linux/aapackage')
execfile( DIRCWD + '/aapackage/allmodule.py')
print 'Directory Folder', DIRCWD
#######################################################################################

#--------------------Input Batch Directory -------------------------------------------
DIRBATCH=         DIRCWD+"/linux/batch/task/elvis_prod_20161228/"
input_param_file= DIRBATCH + '/input_params_all.pkl'
print DIRBATCH; print input_param_file




################## Model Parameters ###################################################
masset=9

wwbasket= {'opticriteria': 1.0, 'minperf': 130.0, 'maxvol': 0.5, 'concenfactor': 1.0,
           'wwpenalty':0.4, 'tlagsignal':1, 'rebfreq': 1, 'costbp': 0.0000, 'feesbp': 0.0,
           'maxweight': 0.0, 'schedule1': [], 'costbpshort': 0.0   }

wwbear0= np.array([ 0.0,  0.0,   0.00,  0.0,   0.0,  0.2, 0.0 , 0.2, 0.6 ])



          #Bear Market      #Range Bound   #Bull Market   #Transition Min Trigger, Max Trigger
bounds1=    [ (0.01, 0.5)] * 3 * masset

          #Bear Market      #Range Bound   #Bull Market
bounds2= [          #Transition Min Trigger, Max Trigger
         (0.01, 0.2),   (0.01, 0.05),   (0.01, 0.3),         #  "SPY",     1
         (0.01, 0.2),   (0.001, 0.1),    (0.01, 0.3),        #  "XLF",     2
         (0.01, 0.2),   (0.001, 0.1),    (0.01, 0.3),        #  "XLE",     3
         (0.01, 0.15),  (0.001, 0.05),  (0.01, 0.15),        #  "ZIV",  4
         (0.01, 0.3),   (0.01, 0.3),    (0.01, 0.3),         #  "XLU",     5
         (0.001, 0.1),  (0.001, 0.2),   (0.01, 0.1),         #  "IEF",    6
         (0.01, 0.2),   (0.01, 0.05),   (0.01, 0.3),         #  "HYG",     7
         (0.01, 0.15),  (0.05, 0.2),    (0.001, 0.1),        #  "UGLD",   8
         (0.15, 0.5),   (0.2, 0.5),     (0.1, 0.5),          #  "SHY",        9
        ]

          #Bear Market      #Range Bound   #Bull Market
bounds3= [          #Transition Min Trigger, Max Trigger
         (0.01, 0.2),   (0.01, 0.05),   (0.01, 0.3),         #  "SPY",     1
         (0.01, 0.2),   (0.001, 0.1),    (0.01, 0.3),        #  "XLF",     2
         (0.01, 0.2),   (0.001, 0.1),    (0.01, 0.3),        #  "XLE",     3
         (0.01, 0.15),  (0.001, 0.05),  (0.01, 0.15),        #  "ZIV",  4
         (0.01, 0.3),   (0.01, 0.3),    (0.01, 0.3),         #  "XLU",     5
         (0.001, 0.2),  (0.001, 0.3),   (0.01, 0.1),         #  "IEF",    6
         (0.01, 0.2),   (0.01, 0.05),   (0.01, 0.3),         #  "HYG",     7
         (0.01, 0.15),  (0.05, 0.2),    (0.001, 0.1),        #  "UGLD",   8
         (0.15, 0.5),   (0.2, 0.6),     (0.1, 0.5),          #  "SHY",        9
        ]



################### Create Params List 0 ##############################################################
'''
ldates= [ (20120420, 20161223)  ]
lbound=    [  bounds1, bounds2, bounds3  ]
lminperf=  100 + 4*np.array([9.0, 20.0, 25.0    ])          # PerfYear * nbYears
lopticriteria= [ 8.0, 0.0, 5.0           ]
lmaxvol= [ 0.08,  0.12,  0.40 ]
lconcen= np.array([ 5.0, 10.0,  15, 20, 40.0 ] )  * masset * 3

'''

################### Create Params List 1  ##############################################################
'''
ldates= [ (20120420, 20161223)  ]
lbound=    [  bounds2 ]
lminperf=  [  180.0, 190.0, 195.0 ]    # PerfYear * nbYears
lopticriteria= [ 8.0, 5.0, 0.0             ]
lmaxvol= [   0.40 ]                              # lmaxvol
lconcen= np.array([ 40.0 ] )  * masset * 3       # Concentration Factor

'''

################### Create Params List  2   ##############################################################
'''

ldates= [ (20120420, 20161223)  ]
lbound=    [  bounds1 ]
lminperf=  [  180.0, 190.0, 195.0 ]    # PerfYear * nbYears
lopticriteria= [ 8.0  ,   7.0     ]
lmaxvol= [   0.40 ]                              # lmaxvol
lconcen= np.array([ 40.0 ] )  * masset * 3       # Concentration Factor

'''



'''
    if   crit==0.0 :  obj1= -(bsk -100)**2 / var1 + penalty
    elif crit==1.0 :  obj1= -(bsk- 100)**2 + penalty
    elif crit==2.0 :  obj1= -(bsk -100) / var1 + penalty
    elif crit==3.0 :  obj1= -(bsk -100) / math.sqrt(var1) + penalty
    elif crit==4.0 :  obj1= -(bsk -100)**2 / (maxdd*maxdd * var1) + penalty
    elif crit==5.0 :  obj1= -(bsk -100)**2 / (maxdd * math.sqrt(var1)  ) + penalty
    elif crit==6.0 :  obj1= -(bsk -100)**2 / (avgdd*avgdd  ) + penalty
    elif crit==7.0 :  obj1= -(bsk -100) / (maxdd * math.sqrt(var1)  ) + penalty
    elif crit==8.0 :  obj1= -(bsk -100)**2 / var1  -(bsk -100)**2 / (maxdd * math.sqrt(var1)  )  + 2*penalty

'''




################# Generate 1 Big List of all params ##################################################
input_params_all= []; ii=0
for tt in ldates:
 for bounds in lbound :
  for concenfactor in lconcen  :
   for opticriteria  in  lopticriteria :
    for  maxvol in lmaxvol :
     for minperf in lminperf  :


                  niter= 1       #
                  pop=   18        #
                  ievolve= 5      #
                  nisland= 1       #
                  krepeat= 2

                  ii+=1
                  vv= [ii,{'datei':tt, 'tt0': tt[0], 'tt1': tt[1], "bounds":bounds, "concenfactor":concenfactor,
                     'opticriteria':opticriteria, 'maxvol':maxvol, 'minperf':minperf, 'wwbear0': wwbear0,


                  'niter':niter, 'pop':pop, 'ievolve':ievolve, 'nisland': nisland, 'krepeat':krepeat }]

                  input_params_all.append(vv)

ntask= ii
print('ntask', ntask)
input_params_all= np.array(input_params_all, dtype= np.object)
util.py_save_obj(input_params_all, input_param_file, isabsolutpath=1)




####### SET Nb of CPU for EC2   ###########################################################################
txt_config='''
NCPU=  1;

'''
util.os_print_tofile(txt_config, DIRBATCH+'/ec2_config.py',  mode1='w+')
execfile( DIRBATCH+'/ec2_config.py')
print  '\n', 'Nb CPU: ', NCPU













##################################################################################################
# batch

'''
###Local Linux Full Batch Launcher


ipython /media/sf_project27/linux/batch/task/elvis_prod_20161228/batch_launcher_02.py


If no CPU available, one process wont be launched



'''
