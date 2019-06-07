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



################## Model Parameters ###################################################
masset=9
nbregime= 3

wwbasket= {'opticriteria': 1.0, 'minperf': 130.0, 'maxvol': 0.5, 'concenfactor': 1.0,
           'wwpenalty':0.4, 'tlagsignal':1, 'rebfreq': 1, 'costbp': 0.0000, 'feesbp': 0.0,
           'maxweight': 0.0, 'schedule1': [], 'costbpshort': 0.0   }

wwbear0= np.array([ 0.0,  0.0,   0.00,  0.0,   0.0,  0.2, 0.0 , 0.2, 0.6 ])



          #Bear Market      #Range Bound   #Bull Market   #Transition Min Trigger, Max Trigger
bounds1=    [ (0.01, 0.5)] * 3 * masset

          #Bear Market      #Range Bound   #Bull Market
bounds2= [          #Transition Min Trigger, Max Trigger
         (0.01, 0.25),   (0.01, 0.1),   (0.01, 0.3),         #  "SPY",     1
         (0.01, 0.25),   (0.001, 0.1),  (0.01, 0.3),        #  "XLF",     2
         (0.01, 0.25),   (0.001, 0.1),  (0.01, 0.3),        #  "XLE",     3
         (0.01, 0.25),  (0.001, 0.1),   (0.01, 0.3),        #  "ZIV",  4
         (0.01, 0.3),   (0.01, 0.3),    (0.01, 0.3),         #  "XLU",     5
         (0.001, 0.2),  (0.001, 0.3),   (0.01, 0.1),         #  "IEF",    6
         (0.01, 0.2),   (0.01, 0.05),   (0.01, 0.3),         #  "HYG",     7
         (0.01, 0.15),  (0.05, 0.2),    (0.001, 0.1),        #  "UGLD",   8
         (0.15, 0.5),   (0.2, 0.5),     (0.1, 0.5),          #  "SHY",        9
        ]

          #Bear Market      #Range Bound   #Bull Market
bounds3= [          #Transition Min Trigger, Max Trigger
         (0.01, 0.2),   (0.01, 0.05),    (0.01, 0.3),         #  "SPY",     1
         (0.01, 0.2),   (0.001, 0.05),   (0.01, 0.3),        #  "XLF",     2
         (0.01, 0.2),   (0.001, 0.05),   (0.01, 0.3),        #  "XLE",     3
         (0.01, 0.2),  (0.001, 0.05),    (0.01, 0.15),        #  "ZIV",  4
         (0.01, 0.3),   (0.01, 0.3),     (0.01, 0.3),         #  "XLU",     5
         (0.001, 0.1),  (0.001, 0.2),    (0.01, 0.08),         #  "IEF",    6
         (0.01, 0.2),   (0.01, 0.1),     (0.01, 0.3),         #  "HYG",     7
         (0.01, 0.15),  (0.05, 0.2),     (0.001, 0.1),        #  "UGLD",   8
         (0.15, 0.5),   (0.2, 0.5),      (0.1, 0.5),          #  "SHY",        9
        ]

          #Bear Market      #Range Bound   #Bull Market
bounds4= [          #Transition Min Trigger, Max Trigger
         (0.01, 0.2),   (0.01, 0.05),    (0.02, 0.3),         #  "SPY",     1
         (0.01, 0.2),   (0.001, 0.05),   (0.02, 0.3),        #  "XLF",     2
         (0.01, 0.2),   (0.001, 0.05),   (0.02, 0.3),        #  "XLE",     3
         (0.01, 0.2),  (0.001, 0.05),    (0.02, 0.3),        #  "ZIV",  4
         (0.01, 0.3),   (0.01, 0.3),     (0.01, 0.3),         #  "XLU",     5
         (0.001, 0.1),  (0.001, 0.2),    (0.001, 0.05),         #  "IEF",    6
         (0.01, 0.2),   (0.01, 0.1),     (0.001, 0.3),         #  "HYG",     7
         (0.01, 0.15),  (0.05, 0.3),     (0.001, 0.1),        #  "UGLD",   8
         (0.15, 0.6),   (0.2, 0.6),      (0.15, 0.6),          #  "SHY",        9
        ]


#hard Penalty on weight
wwpenalty4= np.array([ x[1] for x in bounds4 ]).reshape((masset, nbregime))

wwmat= np.array([[ 0.02315412,  0.07024028,  0.02367605],
       [ 0.383708  ,  0.00151473,  0.12024342],
       [ 0.03051645,  0.01944995,  0.0233775 ],
       [ 0.02380184,  0.00161026,  0.2321944 ],
       [ 0.18128702,  0.41805095,  0.14812398],
       [ 0.00194675,  0.00394435,  0.05824932],
       [ 0.04838526,  0.01548758,  0.00166369],
       [ 0.01923324,  0.16425583,  0.00118137],
       [ 0.28796733,  0.30544608,  0.39129027]])


np.sum(np.maximum(0,np.abs(wwmat) - wwpenalty4))




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

ldates= [ (20120420, 20161223),  (20120420, 20161110),  (20120420, 20160804)     ]
lbound=    [  bounds3, bounds2, bounds1 ]
lminperf=  [  160.0,  195.0 ]    # PerfYear * nbYears
lopticriteria= [ 8.0, 5.0, 0.0 , 4.0,  1.0, 2.0            ]
lmaxvol= [ 0.07, 0.40 ]                              # lmaxvol
lconcen= np.array([  10.0,  50.0 ] )  * masset * 3       # Concentration Factor


'''

################### Create Params List  4   ##############################################################
'''   Reduce the weight of IEF

ldates= [ (20120420, 20161223)   ]
lbound=    [  bounds4 ]
lminperf=  [   185.0 ]    # PerfYear * nbYears
lopticriteria= [ 8.0, 5.0, 0.0 , 2.0            ]
lmaxvol= [ 0.40 ]                              # lmaxvol
lconcen= np.array([  10.0,  50.0 ] )  * masset * 3       # Concentration Factor
lwwpenalty=  np.array([ 0.4, 0.5, 0.6 ])


'''


################### Create Params List  5   ##############################################################
'''   Reduce the weight of IEF

ldates= [ (20120420, 20161223)   ]
lminperf=  [   185.0 ]    # PerfYear * nbYears
lopticriteria= [ 8.0, 5.0, 0.0 , 2.0            ]
lmaxvol= [ 0.40 ]                              # lmaxvol
lconcen= np.array([  10.0,  50.0 ] )  * masset * 3       # Concentration Factor
lbound=    [  bounds4 ]
lwwpenalty=  [ wwpenalty4 ]


'''





'''
best: 5, 8, 4, 1, 2

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


########################################################################################################
################# Generate List if of Tasks NOT done, if Batch incomplete ###############################
ntask_total = 5000
ncpu =        16
kk_already_done_per_cpu = -1   # -1 if to include all, find the if when not finished

kk_todo_list = np.array([  kk  if  (kk / ncpu )  > kk_already_done_per_cpu else -1  for kk in xrange(0, ntask_total)    ])
kk_todo_list = kk_todo_list[ kk_todo_list != -1]
print 'Nb_to_do', len(kk_todo_list[ kk_todo_list != -1]  )



################# Generate 1 Big List of all params ##################################################
input_params_all= []; ii=0; kk=0
for tt in ldates:
 for bounds in lbound :
  for concenfactor in lconcen  :
   for opticriteria  in  lopticriteria :
    for  maxvol in lmaxvol :
     for minperf in lminperf  :
       for wwpenalty in lwwpenalty :

                 kk+=1
                 if kk in kk_todo_list :
                  niter= 100      #
                  pop=   13      #
                  ievolve= 9      #
                  nisland= 1      #
                  krepeat= 15

                  ii+=1
                  vv= [ii,{'datei':tt, 'tt0': tt[0], 'tt1': tt[1], "bounds":bounds, "concenfactor":concenfactor,
                     'opticriteria':opticriteria, 'maxvol':maxvol, 'minperf':minperf, 'wwbear0': wwbear0,
                     'wwpenalty': wwpenalty,


                  'niter':niter, 'pop':pop, 'ievolve':ievolve, 'nisland': nisland, 'krepeat':krepeat }]

                  input_params_all.append(vv)
ntask= ii
print('ntask', ntask)

#######  Compute Time   ###########################################################################
NCPU= 2

time_ec2 = ntask * niter * pop * ievolve * krepeat / (NCPU * 15000) *1.1
print  '\n', 'Nb CPU: ', NCPU, 'Estimate Time in mins:', time_ec2




#######  SAVE Parameter data   #########################################################################
'''
 task1_name =     '       '
 DIRBATCH_local=  DIRCWD+"/linux/batch/task/" + task1_name
'''

execfile( DIRBATCH_local + '/ec2_config.py')

print 'task_name:  ', task1_name
input_param_file= DIRBATCH + '/input_params_all.pkl'
print DIRBATCH; print input_param_file
input_params_all= np.array(input_params_all, dtype= np.object)
util.py_save_obj(input_params_all, input_param_file, isabsolutpath=1)














##################################################################################################
##################################################################################################
''' Total task is 432:
niter= 80     #  pop=   15    #ievolve= 10   #  nisland= 1   #   krepeat= 20
404-378= 26

80 *15 * 10 *20 * 432 = 62400000
Total 6240000 in 6.5 hour (=6.5*60)
16000 / minutes

Convergence Fast :
  niter= 80     #  pop=   12    #ievolve= 9  #  nisland= 1   #   krepeat= 20



START 2016-12-31 13:14:40:273385
END   2016-12-31 19:42:15:135178


'''



'''
c4.4xlarge 	16 CPU  $0.1293 - 0.20, Stable
c4.8xlarge 	36 CPI  $0.30 - 0.50, unstable

Combinaison :
4**6: 4096,    5**6: 15000,   7**5; 16800,  8**5: 32765

'''





##################################################################################################
# batch

'''
###Local Linux Full Batch Launcher


ipython /media/sf_project27/linux/batch/task/elvis_prod_20161228/batch_launcher_02.py


If no CPU available, one process wont be launched



'''
