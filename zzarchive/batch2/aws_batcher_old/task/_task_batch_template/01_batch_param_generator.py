# -*- coding: utf-8 -*-
import os
import sys

import numpy as np

import util
#-------------------- Get Close Price -------------------------------------------
from spyderlib.utils.iofuncs import load_dictionary

DIRCWD=  'D:/_devs/Python01/project27/' if sys.platform.find('win')> -1   else  '/home/ubuntu/notebook/' if os.environ['HOME'].find('ubuntu')>-1 else '/media/sf_project27/'
os.chdir(DIRCWD); sys.path.append(DIRCWD+'/aapackage');  sys.path.append(DIRCWD+'/linux/aapackage')
execfile( DIRCWD + '/aapackage/allmodule.py')
print 'Directory Folder', DIRCWD
#######################################################################################

task1_name=     'elvis_prod_28asset_1'
DIRBATCH_local=  DIRCWD+"/linux/batch/task/" +task1_name +'/'





fpath= DIRBATCH_local +'close_28assets_2012_jan2017.spydata'
globals().update(load_dictionary(fpath)[0])
for i,x in enumerate(sym) : print i,x




################## Model Parameters ###################################################
masset= 29
nbregime= 3

wwbasket= {'opticriteria': 1.0, 'minperf': 130.0, 'maxvol': 0.5, 'concenfactor': 1.0,
           'wwpenalty':0.4, 'tlagsignal':1, 'rebfreq': 1, 'costbp': 0.0000, 'feesbp': 0.0,
           'maxweight': 0.0, 'schedule1': [], 'costbpshort': 0.0   }

# wwbear0= np.array([ 0.0,  0.0,   0.00,  0.0,   0.0,  0.2, 0.0 , 0.2, 0.6 ])
wwbear0= np.zeros(masset)
wwbear0[26]= 0.6     ;   wwbear0[18]= 0.2     ;  wwbear0[23]= 0.2


          #Bear Market      #Range Bound   #Bull Market   #Transition Min Trigger, Max Trigger
bounds1=     [ (0.01, 0.3)] * 3 * masset

#Cash Allocation
bounds1[ 3 * 27 ]=   (0.01, 0.6)
bounds1[ 3 * 27+1 ]= (0.01, 0.6)
bounds1[ 3 * 27+2 ]= (0.01, 0.6)


wwpenalty1=  np.array([ x[1] for x in bounds1 ]).reshape((masset, nbregime))






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




################### Create Params List 0 ##############################################################
'''
ldates= [ (20120420, 20161230)  ]
lminperf=  [  175.0,  190.0 ]    # PerfYear * nbYears
lopticriteria= [  8.0, 5.0, 0.0 , 4.0,  1.0, 2.0              ]
lmaxvol= [   0.40 ]
lconcen= np.array([ 10.0,  20.0,  50.0 ] )  * masset * 3
lbound=    [  bounds1 ]
lwwpenalty=  [ wwpenalty1 ]


'''


################### Create Params List 0 ##############################################################
'''
ldates= [ (20120420, 20161230)  ]
lminperf=  [  175.0 ]    # PerfYear * nbYears
lopticriteria= [  2.0, 8.0, 5.0, 0.0 , 4.0,  1.0             ]
lmaxvol= [   0.40 ]
lconcen= np.array([  50.0 ] )  * masset * 3
lbound_penalty =    [ (bounds1,wwpenalty1), (bounds2,wwpenalty2),   (bounds3, wwpenalty3)    ]

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
 for bounds, wwpenalty in lbound_penalty :
  for concenfactor in lconcen  :
   for opticriteria  in  lopticriteria :
    for  maxvol in lmaxvol :
     for minperf in lminperf  :

                 kk+=1
                 if kk in kk_todo_list :
                  niter= 100      #
                  pop=   20       #
                  ievolve= 10     #
                  nisland= 1      #
                  krepeat= 50

                  ii+=1
                  vv= [ii,{'datei':tt, 'tt0': tt[0], 'tt1': tt[1], "bounds":bounds, "concenfactor":concenfactor,
                     'opticriteria':opticriteria, 'maxvol':maxvol, 'minperf':minperf, 'wwbear0': wwbear0,
                     'wwpenalty': wwpenalty,


                  'niter':niter, 'pop':pop, 'ievolve':ievolve, 'nisland': nisland, 'krepeat':krepeat }]

                  input_params_all.append(vv)
ntask= ii
print('ntask', ntask)

#######  Compute Time   ###########################################################################
NCPU= 20

time_ec2 = ntask * niter * pop * ievolve * krepeat / (NCPU * 7500) *1.1   #12asset --> 15000
print  '\n', 'Nb CPU: ', NCPU, 'Estimate Time in mins:', time_ec2,  time_ec2 /60.0




#######  SAVE Parameter data   ####################################################################
'''
 task1_name =     '       '
 DIRBATCH_local=  DIRCWD+"/linux/batch/task/" + task1_name
'''

execfile( DIRBATCH_local + '/ec2_config.py')
util.os_config_getfile(DIRBATCH_local + '/ec2_config.py');  print '\n'
input_param_file= DIRBATCH_local + input_param_file

input_params_all= np.array(input_params_all, dtype= np.object)
util.py_save_obj(input_params_all, input_param_file, isabsolutpath=1)


#Check
util.py_load_obj( input_param_file  ,isabsolutpath=1).shape


#print 'task_name:  ', task1_name
#
#print DIRBATCH_local; print input_param_file



'''
Genetic Algorithms Programming:

variables X, ...., X-1, X-2, ... , Y-1, Y-2
expr : if

Output : Y+1

Under :  Performance of Asset over period
         Long,

To DO
Here is a source-code example:

from pyevolve import *
import math

error_accum = Util.ErrorAccumulator()

# This is the functions used by the GP core,
# Pyevolve will automatically detect them
# and the they number of arguments
def gp_add(a, b): return a+b
def gp_sub(a, b): return a-b
def gp_mul(a, b): return a*b
def gp_sqrt(a):   return math.sqrt(abs(a))

def eval_func(chromosome):
   global error_accum
   error_accum.reset()
   code_comp = chromosome.getCompiledCode()

   for a in xrange(0, 5):
      for b in xrange(0, 5):
         # The eval will execute a pre-compiled syntax tree
         # as a Python expression, and will automatically use
         # the "a" and "b" variables (the terminals defined)
         evaluated     = eval(code_comp)
         target        = math.sqrt((a*a)+(b*b))
         error_accum += (target, evaluated)
   return error_accum.getRMSE()

def main_run():
   genome = GTree.GTreeGP()
   genome.setParams(max_depth=5, method="ramped")
   genome.evaluator.set(eval_func)

   ga = GSimpleGA.GSimpleGA(genome)
   # This method will catch and use every function that
   # begins with "gp", but you can also add them manually.
   # The terminals are Python variables, you can use the
   # ephemeral random consts too, using ephemeral:random.randint(0,2)
   # for example.
   ga.setParams(gp_terminals       = ['a', 'b'],
                gp_function_prefix = "gp")
   # You can even use a function call as terminal, like "func()"
   # and Pyevolve will use the result of the call as terminal
   ga.setMinimax(Consts.minimaxType["minimize"])
   ga.setGenerations(1000)
   ga.setMutationRate(0.08)
   ga.setCrossoverRate(1.0)
   ga.setPopulationSize(2000)
   ga.evolve(freq_stats=5)

   print ga.bestIndividual()

if __name__ == "__main__":
   main_run()

'''








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
