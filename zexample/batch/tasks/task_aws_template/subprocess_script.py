# -*- coding: utf-8 -*-
'''
New Strategy December 2016 with 
XLE, XLF, Cash, ZIV (Vol Index)  Rates Adjustment.

'''

'''
iibatch=1
batchname= 'elvis_prod_20161220_batch_01'
batch_out_log=''
DIRCWD=  os.getcwd()
fpath= DIRCWD+'/AIPORTFOLIO/research/Strat_elvis_newprod_dec2016/close_15assets_2012_dec216.spydata'
tt0,tt1= 20120420, 20161223        # dateref[-1] 
masset=9

concenfactor=  5.5*masset*3
minperf=  100.0 + 9.0 * 4   # PerfYear * nbYears
opticriteria= 1.0
maxvol= 0.09

aafolio_storage= np.empty((1, 19), dtype= np.object)
## Load Previous Storage of Portfolio
aux3_cols, aafolio_storage=  util.py_load_obj( 'aafolio_storage_20161226' )
pop=70
niter=20
batch_out_name="dec2016_"
iibatch=1

'''

import os

import numpy as np

import portfolio as pf
#-------------------- Load  module --------------------------------------------------
#  %load_ext autoreload
#  %autoreload 2
import PyGMO as pyg
import util
#-------------------- Load Data -----------------------------------------------------
from spyderlib.utils.iofuncs import load_dictionary

util.os_print_tofile( '\n'+util.date_nowtime() +'\n' , batch_out_log)
fpath= filedata
globals().update(load_dictionary(fpath)[0])



############## Date setup ###########################################################
# tt0,tt1= 20120420, 20161223        # dateref[-1] 
# masset=9

t0,t1= util.find(tt0, dateref),util.find(tt1, dateref)+1
print 'backtest period ', tt0,tt1, t0, t1

masset= len(sym)
#####################################################################################


#####################################################################################
# concenfactor=  5.5*masset*3
# minperf=  100.0 + 9.0 * 4   # PerfYear * nbYears
# opticriteria= 1.0
# maxvol= 0.09

'''
    if   criteria[0]==0.0 :  obj1= -(bsk -100)*(bsk -100) / vol + penalty
    elif criteria[0]==1.0 :  obj1= -bsk*50 + penalty
    elif criteria[0]==2.0 :  obj1= -(bsk -100) / vol + penalty
    elif criteria[0]==3.0 :  obj1= -bsk + 4200.0 * vol + penalty

'''

'''
          #Bear Market      #Range Bound   #Bull Market
bounds= [          #Transition Min Trigger, Max Trigger
         (0.01, 0.2),   (0.01, 0.05),   (0.01, 0.3),        #  "SPY",     1
         (0.01, 0.2),   (0.001, 0.1),    (0.01, 0.3),        #  "XLF",     2
         (0.01, 0.2),   (0.001, 0.1),    (0.01, 0.3),        #  "XLE",     3
         (0.01, 0.15),   (0.001, 0.05), (0.01, 0.15),        #  "ZIV",  4
         (0.01, 0.3),   (0.01, 0.3),    (0.01, 0.3),        #  "XLU",     5        
         (0.001, 0.2),   (0.001, 0.3),   (0.01, 0.1),        #  "IEF",    6
         (0.01, 0.2),   (0.01, 0.05),   (0.01, 0.3),        #  "HYG",     7
         (0.01, 0.15),  (0.05, 0.2),    (0.001, 0.1),        #  "UGLD",   8        
         (0.15, 0.5),    (0.2, 0.6),    (0.1, 0.5),        #  "SHY",        9   
        ] 
'''

# wwbear0= np.array([ 0.0,  0.0,   0.00,  0.0,   0.0,  0.2, 0.0 , 0.2, 0.6 ])

boundsextra= [ (-10,-8.0), (-10,15)   ]  
#####################################################################################
util.a_run_ipython("run "+ os.getcwd() +'/aapackage/function_custom.py')


wwbest1= np.ones(3*masset)/masset

wwbasket= {'opticriteria': opticriteria, 'minperf': minperf, 'maxvol': maxvol, 'concenfactor': concenfactor,
           'wwpenalty':wwpenalty, 'tlagsignal':1, 'rebfreq': 1, 'costbp': 0.0000, 'feesbp': 0.0,
           'maxweight': 0.0, 'schedule1': [], 'costbpshort': 0.0   }

wwextra1=  np.array([ -9.01671874,   6.47492034])
lfun=     {'mapping_calc_risk':mapping_calc_risk_v02, 'mapping_risk_ww': mapping_risk_ww_v01  }            
lweight=  ( wwbest1, wwextra1, wwbasket, [], wwbear0)
lbounds=  ( bounds, boundsextra, [], [])
statedata= (riskind[t0:t1, :], [])

pf4= pf.folioOptimizationF(sym, close[:,t0:t1], dateref[t0:t1])
pf4.setcriteria(lweight, lbounds, statedata, name='ELVIS_USD_Leverage_11assets_11', 
optimcrit="sharpe", 
wwtype="regime", nbregime=3, initperiod=1,riskid="multi", lfun=lfun )  

pf4.wwbest= wwbest1
bsk, wwall= pf4.plot(tickperday=252, show1=0)
# volta, wwvolta= pf.folio_volta(bsk,  0.10, 10, 1.0, isweight=1)
# pf.folio_analysis(dateref[t0:t1], volta)

# User Pre-Compute Risk
pf4.riskind_input= wwall[:,0:2]

# pf3.calc_optimal_weight(maxiter=50,  isreset=1, popsize=100)
#------------------------------------------------------------------------------------




#----------------    PYGMO usage ----------------------------------------------------
""" Portfolio Optimization"""
N= len(bounds)
# constructor of PyGMO problem ('dim' dimensions, 1 objective, 0 contraints etc.)    
class problem1(pyg.problem.base):       
    def __init__(self, dim=N):
        super(problem1,self).__init__(dim)
        self.__dim= dim    
        lbound= list([ x[0] for x in bounds ])
        ubound= list([ x[1] for x in bounds ])    
        self.set_bounds(lbound, ubound)

    def _objfun_impl(self, x):
        x= np.array(x, dtype=np.float)
        f=  pf4.calcbasket_obj2(x)       #   f= np.sum(x*x-5*x+6)    
        return (f, )
prob= problem1( N )



############# Meta Paameters   ######################################################
##   aafolio_storage= np.empty((1, 19), dtype= np.object)
## Load Previous Storage of Portfolio
##  aux3_cols, aafolio_storage=  util.py_load_obj( 'aafolio_storage_20161226' )
# pop=70
# niter=20
# batch_out_name="dec2016_"
# iibatch=1



############# Archipel   ############################################################
typepb= "optim_"+ batchname + str(iibatch)
#batch_out_log= batch_out+'/output_result.txt'


archi = pyg.archipelago(topology=pyg.topology.ring())
for j in xrange(0, nisland) :
  if   j==0 :
    if np.random.randint(1,10) > 6 :
      algo0 = pyg.algorithm.de_1220(gen=pop)
    else :
      algo0 = pyg.algorithm.de(gen=pop,variant= np.random.randint(1,10))

    archi.push_back(pyg.island(algo0, prob, pop))

  elif j==1 :
    algo1 = pyg.algorithm.de(gen=pop,variant= np.random.randint(1,10))
    archi.push_back(pyg.island(algo1, prob, pop))
  elif j==2 :
    algo2 = pyg.algorithm.de(gen=pop,variant= np.random.randint(1,10))
    archi.push_back(pyg.island(algo2, prob, pop))


# From Start 
i=0; icount= 0; fbest1=0.0

aux=  'Start PYGMO optimization ',tt0,tt1,str(sym),
aux+= 'init_pop',pop, 'iteration', niter, 'concen', wwbasket['concenfactor']
util.os_print_tofile( str(aux)+ '\n', batch_out_log) 

 
for k in xrange(0,niter) :
  i+=1
  archi.evolve(ievolve)  # Evolve the island
  archi.join()

  fbest= 10000000.0;  optibest= 1000000.0; xbest= 1000000.0; constbest= 10000.0   
  for jj, opti in enumerate(archi):
    fi= opti.population.champion.f[0]  
    if  fi <  fbest :
      xbest, fbest= opti.population.champion.x, fi
      # constbest= opti.population.champion.c  
      #util.py_save_obj(opti.population, 'linux/wwbest_islandpop'+str(jj)+'_'+typepb+'_'+str(i))


  # aux= (tt0,tt1, fbest, 'wwbest1= np.array(['+ str(xbest)[1:-2] +'])' )
  # print i,'\n', aux 
  # util.os_print_tofile( '\n\n' + str(aux), batch_out_log) 


  aux2, aux2_col= pf.folio_metric(pf4, xbest, 1, masset)
  aux3= np.array([ int(util.date_now()),  tt0, tt1, str(sym), str(xbest), str(wwbasket), str(bounds),  wwbasket['concenfactor'],  wwbasket['opticriteria']   ] + aux2, dtype= np.object)
  aafolio_storage= np.row_stack((aafolio_storage,  aux3))
  util.os_print_tofile( str((wwbasket['opticriteria'], kkk, i, icount,  aux2[ 2:6 ])) + '\n', batch_out_log)

  # util.os_print_tofile( '\n' + str(aux2),  batch_out_log) 
  if isverbose : print(str((wwbasket['opticriteria'], kkk, i,icount,  aux2[ 2:6 ])) )
 
  # Check if Optimizer repeats same solution
  if np.round(np.abs(fbest1- fbest), 3)  < 0.01 : icount+=1
  if icount > 6 :     break  
  fbest1= fbest


# batch_out_data= batch_out+ '/aafolio_storage_'  + util.date_now() + '.pkl'
# Save the data on disk
aux3_cols= [ 'date',  'tt0', 'tt1', 'sym', 'ww', 'fbest', 'typepb', 'concenfactor', 'opticriteria'   ] + aux2_col
util.py_save_obj( (aux3_cols, aafolio_storage), batch_out_data, isabsolutpath=1 )

 
txt= '\n END Bacth-------'+util.date_nowtime() +'---------------------------------\n'
util.os_print_tofile( txt, batch_out_log) 





####################################################################################
# util.save_obj(aux, 'linux/wwbest_portfolio_'+typepb+'_'+str(i))
# util.py_save_obj(archi, 'linux/archi_'+typepb+'_'+str(i))

# Load from previous state   
#'/media/sf_project27/aaserialize/linux/wwbest_portfolio_perf_201602_1_15.pkl'
                      
# i=31
# archi.push_back(pyg.island(algo1,util.load_obj( 'linux/wwbest_islandpop0_'+typepb+'_'+str(i))))
# archi.push_back(pyg.island(algo2,util.load_obj( 'linux/wwbest_islandpop1_'+typepb+'_'+str(i))))


####################################################################################
