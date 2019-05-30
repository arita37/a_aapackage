# -*- coding: utf-8 -*-
'''Portfolio Simulation details / simulation '''
import os
import sys

import IPython

import portfolio as pf
#-------------------- Load  module -------------------------------------------------- 
import PyGMO as pyg
import util
#-------------------- Load Data -----------------------------------------------------
from spyderlib.utils.iofuncs import load_dictionary

dir1= os.getcwd()
sys.path.append(dir1+'/aapackage');  sys.path.append(dir1+'/linux/aapackage')

util.a_run_ipython("load_ext autoreload"); util.a_run_ipython("autoreload 2")
# util.a_run_ipython('logstart -o -r -t aaserialize/log/log_'+util.date_now()+' rotate')


sys.path.append(dir1+'/aapackage');  sys.path.append(dir1+'/linux/aapackage')
if sys.platform.find('win') > -1 :  
   util.a_run_ipython("run " + dir1 + '/aapackage/allmodule.py')
else :  
   util.a_run_ipython("run /media/sf_project27/aapackage/allmodule.py")
   
   


dir1= os.getcwd()
fpath= dir1+'/AIPORTFOLIO/data/elvis_9asset_200704_20161028.spydata'
globals().update(load_dictionary(fpath)[0])


#-------------------- Load Function -------------------------------------------------
def islower(v1,v2) :
  if len(v1) == 1:     return 1 if (v1[0] < v2) else 0   
  elif len(v1) == 2:   return 1   




#---- Weight Minimization -----------------------------------------------------------
############## Date setup ###########################################################
tt0,tt1= 20070912, 20160126        # dateref[-1] 

t0,t1= util.find(tt0, dateref),util.find(tt1, dateref)+1
print 'backtest period ', tt0,tt1, t0, t1

masset= len(sym)
#####################################################################################



#####################################################################################
          #Bear Market      #Range Bound   #Bull Market
bounds= [          #Transition Min Trigger, Max Trigger
         (0.01, 0.3),   (0.01, 0.3),   (0.03, 0.1),        #  "SPY",   1
         (0.01, 0.3),   (0.01, 0.3),   (0.03, 0.3),        #  "FDN",   4
         (0.01, 0.3),   (0.01, 0.3),    (0.03, 0.3),        #  "XLU",   5        
         (0.01, 0.3),   (0.01, 0.3),    (0.01, 0.2),        #  "FXY",   9   
         (0.01, 0.2),   (0.1, 0.3),    (0.1, 0.2),         #  "IEF",   10   
         (0.01, 0.25),   (0.01, 0.3),    (0.03, 0.3),        #  "HYG",   8        
         (0.01, 0.3),   (0.01, 0.2),    (0.03, 0.3),        #  "VNQ",   6
         (0.01, 0.3),   (0.01, 0.2),    (0.01, 0.3),        #  "GLD",   6
         (0.01, 0.3),   (0.01, 0.25),    (0.15, 0.25)         #  "MUB",   12            
        ] 

sym= np.array(["SPY", "FDN", "XLU", "FXY", "IEF",  "HYG", "VNQ",  "GLD" ,"MUB" ] )
wwbear0= np.array([ 0.0, 0.0, 0.0, 0.3, 0.3, 0.0, 0.0, 0.2, 0.2   ])


wwbasket= {'opticriteria': 0.0, 'minperf':300.0, 'maxvol': 0.08, 'concenfactor': nfactor, 'wwpenalty':0.3, 'tlagsignal':1, 'rebfreq': 1, 'costbp': 0.0000, 'feesbp': 0.0, 'maxweight': 0.0, 'schedule1': [], 'costbpshort': 0.0   }


boundsextra= [ (-10,-8.0), (-10,15)   ]  
#####################################################################################
util.a_run_ipython("run " + os.getcwd() + '/aapackage/function_custom.py')


masset=len(sym)
wwbest1= np.ones(3*masset)/masset
wwextra1=  np.array([ -9.01671874,   6.47492034])
lfun=     {'mapping_calc_risk':mapping_calc_risk_v02, 'mapping_risk_ww': mapping_risk_ww_v01  }            
lweight=  ( wwbest1, wwextra1, wwbasket, [], wwbear0)
lbounds=  ( bounds, boundsextra, [], [])
statedata= (riskind[t0:t1, :], [])

pf4= pf.folioOptimizationF(sym, close[:,t0:t1], dateref[t0:t1])
pf4.setcriteria(lweight, lbounds, statedata, name='ELVIS_USD_Leverage_9assets_1', 
optimcrit="sharpe", wwtype="regime", nbregime=3, initperiod=1,riskid="multi", lfun=lfun )  

pf4.wwbest= wwbest1
bsk, wwall= pf4.plot(tickperday=252, show1=0)  
volta, wwvolta= pf.folio_volta(bsk,  0.10, 10, 1.0, isweight=1)
pf.folio_analysis(dateref[t0:t1], volta)

# User Pre-Compute Risk
pf4.riskind_input= wwall[:,0:2]


# pf3.calc_optimal_weight(maxiter=50,  isreset=1, popsize=100)
#------------------------------------------------------------------------------------


#----------------    PYGMO usage --------------------------------------------------
N= len(bounds)
# constructor of base class, PyGMO problem ('dim' dimensions, 1 objective, 0 contraints etc.)    
""" Portfolio Optimization"""
class problem1(pyg.problem.base):       
    def __init__(self, dim=N):
        super(problem1,self).__init__(dim)
        self.__dim= dim    
        lbound= list([ x[0] for x in bounds ])
        ubound= list([ x[1] for x in bounds ])    
        self.set_bounds(lbound, ubound)

    def _objfun_impl(self, x):
        x= np.array(x, dtype=np.float)
        f=  pf4.calcbasket_obj2(x)       # Pre-Compute f=  pf4.calcbasket_obj(x)
     #   f= np.sum(x*x-5*x+6)       
        return (f, )


prob= problem1( len(bounds))




############# Archipel   ###########################################################
typepb= "optim_"+ batchname + str(iibatch)
# pop=200
archi = pyg.archipelago(topology=pyg.topology.ring())
algo1 = pyg.algorithm.de_1220(gen=pop)  
algo2 = pyg.algorithm.de(gen=pop,variant=5)  



# From Start 
i=0
archi.push_back(pyg.island(algo1, prob, pop))
archi.push_back(pyg.island(algo2, prob, pop))



print('Start PYGMO optimization')
print 'Concen Factor', nfactor

for k in xrange(0, niter) :
 i+=1
 archi.evolve(1)  # Evolve the island once
 archi.join()

 fbest= 10000000.0;  optibest= 1000000.0; xbest= 1000000.0; constbest= 10000.0   
 for jj,opti in enumerate(archi):
   fi=   opti.population.champion.f 
   if islower(fi,fbest) :
      xbest= opti.population.champion.x
      fbest= opti.population.champion.f      
      # constbest= opti.population.champion.c  
      util.save_obj(opti.population, 'linux/wwbest_islandpop'+str(jj)+'_'+typepb+'_'+str(i))


 aux= (tt0,tt1, fbest, 'wwbest1= np.array(['+ str(xbest)[1:-2] +'])' )
 print i,'\n', aux 
 util.print_tofile( '\n\n' + str(aux), 'linux/output_result.txt') 

 aux2= pf.folio_metric(pf4, xbest, 1, masset)
 print aux2
 util.print_tofile( '\n' + str(aux2), 'linux/output_result.txt') 
 
 print '\n\n'


# util.save_obj(aux, 'linux/wwbest_portfolio_'+typepb+'_'+str(i))
 util.save_obj(archi, 'linux/archi_'+typepb+'_'+str(i))


txt= 'END Bacth----------------------------------------------------------'
util.print_tofile( '\n\n' + txt, 'linux/output_result.txt') 



####################################################################################
# Load from previous state   
#'/media/sf_project27/aaserialize/linux/wwbest_portfolio_perf_201602_1_15.pkl'
                      
# i=31
# archi.push_back(pyg.island(algo1,util.load_obj( 'linux/wwbest_islandpop0_'+typepb+'_'+str(i))))
# archi.push_back(pyg.island(algo2,util.load_obj( 'linux/wwbest_islandpop1_'+typepb+'_'+str(i))))


####################################################################################
