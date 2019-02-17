# -*- coding: utf-8 -*-
'''Portfolio Simulation details / simulation '''
import os, sys
dir1= os.getcwd()
sys.path.append(dir1+'/aapackage');  sys.path.append(dir1+'/linux/aapackage')

import  IPython, util
util.a_run_ipython("load_ext autoreload"); util.a_run_ipython("autoreload 2")
# util.a_run_ipython('logstart -o -r -t aaserialize/log/log_'+util.date_now()+' rotate')


import sys, os, util
sys.path.append(dir1+'/aapackage');  sys.path.append(dir1+'/linux/aapackage')
if sys.platform.find('win') > -1 :  
   util.a_run_ipython("run " + dir1 + '/aapackage/allmodule.py')
else :  
   util.a_run_ipython("run /media/sf_project27/aapackage/allmodule.py")
   
   
#-------------------- Load  module -------------------------------------------------- 
import PyGMO as pyg,  portfolio as pf


#-------------------- Load Data -----------------------------------------------------
from spyderlib.utils.iofuncs import load_dictionary
dir1= os.getcwd()
fpath= dir1+'/AIPORTFOLIO/research/Strat_elvis_new_strart/elvis_2_11asset_3xbond_2007_oct2016_v2.spydata'
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
         (0.01, 0.3),   (0.01, 0.3),   (0.01, 0.2),        #  "SPY",   1
         (0.01, 0.3),   (0.01, 0.2),    (0.01, 0.3),        #  "VDC",   2
         (0.01, 0.3),   (0.01, 0.3),    (0.01, 0.3),        #  "VPU",   3
         (0.01, 0.3),   (0.01, 0.3),   (0.01, 0.3),        #  "FDN",   4
         (0.01, 0.3),   (0.01, 0.3),    (0.01, 0.3),        #  "VNQ",   5        
         (0.01, 0.3),   (0.01, 0.2),    (0.01, 0.3),        #  "GLD",   6
         (0.01, 0.3),   (0.01, 0.3),   (0.01, 0.3),        #  "DBC",   7
         (0.01, 0.3),   (0.01, 0.3),    (0.01, 0.3),        #  "HYG",   8        
         (0.01, 0.3),   (0.01, 0.3),    (0.01, 0.3),        #  "FXY",   9   
         (0.01, 0.3),   (0.1, 0.3),    (0.1, 0.3),        #  "IEF",   10   
         (0.01, 0.3),   (0.01, 0.3),    (0.05, 0.3),        #  "TLT",   11  
         (0.01, 0.4),   (0.01, 0.4),    (0.08, 0.3)         #  "MUB",   12            
        ] 


'''

          #Bear Market      #Range Bound   #Bull Market
bounds= [          #Transition Min Trigger, Max Trigger
         (0.01, 0.15),   (0.01, 0.10),   (0.01, 0.2),        #  "SPY",   1
         (0.1, 0.2),   (0.1, 0.2),    (0.15, 0.22),        #  "VDC",   2
         (0.01, 0.15),   (0.01, 0.25),    (0.1, 0.22),        #  "VPU",   3
         (0.01, 0.15),   (0.01, 0.25),   (0.01, 0.22),        #  "FDN",   4
         (0.05, 0.2),   (0.01, 0.2),    (0.01, 0.2),        #  "VNQ",   5        
         (0.01, 0.1),   (0.1, 0.22),    (0.01, 0.1),        #  "GLD",   6
         (0.01, 0.1),   (0.01, 0.1),   (0.02, 0.1),        #  "DBC",   7
         (0.05, 0.2),   (0.01, 0.15),    (0.02, 0.2),        #  "HYG",   8        
         (0.01, 0.2),   (0.01, 0.2),    (0.01, 0.2),        #  "FXY",   9   
         (0.1, 0.25),   (0.1, 0.25),    (0.05, 0.3),        #  "IEF",   10   
         (0.1, 0.25),   (0.01, 0.2),    (0.05, 0.3),        #  "TLT",   11  
         (0.01, 0.25),   (0.01, 0.2),    (0.05, 0.25)         #  "MUB",   12            
        ] 




          #Bear Market      #Range Bound   #Bull Market
bounds= [          #Transition Min Trigger, Max Trigger
         (0.01, 0.25),   (0.01, 0.25),   (0.01, 0.2),        #  "SPY",   1
         (0.01, 0.25),   (0.01, 0.2),    (0.01, 0.25),        #  "VDC",   2
         (0.01, 0.25),   (0.01, 0.25),    (0.01, 0.25),        #  "VPU",   3
         (0.01, 0.25),   (0.01, 0.25),   (0.01, 0.25),        #  "FDN",   4
         (0.01, 0.25),   (0.01, 0.25),    (0.01, 0.25),        #  "VNQ",   5        
         (0.01, 0.25),   (0.01, 0.2),    (0.01, 0.25),        #  "GLD",   6
         (0.01, 0.25),   (0.01, 0.25),   (0.01, 0.25),        #  "DBC",   7
         (0.01, 0.25),   (0.01, 0.25),    (0.01, 0.25),        #  "HYG",   8        
         (0.01, 0.25),   (0.01, 0.25),    (0.01, 0.25),        #  "FXY",   9   
         (0.01, 0.25),   (0.01, 0.25),    (0.01, 0.25),        #  "IEF",   10   
         (0.01, 0.25),   (0.01, 0.25),    (0.01, 0.25),        #  "TLT",   11  
         (0.01, 0.25),   (0.1, 0.25),    (0.08, 0.25)         #  "MUB",   12            
        ] 
'''



boundsextra= [ (-10,-8.0), (-10,15)   ]  
#####################################################################################
util.a_run_ipython("run " + os.getcwd() + '/aapackage/function_custom.py')


wwbest1= np.array([0.03940309314216189, 0.03940309314216189, 0.03940309314216189, 0.03940309314216189, 0.03940309314216189, 0.03940309314216189, 0.03940309314216189, 0.03940309314216189, 0.03940309314216189, 0.03940309314216189, 0.023784879127515618, 0.16036715782902639, 0.0849388403995459, 0.07150662572729415, 0.18409286461246807, 0.09393039701225335, 0.04953585481211596, 0.052456153200159616, 0.029841700256935874, 0.0892243252067127, 0.012479193999853682, 0.0855417605626271, 0.2531798077923069, 0.09424696768248698, 0.21215967895462623, 0.06296441920186456, 0.01468872974339906, 0.2549750255522136, 0.10699180971684152, 0.031937298845936365, 0.011407281656234768, 0.27331930297220036, 0.010275950869191813, 0.04216113634525187, 0.012177993354049675, 0.06561260216689191 ])
 

wwextra1=  np.array([ -9.01671874,   6.47492034])
wwbear0= np.array([ 0.0,    0.0,   0.00,  0.0,   0.0,    0.1,   0.0,   0.0,   0.25,    0.3,   0.05,  0.3 ])
lfun=     {'mapping_calc_risk':mapping_calc_risk_v02, 'mapping_risk_ww': mapping_risk_ww_v01  }            
wwbasket= {'concenfactor': 130.0, 'wwpenalty':0.3, 'tlagsignal':1, 'rebfreq': 1, 'costbp': 0.0000, 'feesbp': 0.0, 'maxweight': 0.0, 'schedule1': [], 'costbpshort': 0.0   }
lweight=  ( wwbest1, wwextra1, wwbasket, [], wwbear0)
lbounds=  ( bounds, boundsextra, [], [])
statedata= (riskind[t0:t1, :], [])

pf4= pf.folioOptimizationF(sym, close[:,t0:t1], dateref[t0:t1])
pf4.setcriteria(lweight, lbounds, statedata, name='ELVIS_USD_Leverage_11assets_11', 
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
pop=200
archi = pyg.archipelago(topology=pyg.topology.ring())
algo1 = pyg.algorithm.de_1220(gen=pop)  
algo2 = pyg.algorithm.de(gen=pop,variant=5)  



# From Start 
i=0
archi.push_back(pyg.island(algo1, prob, pop))
archi.push_back(pyg.island(algo2, prob, pop))



print('Start PYGMO optimization')
for k in xrange(0,40) :
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

 aux2= pf.folio_metric(pf4, xbest, 1)
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






