%load_ext autoreload
%autoreload

import gc
import os
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, KFold
warnings.filterwarnings('ignore')

print(os.getcwd())





0.20*0.20




dir1 = "/home/ubuntu/aagit/aapackage/aapackage/control/numpy_arrays/"

dir1 = "/home/ubuntu/zs3drive/"


dir1 =  r"D:/_devs/Python01/gitdev/zs3drive/"





from config import  export_folder
dir1 =  export_folder
print( export_folder)

##################################################################################
x = np.load( dir1 + "x.npy"  )
z = np.load( dir1 +"z.npy"  )
p = np.load( dir1 +"p.npy"  )
w = np.load( dir1 +"w.npy"  )

x.shape, z.shape, p.shape, w.shape

##################################################################################
def get_sample(i) :
  dd = { "x1" : x[i][0][0][:x.shape[3]-1],
   "x2" : x[i][1][0][:x.shape[3]-1],
   "pret" : p[i],
   "w1" : w[i][0],
   "w2" : w[i][1],
   "w3" : w[i][2],   
   "z1" : z[i][0],
   "z2" : z[i][1],   
  }  
  df = pd.DataFrame(dd)
  return df

get_sample( 90000 )[ [  "w1", "w2", "w3" ]   ]  




dfw = pd.DataFrame(  
   {   "w"+str(i+1) : w[:,i,-1] for i in range(w.shape[1])    }     
 )       
        


dfw.to_csv(dir1 + "/weight_conv.txt" )
dfw[["w2", "w1", "w3" ]].iloc[:90000, :].plot()

import matplotlib.pyplot as plt
plt.savefig(dir1 + 'output.png')




#############################################################################################
#############################################################################################
### Sum 0...T   and calculate volatility over all samples.
dfx = pd.DataFrame(  { "x1" : np.sum(x[:,0,0, :], axis=-1) ,
     "x2" : np.sum( x[:,1,0, :], axis=-1) ,
     "x3" : np.sum( x[:,2,0, :], axis=-1) 
   } )



### StdDev on All samples   ################################################################
print( dfx["x1"].std() , dfx["x2"].std(),  dfx["x3"].std() )


print( 
np.corrcoef( dfx["x1"].values , dfx["x2"].values )[1,0] ,

np.corrcoef( dfx["x1"].values , dfx["x3"].values )[1,0] ,

np.corrcoef( dfx["x2"].values , dfx["x3"].values )[1,0] , )


Variance over time .
Variance of all variance, Minimize.





#############################################################################################  
#############################################################################################









####  Scenario No stationary regime :   ############################################
    correlation



  np.cos( w.t )



####### MV Portfolio   ############################################################  
from pypfopt.efficient_frontier import EfficientFrontier

SigmaMatrix  = col * identiy'I)
Cov =  SigmaMatrix x Correl X SigmaMatrix x 



def get_portfolio(mu, vol, cor) :
  volmat = np.diag(vol )
  cov = np.dot( np.dot(volmat, cor)    , volmat )    
  ef = EfficientFrontier(mu, cov)
  print("vol", vol, "return", mu)
  print("cor", cor )
  print("Min Vol", ef.min_volatility() )
  print( ef.portfolio_performance(verbose=True)) 
  print("max sharpe", ef.max_sharpe() )
  print( ef.portfolio_performance(verbose=True))   
  

mu = np.array([0.10,0.05 ] ) 
vol = np.array([0.10,0.09 ] ) 
cor = np.array([[100, 0 ],
               [ 0, 100 ]])/100.0

        
volmat = np.diag(vol )
cov = np.dot( np.dot(volmat, cor)    , volmat )



####### Min Vol    ####################################
mu = np.array([0.10,0.10 ] ) 
vol = np.array([0.10,0.07 ] ) 
cor = np.array([[100,  0.0 ],
               [ 0.0, 100 ]])/100.0
get_portfolio(mu, vol, cor) 
"""
vol [0.1  0.07] return [0.1 0.1]
cor [[1. 0.]
 [0. 1.]]
Min Vol {0: 0.3288590620307241, 1: 0.6711409379692759}
Expected annual return: 10.0%
Annual volatility: 5.7%
Sharpe Ratio: 1.40
(0.1, 0.057346234436332834, 1.3950349275124232)
max sharpe {0: 0.3288589077648647, 1: 0.6711410922351353}
Expected annual return: 10.0%
Annual volatility: 5.7%
Sharpe Ratio: 1.40
(0.1, 0.057346234436335866, 1.3950349275123495)


9   98.517625   93.469500 -0.024047  0.330275  0.669725  0.172221  0.349226

"""




####### Min Vol    ####################################
mu = np.array([0.10, 0.10 ] ) 
vol = np.array([0.10, 0.07 ] ) 
cor = np.array([[100,  -50.0 ],
               [ -50.0, 100 ]])/100.0
get_portfolio(mu, vol, cor) 
"""
vol [0.1  0.07] return [0.1 0.1]
cor [[ 1.  -0.5]
 [-0.5  1. ]]
Min Vol {0: 0.3835616411209164, 1: 0.6164383588790836}
Expected annual return: 10.0%
Annual volatility: 4.1%
Sharpe Ratio: 1.95
(0.1, 0.04096440151864571, 1.9529151417868638)
max sharpe {0: 0.3835618317352055, 1: 0.6164381682647945}
Expected annual return: 10.0%
Annual volatility: 4.1%
Sharpe Ratio: 1.95
(0.1, 0.04096440151865514, 1.9529151417864143)


9  102.548081   96.277328 -0.000430  0.303653  0.696347  0.179486  0.411602

"""




####### Min Vol    ####################################
mu = np.array([0.10, 0.10 ] ) 
vol = np.array([0.10, 0.07 ] ) 
cor = np.array([[100,  50.0 ],
               [ 50.0, 100 ]])/100.0
get_portfolio(mu, vol, cor) 
"""
Min Vol {0: 0.18030065216589672, 1: 0.8196993478341033}
Expected annual return: 10.0%
Annual volatility: 6.8%
Sharpe Ratio: 1.17
(0.1, 0.06820538059999318, 1.1729279903762901)
max sharpe {0: 0.17742495826898114, 1: 0.8225750417310189}
Expected annual return: 10.0%
Annual volatility: 6.8%
Sharpe Ratio: 1.17
(0.10000000000000002, 0.06820483180683885, 1.172937428048587)




9  102.250718   98.527843  0.009094  0.067539  0.932461  0.025022  0.345460

"""



####### Min Vol  ###################################################################################
mu = np.array([0.0,0.0, 0.0 ] ) 
vol = np.array([0.24, 0.16, 0.06 ] ) 
cor = np.array([[100,  -16, -34 ],
                [-16,  100, -55 ],
                [-34,  -55, 100 ],                
               ])/100.0
get_portfolio(mu, vol, cor) 
# Vol {0: 0.09990258275835834, 1: 0.2160461468927457, 2: 0.684051270348896}
## NN :  9  0.095334  0.201686  0.702979




####### MIn Vol
mu = np.array([0.0,0.0, 0.0 ] ) 
vol = np.array([0.20, 0.15, 0.08 ] ) 
cor = np.array([[100,  -50, -40 ],
                [-50,  100, -30 ],
                [-40,  -30, 100 ],                
               ])/100.0
get_portfolio(mu, vol, cor) 
"""
vol [0.2  0.15 0.08] return [0.1 0.1 0.1]
cor [[ 1.  -0.5 -0.4]
 [-0.5  1.  -0.3]
 [-0.4 -0.3  1. ]]
Min Vol {0: 0.20909644961889745, 1: 0.26905710025071106, 2: 0.5218464501303914}
Expected annual return: 10.0%
Annual volatility: 3.2%
Sharpe Ratio: 2.50
(0.1, 0.032018079624367014, 2.4985883269250433)
max sharpe {0: 0.20984180165594454, 1: 0.2688605769087439, 2: 0.5212976214353117}
Expected annual return: 10.0%
Annual volatility: 3.2%
Sharpe Ratio: 2.50
(0.10000000000000002, 0.03201753047507891, 2.498631181510661)



9  0.091283  0.200547  0.708170

"""



####### MIn Vol
mu = np.array([0.10,0.10, 0.10 ] ) 
vol = np.array([0.10, 0.09, 0.07 ] ) 
cor = np.array([[100,  90, 80 ],
                [90,  100, 70 ],
                [80,  70, 100 ],                
               ])/100.0
get_portfolio(mu, vol, cor) 
"""
vol [0.1  0.09 0.07] return [0.1 0.1 0.1]
cor [[1.  0.9 0.8]
 [0.9 1.  0.7]
 [0.8 0.7 1. ]]
Min Vol {0: 3.004629197474327e-18, 1: 0.11722487418658731, 2: 0.8827751258134127}
Expected annual return: 10.0%
Annual volatility: 7.0%
Sharpe Ratio: 1.15
(0.1, 0.06958850342270942, 1.1496151816061748)
max sharpe {0: 1.3520831388634434e-16, 1: 0.11721677181184023, 2: 0.8827832281881597}
Expected annual return: 10.0%
Annual volatility: 7.0%
Sharpe Ratio: 1.15
(0.10000000000000002, 0.0695885034246841, 1.1496151815735527)


#### Big Difference
9  0.097814  0.487812  0.414374

"""




####### MIn Vol
mu = np.array([0.10,0.10, 0.10 ] ) 
vol = np.array([0.10, 0.09, 0.07 ] ) 
cor = np.array([[100,  0, 0 ],
                [0,  100, 0 ],
                [0,  0, 100 ],                
               ])/100.0
get_portfolio(mu, vol, cor) 
"""
vol [0.1  0.09 0.07] return [0.1 0.1 0.1]
cor [[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
Min Vol {0: 0.22319098495690673, 1: 0.31163688384238153, 2: 0.46517213120071177}
Expected annual return: 10.0%
Annual volatility: 4.8%
Sharpe Ratio: 1.65
(0.1, 0.048426039879695314, 1.6520037607606113)
max sharpe {0: 0.23390006161433718, 1: 0.2887680147123651, 2: 0.4773319236732977}
Expected annual return: 10.0%
Annual volatility: 4.8%
Sharpe Ratio: 1.65
(0.1, 0.04836291002083988, 1.654160181129041)


#Ok 
9  0.232698  0.299768  0.467534

"""



####### MIn Vol
mu = np.array([0.10,0.10, 0.10 ] ) 
vol = np.array([0.17, 0.12, 0.073 ] ) 
cor = np.array([[100,  0, 0 ],
                [0,  100, 0 ],
                [0,  0, 100 ],                
               ])/100.0
get_portfolio(mu, vol, cor) 



"""
vol [0.17  0.12  0.073] return [0.1 0.1 0.1]
cor [[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
Min Vol {0: 0.11841814927242073, 1: 0.23704921186827646, 2: 0.6445326388593028}
Expected annual return: 10.0%
Annual volatility: 5.9%
Sharpe Ratio: 1.37
(0.1, 0.05855096660795398, 1.3663309870811293)
max sharpe {0: 0.11863172689677262, 1: 0.23806269291354182, 2: 0.6433055801896856}
Expected annual return: 10.0%
Annual volatility: 5.9%
Sharpe Ratio: 1.37
(0.10000000000000002, 0.058550760292185804, 1.3663358016322262)

#lstm
9  0.193310  0.240782  0.565908



"""


























print("max sharpe", ef.max_sharpe() )
print( ef.portfolio_performance(verbose=True))  


wsharpe = ef.weights


cov1 = np.linalg.inv(cov)
np.dot(cov1, mu)

10/ 0.4

20/0.6





mu = np.array([0.10,0.05 ] ) 
vol = np.array([0.15,0.09 ] ) 
cor = np.array([[100,  -50.0 ],
               [ -50.0, 100 ]])/100.0

        
volmat = np.diag(vol )
cov = np.dot( np.dot(volmat, cor)    , volmat )


####### Min Volaitlity
ef = EfficientFrontier(mu, cov)

print("Min Vol", ef.min_volatility() )
print( ef.portfolio_performance(verbose=True))  

















Inputs:

cov_matrix
n_assets
tickers
bounds

Optimisation parameters:
initial_guess
constraints
Output: weights

Public methods:

max_sharpe() optimises for maximal Sharpe ratio (a.k.a the tangency portfolio)
min_volatility() optimises for minimum volatility
custom_objective() optimises for some custom objective function
efficient_risk() maximises Sharpe for a given target risk
efficient_return() minimises risk for a given target return

portfolio_performance() calculates the expected return, volatility and Sharpe ratio for the optimised portfolio.
















1/ 0.4 , 1/0.1
####################################################################################################
####################################################################################################
from pyrb import EqualRiskContribution

ERC = EqualRiskContribution(cov)
ERC.solve()
ERC.get_risk_contributions()
ERC.get_volatility()



import pandas as pd
import numpy as np
from pyrb import ConstrainedRiskBudgeting


vol = [0.05,0.05,]




vol = [0.05,0.05,0.07,0.1,0.15,0.15,0.15,0.18]
cor = np.array([[100,  80,  60, -20, -10, -20, -20, -20],
               [ 80, 100,  40, -20, -20, -10, -20, -20],
               [ 60,  40, 100,  50,  30,  20,  20,  30],
               [-20, -20,  50, 100,  60,  60,  50,  60],
               [-10, -20,  30,  60, 100,  90,  70,  70],
               [-20, -10,  20,  60,  90, 100,  60,  70],
               [-20, -20,  20,  50,  70,  60, 100,  70],
               [-20, -20,  30,  60,  70,  70,  70, 100]])/100
cov = np.outer(vol,vol)*cor




C = None
d = None

CRB = ConstrainedRiskBudgeting(cov,C=C,d=d)
CRB.solve()
print(CRB)



C = np.array([[0,0,0,0,-1.0,-1.0,-1.0,-1.0]]) 
d = [-0.3]

CRB = ConstrainedRiskBudgeting(cov,C=C,d=d)
CRB.solve()
print(CRB)



ap = 2 *3.14/2

t = np.arange(0, 1, 0.1)
np.cos( ap*t)



####################################################################################################
10 years, 6 months period,







####################################################################################################
####################################################################################################
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Read in price data
df = pd.read_csv("tests/stock_prices.csv", parse_dates=True, index_col="date")


# Calculate expected returns and sample covariance
mu = drift
S = cov

# Optimise for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)



def random_portfolio(returns):  
    '''  
    Returns the mean and standard deviation of returns for a random portfolio  
    '''

    p = np.asmatrix(np.mean(returns, axis=1))  
    w = np.asmatrix(rand_weights(returns.shape[0]))  
    C = np.asmatrix(np.cov(returns))  
    mu = w * p.T  
    sigma = np.sqrt(w * C * w.T)  
    # This recursion reduces outliers to keep plots pretty  
    if sigma > 2:  
        return random_portfolio(returns)  
    return mu, sigma  



n_portfolios = 500  
means, stds = np.column_stack([  
    random_portfolio(return_vec)  
    for _ in xrange(n_portfolios)  
])




    
 def optimal_portfolio(returns):  
    n = len(returns)  
    returns = np.asmatrix(returns)  
    N = 100  
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]  
    # Convert to cvxopt matrices  
    S = opt.matrix(np.cov(returns))  
    pbar = opt.matrix(np.mean(returns, axis=1))  
    # Create constraint matrices  
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix  
    h = opt.matrix(0.0, (n ,1))  
    A = opt.matrix(1.0, (1, n))  
    b = opt.matrix(1.0)  
    # Calculate efficient frontier weights using quadratic programming  
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']  
                  for mu in mus]  
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER  
    returns = [blas.dot(pbar, x) for x in portfolios]  
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]  
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE  
    m1 = np.polyfit(returns, risks, 2)  
    x1 = np.sqrt(m1[2] / m1[0])  
    # CALCULATE THE OPTIMAL PORTFOLIO  
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']  
    return np.asarray(wt), returns, risks



weights, returns, risks = optimal_portfolio(return_vec)

plt.plot(stds, means, 'o')  
plt.ylabel('mean')  
plt.xlabel('std')  
plt.plot(risks, returns, 'y-o')  



   
# find min Volatility & max sharpe values in the dataframe (df)
min_volatility = df['Volatility'].min()
max_sharpe = df['Sharpe Ratio'].max()

# use the min, max values to locate and create the two special portfolios
sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
min_variance_port = df.loc[df['Volatility'] == min_volatility]

# plot frontier, max sharpe & min Volatility values with a scatterplot
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200 )
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()








####################################################################################################
####################################################################################################
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv,pinv
from scipy.optimize import minimize

 # risk budgeting optimization
def calculate_portfolio_var(w,V):
    # function that calculates portfolio risk
    w = np.matrix(w)
    return (w*V*w.T)[0,0]

def calculate_risk_contribution(w,V):
    # function that calculates asset contribution to total risk
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w,V))
    # Marginal Risk Contribution
    MRC = V*w.T
    # Risk Contribution
    RC = np.multiply(MRC,w.T)/sigma
    return RC

def risk_budget_objective(x,pars):
    # calculate portfolio risk
    V = pars[0]# covariance table
    x_t = pars[1] # risk target in percent of portfolio risk
    sig_p =  np.sqrt(calculate_portfolio_var(x,V)) # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p,x_t))
    asset_RC = calculate_risk_contribution(x,V)
    J = sum(np.square(asset_RC-risk_target.T))[0,0] # sum of squared error
    return J

def total_weight_constraint(x):
    return np.sum(x)-1.0

def long_only_constraint(x):
    return x

x_t = [0.25, 0.25, 0.25, 0.25] # your risk budget percent of total portfolio risk (equal risk)
cons = ({'type': 'eq', 'fun': total_weight_constraint},
{'type': 'ineq', 'fun': long_only_constraint})
res= minimize(risk_budget_objective, w0, args=[V,x_t], method='SLSQP',constraints=cons, options={'disp': True})
w_rb = np.asmatrix(res.x)


















##################################################################################
##################################################################################
import logging
import os
import time

import numpy as np
import tensorflow as tf

TF_DTYPE = tf.float32
MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0

from submodels import subnetwork_bidirectionalattn, subnetwork_dila, subnetwork_ff
from submodels import subnetwork_lstm, subnetwork_lstm_attn

tf.enable_eager_execution()


def log(s):
  logging.info(s)


"""
Variance Realized =  Sum(ri*:2)
rI**2 =   Sum(wi.ri)**2
Return = sum(ri) = Total return
"""

export_folder = "/home/ubuntu/zs3drive/"


def save_history(export_folder, train_history, x_all, z_all, p_all, w_all, y_all) :
    print("Writing path history on disk, {}/".format(export_folder))
    if not os.path.exists(export_folder):
      os.makedirs(export_folder)
    
    np.save(os.path.join(export_folder, 'x.npy'), np.concatenate(x_all, axis=0))
    # np.save(export_folder + '/y.npy', np.concatenate(y_all, axis=0))
    np.save(os.path.join(export_folder, 'z.npy'), np.concatenate(z_all, axis=0))
    np.save(os.path.join(export_folder, 'p.npy'), np.concatenate(p_all, axis=0))
    np.save(os.path.join(export_folder, 'w.npy'), np.concatenate(w_all, axis=0))




###################################################################################################
class FeedForwardModel(object):
  """
     Global model over the full time steps
  """
  
  def __init__(self, config, bsde, sess, usemodel):
    self._config = config
    self._bsde = bsde  # BSDE Equation
    self._sess = sess  # TF session
    self._usemodel = usemodel
    
    # make sure consistent with FBSDE equation
    self._dim = bsde.dim
    self._T = bsde.num_time_interval
    self._total_time = bsde.total_time
    
    self._smooth = 1e-8
    
    self._is_training = tf.placeholder(tf.bool)
    if usemodel == 'attn':
      self.subnetwork = subnetwork_lstm_attn(config)
    
    if usemodel == 'lstm':
      self.subnetwork = subnetwork_lstm(config)
    
    if usemodel == 'ff':
      self.subnetwork = subnetwork_ff(config, self._is_training)
    
    if usemodel == 'dila':
      self.subnetwork = subnetwork_dila(config)
    
    if usemodel == 'biattn':
      self.subnetwork = subnetwork_bidirectionalattn(config)
    
    self._extra_train_ops = []
    
    # self._config.num_iterations = None  # "Nepoch"
    # self._train_ops = None  # Gradient, Vale,
  
  def train(self):
    t0 = time.time()
    
    ## Validation DATA : Brownian part, drift part from MC simulation
    # dw_valid, x_valid = self._bsde.sample(self._config.batch_size)
    #################################################################
    dw_valid, x_valid = self.generate_feed()
    feed_dict_valid = {self._dw: dw_valid, self._x: x_valid, self._is_training: False}
    
    # update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)  # V1 compatibility
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    self._train_ops = tf.group([self._train_ops, update_ops])

    
    # initialization
    self._sess.run(tf.global_variables_initializer())  
    val_writer = tf.summary.FileWriter('logs', self._sess.graph)
    merged = tf.summary.merge_all()
    train_history = []  # to save iteration results
    
    for step in range(self._config.num_iterations + 1):
      # Generate MC sample AS the training input
      dw_train, x_train = self.generate_feed()
      
      self._sess.run(self._train_ops,
        feed_dict={self._dw: dw_train,
                   self._x: x_train, self._is_training: True},)
      
      
      ### Validation Data Eval.      
      self.validation_train(feed_dict_valid, train_history, merged, t0, step, val_writer)
      
    return np.array(train_history)
  
  
  def validation_train(self, feed_dict_valid, train_history, merged, t0, step, val_writer):
    if step % self._config.logging_frequency != 0:
        return 0
    loss, init, summary = self._sess.run([self._loss, self._y_init, merged],
                                         feed_dict=feed_dict_valid)
    dt0 = time.time() - t0 + self._t_build
    train_history.append([step, loss, init, dt0])
    print("step: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u"
          % (step, loss, init, dt0))
    val_writer.add_summary(summary, step)
  
  
  def generate_feed(self):
    dw_valid, x_valid = [], []
    for clayer in range(self._config.clayer):
      dw, x = self._bsde.sample(self._config.batch_size, clayer)
      dw_valid.append(dw)
      x_valid.append(x)
    
    dw_valid, x_valid = np.stack(dw_valid, axis=2), np.stack(x_valid, axis=2)
    dw_valid = np.reshape(dw_valid,
                          [self._config.batch_size,
                           self._config.clayer * self._dim, self._T])
    
    x_valid = np.reshape(x_valid,
                         [self._config.batch_size,
                          self._config.clayer * self._dim, self._T + 1])
    ##################################################################
    return dw_valid, x_valid
  
  
  def train2(self):
    t0 = time.time()
    
    ## Validation DATA : Brownian part, drift part from MC simulation
    dw_valid, x_valid = self.generate_feed()
    feed_dict_valid = {self._dw: dw_valid, self._x: x_valid, self._is_training: False}
    
    
    # update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)  # V1 compatibility
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    self._train_ops = tf.group([self._train_ops, update_ops])
    
    
    # initialization
    self._sess.run(tf.global_variables_initializer())  
    val_writer = tf.summary.FileWriter('logs', self._sess.graph)
    merged = tf.summary.merge_all()
    train_history = []  # to save iteration results
    x_all, z_all, p_all, w_all, y_all = [], [], [], [], []
    
    
    for step in range(self._config.num_iterations + 1):
      # Generate MC sample AS the training input
      dw_train, x_train = self.generate_feed()
      
      y, p, z, w = self._sess.run([self._train_ops, self.all_p, self.all_z, self.all_w],
                                  feed_dict={self._dw: dw_train,
                                             self._x: x_train, self._is_training: True},
                                  )
                                  
      x_train_orig = np.reshape(x_train, [self._config.batch_size, self._config.dim,
                                          self._config.clayer, self._T + 1])
      x_all.append(x_train_orig)
      p_all.append(p)
      z_all.append(z)
      w_all.append(w)
      y_all.append(y)
      
      ### Validation Data Eval.      
      self.validation_train(feed_dict_valid, train_history, merged, t0, step, val_writer)


    save_history(export_folder, train_history, x_all, z_all, p_all, w_all, y_all)
    return np.array(train_history)

    
  
  def build2(self):
    """"
       y : State
       dw : Brownian,  x : drift deterministic
       z : variance
    """
    t0 = time.time()
    TT = np.arange(0, self._bsde.num_time_interval) * self._bsde.delta_t
    M = self._config.dim  #Nb of assets
    
    
    ### dim X Ntime_interval for Stochastic Process
    self._dw = tf.placeholder(TF_DTYPE,
                              [None, self._dim * self._config.clayer, self._T],
                              name="dW")
    self._x = tf.placeholder(TF_DTYPE, [None, self._dim * self._config.clayer,
                                        self._T + 1],
                             name="X")
    
    ### Initialization
    ### x : state,  Cost
    self._y_init = tf.Variable(tf.random_uniform(
      [1], minval=self._config.y_init_range[0], maxval=self._config.y_init_range[1],
      dtype=TF_DTYPE))
    
    ## Control
    z_init = tf.Variable(
      tf.random_uniform([1, self._dim], minval=0.01, maxval=0.3, dtype=TF_DTYPE)
      # tf.random_uniform([1, self._dim], minval=-0.1, maxval=0.1, dtype=TF_DTYPE)
    )
    
    p0 = tf.Variable(tf.random_uniform(shape=[self._config.batch_size],
                                          minval=0.0, maxval=0.0, dtype=TF_DTYPE))
    
    all_one_vec = tf.ones(shape=tf.stack([tf.shape(self._dw)[0], 1]), dtype=TF_DTYPE)
    z0 = tf.matmul(all_one_vec, z_init)
    
    
    w0 = tf.Variable( tf.random.uniform( [self._config.batch_size, M * self._config.clayer],
                      minval=0.1, maxval=0.3, dtype=TF_DTYPE))
    
    all_p, all_z, all_w, all_y = [p0], [z0], [w0], []
    with tf.variable_scope("forward"):
      for t in range(1, self._T):
        # y = (y_old
        #        - self._bsde.delta_t * (self._bsde.f_tf(TT[t], self._x[:, :, t], y_old, z))
        #        + tf.reduce_sum(z * self._dw[:, :, t], 1, keepdims=True))

        ## Neural Network per Time Step, Calculate Gradient
        ######################################################################
        ## Z = [batch, dim * clayer] -> [batch, 4] for optionprice config
        if self._usemodel == 'lstm':
          z = self.subnetwork.build([ self._x[:, :, t -1] ], t-1) / self._dim
        
        elif self._usemodel == 'ff':
          z = self.subnetwork.build(self._x[:, :, t -1], t-1) / self._dim
        
        elif self._usemodel == 'attn':
          z = self.subnetwork.build([ self._x[:, :, t -1] ], t-1) / self._dim
        
        elif self._usemodel == 'dila':
          z = self.subnetwork.build([self._x[:, :, t -1]], t-1) / self._dim
        
        elif self._usemodel == 'biattn':
          z = self.subnetwork.build(self._x[:, :, t -1], t-1) / self._dim
        all_z.append(z)
        
        ######################################################################
        y = z
        all_y.append(y)
        
        ###################################################################### 
        if t == 1 :
          w = 0.0 + z 
    
        else :
          ### t=1 has issue
          w = 0.0 + z + 1 / tf.sqrt((tf.nn.moments(
            tf.log((self._x[:, :M, 1:t ]) / (
                    self._x[:, :M, 0:t-1 ])), axes=2)[1] + self._smooth))
        
        # w = z 
        w = w / tf.reduce_sum(w, -1, keepdims=True)  ### Normalize Sum to 1
        all_w.append(w)
        
        ######################################################################
        # p =  p_old * (1 + tf.reduce_sum( w * (self._x[:, :, t] / self._x[:, :, t-1] - 1), 1))
        p = tf.reduce_sum(w * (
                  self._x[:, :M, t] / self._x[:, :M, t - 1] - 1), 1)
        all_p.append(p)
      
      # Terminal time
      # y = (y  - self._bsde.delta_t * self._bsde.f_tf(TT[-1], self._x[:, :, -2], y, z)
      #        + tf.reduce_sum(z * self._dw[:, :, -1], 1, keepdims=True) )
      # y = self._x[:, :, -2]
      # all_y.append(y)

      
      self.all_y = tf.stack(all_y, axis=-1)
      self.all_z = tf.stack(all_z, axis=-1)
      self.all_w = tf.stack(all_w, axis=-1)
      self.all_p = tf.stack(all_p, axis=-1)
      p = self.all_p
      
      # Final Difference :
      #######  -Sum(ri)   +Sum(ri**2)
      delta = -0.01 * tf.reduce_sum(p[:, 1:], 1) + tf.nn.moments(p[:, 1:], axes=1)[1]*10000.0
      self._loss = tf.reduce_mean(delta)
    
    tf.summary.scalar('loss', self._loss)
    # train operations
    global_step = tf.get_variable(
      "global_step", [],
      initializer=tf.constant_initializer(0),
      trainable=False,
      dtype=tf.int32,
    )
    
    learning_rate = tf.train.piecewise_constant(
      global_step, self._config.lr_boundaries, self._config.lr_values
    )
    
    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self._loss, trainable_variables)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    apply_op = optimizer.apply_gradients(
      zip(grads, trainable_variables), global_step=global_step, name="train_step"
    )
    all_ops = [apply_op] + self._extra_train_ops
    self._train_ops = tf.group(*all_ops)
    self._t_build = time.time() - t0
  
  
  def build(self):
    """"
       y : State
       dw : Brownian,  x : drift deterministic
       z : variance
    """
    t0 = time.time()
    TT = np.arange(0, self._bsde.num_time_interval) * self._bsde.delta_t
    
    ### dim X Ntime_interval for Stochastic Process
    self._dw = tf.placeholder(TF_DTYPE,
                              [None, self._config.clayer * self._dim, self._T],
                              name="dW")
    self._x = tf.placeholder(TF_DTYPE, [None, self._config.clayer * self._dim,
                                        self._T + 1],
                             name="X")
    
    ### Initialization
    ## x : state,  Cost
    self._y_init = tf.Variable(
      tf.random_uniform(
        [1], minval=self._config.y_init_range[0], maxval=self._config.y_init_range[1],
        dtype=TF_DTYPE,
      )
    )
    
    
    ## Control
    z_init = tf.Variable(
      tf.random_uniform([1, self._dim], minval=-0.1, maxval=0.1, dtype=TF_DTYPE)
    )
    all_one_vec = tf.ones(shape=tf.stack([tf.shape(self._dw)[0], 1]), dtype=TF_DTYPE)
    y = all_one_vec * self._y_init
    z = tf.matmul(all_one_vec, z_init)
    
    with tf.variable_scope("forward"):
      for t in range(0, self._T - 1):
        y = (
                y
                - self._bsde.delta_t * (
                  self._bsde.f_tf(TT[t], self._x[:, :self._dim, t], y, z))
                + tf.reduce_sum(z * self._dw[:, :self._dim, t], 1, keepdims=True)
        )
        
        ## Neural Network per Time Step, Calculate Gradient
        if self._usemodel == 'lstm':
          z = self.subnetwork.build([self._x[:, :, t + 1]], t) / self._dim
        
        elif self._usemodel == 'ff':
          z = self.subnetwork.build(self._x[:, :, t + 1], t) / self._dim
        
        elif self._usemodel == 'attn':
          z = self.subnetwork.build([self._x[:, :, t + 1]], t) / self._dim
        
        elif self._usemodel == 'dila':
          z = self.subnetwork.build([self._x[:, :, t + 1]], t) / self._dim
        
        elif self._usemodel == 'biattn':
          z = self.subnetwork.build(self._x[:, :, t + 1], t) / self._dim
      
      # Terminal time
      y = (
              y
              - self._bsde.delta_t * self._bsde.f_tf(TT[-1], self._x[:, :self._dim, -2], y, z)
              + tf.reduce_sum(z * self._dw[:, :self._dim, -1], 1, keepdims=True)
      )
      
      # Final Difference :
      delta = y - self._bsde.g_tf(self._total_time, self._x[:, :self._dim, -1])
      
      # use linear approximation outside the clipped range
      self._loss = tf.reduce_mean(
        tf.where(
          tf.abs(delta) < DELTA_CLIP,
          tf.square(delta),
          2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2,
        )
      )
    tf.summary.scalar('loss', self._loss)
    
    # train operations
    global_step = tf.get_variable(
      "global_step", [],
      initializer=tf.constant_initializer(0),
      trainable=False,
      dtype=tf.int32,
    )
    
    learning_rate = tf.train.piecewise_constant(
      global_step, self._config.lr_boundaries, self._config.lr_values
    )
    
    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self._loss, trainable_variables)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    apply_op = optimizer.apply_gradients(
      zip(grads, trainable_variables), global_step=global_step, name="train_step"
    )
    all_ops = [apply_op] + self._extra_train_ops
    self._train_ops = tf.group(*all_ops)
    self._t_build = time.time() - t0
