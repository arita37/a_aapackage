
import sys, os
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal



####################################################################################################
from utils import gbm_multi, gbm_multi_regime
import json
from config import export_folder



####################################################################################################
class Equation(object):
    """Base class for defining equations"""

    def __init__(self, dim, total_time, num_time_interval):
        self._dim = dim
        self._total_time = total_time
        self._num_time_interval = num_time_interval
        self._delta_t = (self._total_time + 0.0) / self._num_time_interval
        self._sqrt_delta_t = np.sqrt(self._delta_t)
        self._y_init = None

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError

    @property
    def y_init(self):
        return self._y_init

    @property
    def dim(self):
        return self._dim

    @property
    def num_time_interval(self):
        return self._num_time_interval

    @property
    def total_time(self):
        return self._total_time

    @property
    def delta_t(self):
        return self._delta_t



def get_equation(name, dim, total_time, num_time_interval):
    try:
        return globals()[name](dim, total_time, num_time_interval)
    except KeyError:
        raise KeyError("Equation for the required problem not found.")





def scenario(name, nasset) :
   if name== "zero" : 
        s0    = np.ones((nasset)) * 100.0
        drift = np.ones((nasset,1)) * 0.0
        vol0  = np.ones((nasset,1)) * 0.2      
        vol0 =  np.array([[ 0.20, 0.15, 0.08 ]]).T 
        correl =  np.array([[100, 0 , 0 ],
                       [0,  100, -0 ],
                       [-0,  -0, 100 ],                ])/100.0
 
   if name== "neg" : 
        s0    = np.ones((nasset)) * 100.0
        drift = np.ones((nasset,1)) * 0.0
        vol0  = np.ones((nasset,1)) * 0.2      
        vol0 =  np.array([[ 0.20, 0.15, 0.08 ]]).T 
        correl =  np.array([[100, -30 , -50 ],
                         [ -30,  100, -40 ],
                         [ -50,  -40, 100 ],    ])/100.0
         
   if name== "reg" :       
        s0    = np.ones((nasset)) * 100.0
        drift = [ np.ones((nasset,1)) * 0.0  for i in range(3) ]     
        

        vol0 = [
                 np.array([[ 0.30, 0.20, 0.10 ]]).T ,
                 np.array([[  0.05, 0.1, 0.40  ]]).T ,
                 np.array([[ 0.4, 0.05, 0.10 ]]).T ,
                ]
        
        correl = [ np.array([[100, 0 , 40 ],
                             [ 0,  100, -20 ],
                             [ 40,  -20, 100 ],    ])/100.0 ,
   
                   np.array([[100, 0 , 40 ],
                             [ 0,  100, -20 ],
                             [ 40,  -20, 100 ],    ])/100.0  ,
       
                   np.array([[100, 0 , 40 ],
                             [ 0,  100, -20 ],
                             [ 40,  -20, 100 ],    ])/100.0
        
                 ]

   return drift, vol0, correl      



####################################################################################################
####################################################################################################

class PricingOption(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(PricingOption, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.ones(self._dim) * 100
        self._sigma = 0.30
        self._mu_bar = 0.10
        self._rl = 0.04
        self._rb = 0.06
        self._alpha = 1.0 / self._dim

        nasset = self._dim
        self.s0    = np.ones((nasset)) * 100.0

        self.drift, self.vol0, self.correl = scenario("reg", nasset)


        dd = { "drift": str(self.drift),  "vol0" : str(self.vol0), "correl" : str(self.correl) }        
        json.dump(dd,  open(  export_folder + "param_file.txt", "w") )

        self.allret = np.load( os.path.join(export_folder, 'x_generated.npy') )
        self.ii = 0


    def sample_save(self, num_sample, clayer=0):
        if clayer > -1 :

          nsimul = num_sample
          nasset = self._dim
          nstep = self._num_time_interval
          T = self._num_time_interval * self._delta_t
        
          s0    = self.s0
          drift = self.drift
          vol0  = self.vol0
          correl = self.correl 


          allret, allpaths, bm_process, corrbm, correl_upper_cholesky, iidbrownian = gbm_multi_regime(nsimul, nasset, nstep, T, s0, vol0, drift, 
                    correl, choice="all", regime=[0,1,2])

          np.save(os.path.join(export_folder, 'x_generated.npy'), allret)



    def sample(self, num_sample, clayer=0):
        if clayer > -1 :

          nsimul = num_sample
          nasset = self._dim
          nstep = self._num_time_interval
          T = self._num_time_interval * self._delta_t
        
          s0    = self.s0
          drift = self.drift
          vol0  = self.vol0
          correl = self.correl 

          corrbm = []
          allret = self.allret[ self.ii:self.ii+num_samples,  :,  :]
          self.ii = self.ii + num_samples

          return corrbm, allret




    def sample_multi(self, num_sample, clayer=0):
        if clayer > -1 :

          nsimul = num_sample
          nasset = self._dim
          nstep = self._num_time_interval
          T = self._num_time_interval * self._delta_t
        
          s0    = self.s0
          drift = self.drift
          vol0  = self.vol0
          correl = self.correl 

          allret, allpaths, bm_process, corrbm, correl_upper_cholesky, iidbrownian = gbm_multi(nsimul, nasset, nstep, T, s0, vol0, drift,
                     correl, choice="all",)

          # allret, allpaths, bm_process, corrbm, correl_upper_cholesky, iidbrownian = gbm_multi_regime(nsimul, nasset, nstep, T, s0, vol0, drift,
          #          correl, choice="all", regime=[0,1,2])

          # dw_sample = corrbm
          # x_sample = allpaths
          # return dw_sample, x_sample
          return corrbm, allret




    def sample5(self, num_sample, clayer=0):
        if clayer > -1 :

          nsimul = num_sample
          nasset = self._dim
          nstep = self._num_time_interval
          T = self._num_time_interval * self._delta_t
        
          s0    = self.s0
          drift = self.drift
          vol0  = self.vol0
          correl = self.correl 

          allpaths, bm_process, corrbm, correl_upper_cholesky, iidbrownian = gbm_multi(nsimul, nasset, nstep, T, s0, vol0, drift, 
                    correl, choice="all")

          # dw_sample = corrbm
          # x_sample = allpaths
          # return dw_sample, x_sample
          return corrbm, allpaths
      
        
        
    def sample4(self, num_sample, clayer=0):
        
        if clayer > -1 :
        
          dw_sample = (
             normal.rvs(size=[num_sample, self._dim, self._num_time_interval]) * self._sqrt_delta_t
          )
          x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
          x_sample[:, :, 0] = np.ones([num_sample, self._dim]) * self._x_init
        
        
          factor = np.exp((self._mu_bar - (self._sigma ** 2) / 2) * self._delta_t)
          for t in range(self._num_time_interval):
             x_sample[:, :, t + 1] = factor * np.exp(self._sigma * dw_sample[:, :, t]) * x_sample[ :, :, t] 
          return dw_sample, x_sample

          # for i in xrange(self._n_time):
          # 	x_sample[:, :, i + 1] = (1 + self._mu_bar * self._delta_t) * x_sample[:, :, i] + (
          # 		self._sigma * x_sample[:, :, i] * dw_sample[:, :, i])
        


    def sample2(self, num_sample):
        dw_sample = (
            normal.rvs(size=[num_sample, self._dim, self._num_time_interval]) * self._sqrt_delta_t
        )
        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self._dim]) * self._x_init
        # for i in xrange(self._n_time):
        # 	x_sample[:, :, i + 1] = (1 + self._mu_bar * self._delta_t) * x_sample[:, :, i] + (
        # 		self._sigma * x_sample[:, :, i] * dw_sample[:, :, i])
        factor = np.exp((self._mu_bar - (self._sigma ** 2) / 2) * self._delta_t)
        for i in range(self._num_time_interval):
            x_sample[:, :, i + 1] = (factor * np.exp(self._sigma * dw_sample[:, :, i])) * x_sample[
                :, :, i
            ]
        return dw_sample, x_sample



    def f_tf(self, t, x, y, z):
        temp = tf.reduce_sum(z, 1, keepdims=True) / self._sigma
        return (
            - self._rl * y
            - (self._mu_bar - self._rl) * temp
            + ((self._rb - self._rl) * tf.maximum(temp - y, 0))
        )


    def g_tf(self, t, x):
        temp = tf.reduce_max(x, 1, keepdims=True)
        return tf.maximum(temp - 100, 0)  







####################################################################################################
####################################################################################################



