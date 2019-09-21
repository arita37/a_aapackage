"""

###############################################################################
### Generated on disk: 100k sample of numpy 3D tensors
 python equation.py  --do sample_save --n_sample 100000


### Load from disk and sample generate
python equation.py  --do sample_load  


###############################################################################
4D generation is as follow :
   List of size Clayer (number of layer) of 3D tensor
       [Nsample, Mdim, Time_steps]

  sample == [ 3Dtensor_1, 3Dtensor_2,  3Dtensor_3,  ...   ]


Nsample : Nb of Simulation sample  (ie data sample size) 
Mdim : Nb of assets  (3  in our case)
Time_steps : nb of time steps (ex : 30 , 10 step per year)

Clayer :  We want to enhance input by NEW features : (think of Layer in CNN )
   clayer=1 :    (return1, return2, return3)
   clayer=2 :    (Variance_ret1, variance_ret3, variance_ret3)
   clayer=3 :    (Feature_ret1, Feature_ret3, Feature_ret3)






"""
import sys, os
import numpy as np
# import tensorflow as tf
from scipy.stats import multivariate_normal as normal
from argparse import ArgumentParser
import json


from os.path import join as osp


####################################################################################################
from utils import gbm_multi, gbm_multi_regime
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
   if name== "zero" :   # Zero correl
        s0    = np.ones((nasset)) * 100.0
        drift = np.ones((nasset,1)) * 0.0
        vol0  = np.ones((nasset,1)) * 0.2      
        vol0 =  np.array([[ 0.20, 0.15, 0.08 ]]).T 
        correl =  np.array([[100, 0 , 0 ],
                       [0,  100, -0 ],
                       [-0,  -0, 100 ],                ])*0.01


   if name== "neg" :  # negative correl
        s0    = np.ones((nasset)) * 100.0
        drift = np.ones((nasset,1)) * 0.0
        vol0  = np.ones((nasset,1)) * 0.2      
        vol0 =  np.array([[ 0.20, 0.15, 0.08 ]]).T 
        correl =  np.array([[100, -30 , -50 ],
                         [ -30,  100, -40 ],
                         [ -50,  -40, 100 ],    ])*0.01


   if name== "regime" :
        s0    = np.ones((nasset)) * 100.0
        drift = [ np.ones((nasset,1)) * 0.0  for i in range(3) ]     
        

        vol0 = [
                 np.array([[ 0.30, 0.20, 0.10 ]]).T ,
                 np.array([[  0.05, 0.1, 0.40  ]]).T ,
                 np.array([[ 0.4, 0.05, 0.10 ]]).T ,
                ]
        
        correl = [ np.array([[100, 0 , 40 ],
                             [ 0,  100, -20 ],
                             [ 40,  -20, 100 ],    ])*0.01 ,
   
                   np.array([[100, 0 , 40 ],
                             [ 0,  100, -20 ],
                             [ 40,  -20, 100 ],    ])*0.01  ,
       
                   np.array([[100, 0 , 40 ],
                             [ 0,  100, -20 ],
                             [ 40,  -20, 100 ],    ])*0.01
        
                 ]

   return drift, vol0, correl      



####################################################################################################
####################################################################################################
class PricingOption(Equation):
    def __init__(self, dim, total_time, num_time_interval, clayer=1, num_sample=200000,
                 scenario_choice="neg", filename="x_generated.npy"):
        super(PricingOption, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.ones(self._dim) * 100
        self._sigma = 0.30
        self._mu_bar = 0.10
        self._rl = 0.04
        self._rb = 0.06
        self._alpha = 1.0 / self._dim


        self.num_sample = num_sample
        self.clayer = clayer
        self.filename = filename


        nasset = self._dim
        self.s0    = np.ones((nasset)) * 100.0


        self.scenario_choice = scenario_choice
        self.drift, self.vol0, self.correl = scenario(scenario_choice, nasset)
        self.regime_list = [1,2,3]


        ### Export sampling data confiuration
        dd = { "drift": str(self.drift),  "vol0" : str(self.vol0), "correl" : str(self.correl) }        
        json.dump(dd,  open(  export_folder + "param_file.txt", "w") )

        ####  only if path are pre-generated
        self.ii = 0
        self.allret = None



    def sample_save(self, num_sample, clayer=0, filename=None ):
        if clayer > -1 :

          nsimul = num_sample
          nasset = self._dim
          nstep = self._num_time_interval
          T = self._num_time_interval * self._delta_t
        
          s0    = self.s0
          drift = self.drift
          vol0  = self.vol0
          correl = self.correl 


          if self.scenario_choice in  { "neg", "zero" } :
            allret, allpaths, bm_process, corrbm, correl_upper_cholesky, iidbrownian = gbm_multi(nsimul, nasset, nstep, T, s0, vol0, drift,
                     correl, choice="all",)


          if self.scenario_choice in  { "regime" } :
            allret, allpaths, bm_process, corrbm, correl_upper_cholesky, iidbrownian = gbm_multi_regime(nsimul, nasset, nstep, T, s0, vol0, drift,
                     correl, choice="all", regime=self.regime_list )


          filename = self.filename if filename is  None else filename
          np.save(os.path.join(export_folder, filename), allret)
          os_file_stat( os.path.join(export_folder, filename) )



    def sample_load(self, filename=None):
         # Load from file
         filename = self.filename if filename is  None else filename
         try :
            self.allret = np.load( os.path.join(export_folder, filename) )
         except :
            print("No file", filename, "Generating New one")
            self.sample_save(self.num_sample, self.clayer, filename )
            self.allret = np.load( os.path.join(export_folder, filename) )

         print("Loaded:", filename, self.allret.shape )



    def sample_fromfile(self, num_sample, clayer=0,  filename=None ):
        """
         ValueError: cannot reshape array of size 2976 into shape (32,3,30)

        """
        filename = self.filename if filename is  None else filename
        if self.allret is None :
           self.sample_load(filename)


        if clayer > -1 :
          # Seelect sample
          allret = self.allret[ self.ii:self.ii + num_sample,  :,  :]
          self.ii = self.ii + num_sample

          ## This one is NOT used
          corrbm = np.empty_like(allret)
          corrbm = corrbm[:, :, :-1]  # time step rescale to T

          # print(allret.shape)
          if self.ii < 3 :
              print("xshape", x.shape)


          return corrbm, allret



    def sample(self, num_sample, clayer=0):
        ### Official version
        if clayer > -1 :

          nsimul = num_sample
          nasset = self._dim
          nstep = self._num_time_interval
          T = self._num_time_interval * self._delta_t
        
          s0    = self.s0
          drift = self.drift
          vol0  = self.vol0
          correl = self.correl 


          if self.scenario_choice in  { "neg", "zero" } :
            allret, allpaths, bm_process, corrbm, correl_upper_cholesky, iidbrownian = gbm_multi(nsimul, nasset, nstep, T, s0, vol0, drift,
                     correl, choice="all",)


          if self.scenario_choice in  { "regime" } :
            allret, allpaths, bm_process, corrbm, correl_upper_cholesky, iidbrownian = gbm_multi_regime(nsimul, nasset, nstep, T, s0, vol0, drift,
                     correl, choice="all", regime= self.regime_list)


          # allret, allpaths, bm_process, corrbm, correl_upper_cholesky, iidbrownian = gbm_multi_regime(nsimul, nasset, nstep, T, s0, vol0, drift,
          #          correl, choice="all", regime=[0,1,2])
          # dw_sample = corrbm
          # x_sample = allpaths
          # return dw_sample, x_sample
          # print(allret.shape)

          return corrbm, allret


    def f_tf(self, t, x, y, z):
        import tensorflow as tf
        temp = tf.reduce_sum(z, 1, keepdims=True) / self._sigma
        return (
            - self._rl * y
            - (self._mu_bar - self._rl) * temp
            + ((self._rb - self._rl) * tf.maximum(temp - y, 0))
        )


    def g_tf(self, t, x):
        import tensorflow as tf
        temp = tf.reduce_max(x, 1, keepdims=True)
        return tf.maximum(temp - 100, 0)  


####################################################################################################
####################################################################################################



####################################################################################################
class dict2(object):
    def __init__(self, d):
        self.__dict__ = d


def os_file_stat(filename):
  import time
  (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(filename)
  print( filename,  size/1E6,  ",last modified: %s" % time.ctime(mtime))


def load_argument():
    p = ArgumentParser()
    p.add_argument("--do", type=str, default='train/predict/generate paths')
    p.add_argument("--problem_name", type=str, default='PricingOption')
    p.add_argument("--n_sample", type=int, default=1)
    p.add_argument("--filename", type=str, default='x_generated.npy' )

    arg = p.parse_args()
    return arg



if __name__ == "__main__":
   arg = load_argument()
   print(arg)

   c = dict2({ "dim" : 3, "total_time": 3.0, "num_time_interval": 30,
               "num_sample": 10,  "clayer" : 2
            })

   # from config import get_config, export_folder
   # c = get_config(arg.problem_name)

   from equation import get_equation as get_equation_tf
   bsde = get_equation_tf(arg.problem_name, c.dim, c.total_time, c.num_time_interval)


   ## Basic sampling
   # dw, x = bsde.sample( num_sample= c.num_sample, clayer= c.clayer )
   # print(x, x.shape)


   if arg.do == "sample_save" :
     ## Save sampling on disk
     bsde.sample_save( arg.n_sample, c.clayer, arg.filename )
     sys.exit(0)



   if arg.do == "sample_fromfile" :
     ## Load sampling on disk
     bsde.sample_load( arg.filename)


     ## load sampling disk, bsde.ii is GLOBAL COUNT for sampling
     dw, x = bsde.sample_fromfile( 3, c.clayer )
     print(x, x.shape, bsde.ii)


     dw, x = bsde.sample_fromfile( 4, c.clayer )
     print(x, x.shape, bsde.ii)






"""
    def sample_backup(self, num_sample, clayer=0):
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



"""




