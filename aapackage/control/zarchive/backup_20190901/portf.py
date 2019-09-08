# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

# import matplotlib.pyplot as plt
# plt.style.use('ggplot')

# import seaborn as sns
# sns.set_style("darkgrid")


# from decimal import Decimal
# from agents.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
# from agents.pg import PG










os.environ





"""
Multivariate Garch
    over a time step
    
    
    p. w1 +  (1-p).W2
    
    
    dS/S 
    
    
    dS1/S1 . dS2/S2 = Correl
   
   
    Projection of S1 over Browninan
    S1 is variable.
    
    Use Variance sigma(t) as input.

    W2 = variance



Simulate Path
Correlated path
Brownian Motions



Cost
    Variance :
    
    
Weight at each steps


Xinput = PastReturns
Output :  Weights



Return target



Scenarios :

  regime change
  
  




"""
