# -*- coding: utf-8 -*-
"""  All module here for include  """
import copy
import math as mth
import os
import re
import sys
from calendar import isleap
from collections import OrderedDict
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy as sci
from bs4 import BeautifulSoup
from matplotlib.collections import LineCollection

from tabulate import tabulate

from . import datanalysis as da
from . import fast
from . import portfolio as pf
from . import util

if sys.platform.find("win") > -1:
    from guidata import qthelpers  # Otherwise Erro with Spyder Save




# Pandas
# from pandas.types.dtypes import CategoricalDtype

# import  ggplot as gg


#####################################################################################


if (
    sys.version.find("Continuum") > 0 and sys.platform.find("linux") > -1
):  # Import the Packages for Ananconda
    import PyGMO as py


#####################################################################################
# ---------------------             --------------------------------------------------


#####################################################################################


#####################################################################################
# ---------------------             --------------------


#####################################################################################


############################################################################
# ---------------------             --------------------
