# -*- coding: utf-8 -*-
import os, sys
# DIRCWD=  r'C:/D/project27_raku/datanalysis/' if sys.platform.find('win')> -1   else  '/home/ubuntu/notebook/' if os.environ['HOME'].find('ubuntu')>-1 else '/media/sf_project27/'
# DIRCWD =  r'C:/D/project27_raku/git_dev/aacredit/' if sys.platform == 'win32' else   r'/mnt/hgfs/project27_raku/git_dev/aacredit/' if sys.platform == 'linux' else ''

sys.path.append(DIRCWD + '/aapackage') 
#os.chdir(DIRCWD); 
#import sqlalchemy as sql, dask.dataframe as dd, dask
import arrow
from time import time
import copy
import gc
import random
import copy


from aapackage import util 
from aapackage import datanalysis as da
#import util_cat as ca
#####################################################################################
#####################################################################################
import numpy as np, gc, pandas as pd, copy
from attrdict import AttrDict as dict2

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


import sklearn as sk
from sklearn import manifold, datasets
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing


from sklearn.pipeline import make_pipeline

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_iris


from collections import OrderedDict


try :
  from catboost import CatBoostClassifier, Pool, cv
  from imblearn.over_sampling import SMOTE, ADASYN  ####Imbalance
  from imblearn.combine import SMOTETomek
  from pylmnn import LargeMarginNearestNeighbor as LMNN
  import seaborn as sns; sns.set()
  import lightgbm as lgb

except Exception as e:
    print(e)


print('os.getcwd', os.getcwd())
#print('DIRCWD', DIRCWD)

from random import random
#####################################################################################
dirsession = r"C:/a/session/"

dirmodel = r"C:/a/model/"

#####################################################################################
def list_replace(ll, dd) :
  ll2 = []
  for t in ll :
    t2 = dd.get(t)
    ll2.append(  t2 if t2 is not None else t)
  return ll2



def toint(x):
   try :
       return int(x)
   except :
       return x


def toint2(x, def_val=-1):
   try :
      return int(x)
   except :
      return def_val


def to_int(x):
    try:
         return int(x)
    except: return -1


def zip_toint(x):
   try :
       return  int( str(x)[:3].replace("-", "") )
   except :
       #print(x)
       return -1


def to_dateint(x) :
  try :  
      v = x.split("/")
      return   int( v[0] )*10000 +  int( v[1] )*100 + int( v[2] )  
  except :
      return -1






####################################################################################################
def model_get(name="", **kw) :
    if name == "RandomForestClassifier" :
       clf = RandomForestClassifier(**kw)  # random_state=0,
       use_eval = 0

    if name == "LGBMClassifier" :
      import lightgbm as lg
      from lightgbm import LGBMClassifier
      clf = LGBMClassifier(**kw)
      use_eval = 1   # train vs eval

    return clf, use_eval



def stat_hist_2d_table(dfu, colx='score_cic', coly='scoress', colx_range=None, coly_range=None,  nbin=10) :

  colx_range = list(np.arange(0, 1000, 100)) if colx_range is None else colx_range
  coly_range = list(np.arange(0, 1000, 100)) if coly_range is None else colx_range


  hist_2d = np.histogram2d(x= dfu[colx].values, y=dfu[coly].values,
                           bins= nbin, range=None, normed=None, weights=None, density=None)[0]

  hist_2d = pd.DataFrame(hist_2d, columns = [ f"{colx}_{x}" for x in colx_range ] ,
                         index   = [ f"{coly}_{x}" for x in coly_range ] ,
                        )
  return hist_2d



def stat_groupby(df, colist = ['demo_gender'],  colref='easy_id') :
  dfstat = None
  for colname in colist :
    df[colname] = df[colname].fillna(-1)
    df1 = df.groupby(colname).agg({ colref : 'count'}).reset_index()
    df1['pd'] = df1[colref] /df1[colref].sum()

    dfstat = pd_stats_merge(dfstat, df1 ) if dfstat is not None else df1
  return dfstat



def pd_stats_merge(df1, df2) :
  # Merge 2 dataframe as string type
  ll = []
  mcol = len(df1.columns)

  ll.append( list( df1.columns) )
  ll = ll + [  list(v)  for v in list( df1.values) ]
  ll.append([ "" ] * mcol)


  ll.append( list( df2.columns) )
  ll = ll + [  list(v)  for v in list( df2.values) ]

  ll = pd.DataFrame(ll)
  return ll




#####################################################################################
def check_dfraw(dfraw) :
  for x in [ 'lifetime_earned_point_transaction_count', 'earned_point_lifetime',
             'ichiba_visit_count_lifetime',
             'lifetime_earned_point_transaction_count'             
           ] :
      print(  x, len( dfraw[dfraw[x] > 0.0 ]) ,  dfraw[x].max() )





def confidence_level( df, lb = [ 0.0, 0.2, 0.3 ,  0.5,   0.7,  0.8, 1.0   ],
                      colscore = "ygms_proba",
                      ytrue="actual_increase", ypred="prediction_increase"  ) :
  ll = []    
  for i in range(0, len(lb)-1)  :  
    dfi =  df[ (df[colscore] >= lb[i]) & ( df[colscore] <= lb[i+1]  ) ]
    a = sk.metrics.accuracy_score( dfi[ ytrue ] , 
                            dfi[ ypred ] , 
                            normalize=True, sample_weight=None)
    ll.append(a)
  return pd.DataFrame( { "range" : lb[:-1], "accuracy" : ll })




def confidence_level_range( df, lb = [ 0.0, 0.2, 0.3 ,  0.5,   0.7,  0.8, 1.0   ],
                     colscore = "ygms_proba",
                     ytrue="actual_increase", ypred="prediction_increase",
                     tlist=[  20160301, 20160601 ]  , coldate="dateint" ) :
  dfall = None
  for t in tlist :
     dftmp =  df[ df[coldate]  == t ]
     dfi = confidence_level( dftmp, 
                  lb = lb,
                  colscore = colscore,
                  ytrue= ytrue , ypred= ypred  ) 

     dfi['date'] = t
     dfi['ntotal'] = len(dftmp)
     dfi['ytrue'] = sum( dftmp[ ytrue ])
     dfall = dfi if dfall is None else pd.concat(( dfall, dfi ))
  return dfall






#####################################################################################
#####################################################################################
def sk_metrics_eval(clf, Xtest, ytest, cv=1, metrics=["f1_macro", "accuracy",
                                                      "precision_macro", "recall_macro"] ) :
  #
  entries = []
  model_name = clf.__class__.__name__
  for metric in  metrics :
    metric_val = cross_val_score(clf_log, Xtest, ytest, scoring= metric, cv=3)
    for i, metric_val_i in enumerate(metric_val):
       entries.append((model_name, i, metric, metric_val_i ))
  cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', "metric", 'metric_val'])
  return cv_df




def save(x, folder, filename=None ) :
     util.save( x, folder  + f"/{filename}.pkl")



def load(filename=None ) :
     return util.load(  f"{filename}.pkl")





def save_model(model, dfeatures, data, folder="", filename="model", glob=None ) :
   d = {'model' : model,
        'features': dfeatures,
        'data' : data }

   path = f"{folder}/{filename}.pkl"
   util.save( d , path )
   print( path )


def save_session(folder , glob ) :
    
  if not os.path.exists(folder) :
    os.makedirs( folder )   
  
  lcheck = [ "<class 'pandas.core.frame.DataFrame'>", "<class 'list'>", "<class 'dict'>",
             "<class 'str'>" ,  "<class 'numpy.ndarray'>" ]  
  lexclude = {   "In", "Out" }
  
  for x, _ in glob.items() :
     if not x.startswith('_') and  x not in lexclude :
        x_type =  str(type(glob.get(x) ))
        if x_type in lcheck or x.startswith('clf')  :
            try :
              print( util.save( glob[x], folder  + "/" + x + ".pkl") )
            except Exception as e:
              print(x, x_type, e)




def pd_colnum_tocat(df, colname=None, colexclude=[], nbin=10, suffix="_bin", method=""):
    """
    Preprocessing.KBinsDiscretizer([n_bins, …])	Bin continuous data into intervals.
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    :param df:
    :param method:
    :return:
    """
    colname = colname if colname is not None else list(df.columns)
    colnew  = []
    for c in colname:
        if c in colexclude:
            continue

        df[c] = df[c].astype(np.float32)
        mi, ma = df[c].min(), df[c].max()
        space = (ma - mi) / nbin
        bins = [mi + i * space for i in range(nbin+1)]
        bins[0] -= 0.0000001

        labels = np.arange(0, len(bins)-1)
        colnew.append( c + suffix )
        df[c + suffix ] = pd.cut(df[c], bins=bins, labels=labels)
        print( c + suffix )
    return df



def na(df, x=None):
    a = df[x].isnull().sum()
    b = len(df)
    print( x, a, b, a/(b + 0.0) , sum( df[x] ))




def model_eval(clf0, df, colX, coly="y", test_size=0.5, istrain=1, use_eval=0 ) :
  clf = copy.deepcopy(clf0)  
  yy  = df[coly].values
  X   = df[colX].values  
  X   = X.reshape(-1,1) if len(colX) == 1  else X
  print("X", X.shape )

  if istrain : 
     X_train, X_test, y_train, y_test = train_test_split( X, yy, test_size=test_size, 
                                                          random_state=42)
     del X, yy
     gc.collect()
     if use_eval  :
       clf.fit(X_train, y_train , eval_set= (X_test, y_test) )
     else :
       clf.fit(X_train, y_train )


     ytest_proba   = clf.predict_proba(X_test)[:, 1]
     ytest_pred    = clf.predict(X_test)
     sk_showmetrics(y_test, ytest_pred, ytest_proba)
     return clf

  else :
     y_proba   = clf.predict_proba(X)[:, 1]
     y_pred    = clf.predict(X)
     sk_showmetrics(yy, y_pred, y_proba)



def show_session(folder) :
  """
     Show Load session      
    
  """      
  for dirpath, subdirs, files in os.walk( folder ):
     for x in subdirs :
         print(x)





def load_session(folder, glob=None) :
  """
     Data Load session      
    
  """
  print(folder)
  for dirpath, subdirs, files in os.walk( folder ):
    for x in files:
       filename = os.path.join(dirpath, x) 
       x = x.replace(".pkl", "")
       try :
         glob[x] = util.load(  filename )
         print(filename) 
       except Exception as e :
         print(filename, e)



def pd_to_onehot(df, colnames) :
 for x in colnames :
   try :   
    nunique = len( df[x].unique() )
    print( x, df.shape , nunique, flush=True)
     
    if nunique > 2  :  
      try :
         df[x] = df[x].astype("int")
      except : pass
   
      df = pd.concat([df , pd.get_dummies(df[x], prefix= x) ],axis=1).drop( [x],axis=1)
      # coli =   [ x +'_' + str(t) for t in  lb.classes_ ] 
      # df = df.join( pd.DataFrame(vv,  columns= coli,   index=df.index) )
      # del df[x]
    else :
      lb = preprocessing.LabelBinarizer()  
      vv = lb.fit_transform(df[x])  
      df[x] = vv
   except : pass
 return df




#####################################################################################
def feature_impt_logis(clf, cols2) :
  #### Do not Sort it breaks the order of input feaure
  dfeatures = pd.DataFrame( { 'feature' :  cols2  ,  'coef' :   clf.coef_[0]  ,
                             'coef_abs' : np.abs(  clf.coef_[0]  )  })
  dfeatures['pos'] = np.arange(0, len(dfeatures))
  return dfeatures



def feature_impt_rf(clf, cols2) :    
  ww = clf.feature_importances_
  print("wweight : ", len(ww))
  dfeatures = pd.DataFrame( { 'feature' :  cols2  ,  
                              'coef' :   ww  ,
                              'coef_abs' : np.abs(  ww  )  })


  dfeatures['pos'] = np.arange(0, len(dfeatures))
  return dfeatures









#####################################################################################
#####################################################################################
def merge_columns( dfm3, ll0 ) :
  dd = {}
  for x in ll0 :
     ll2 = [] 
     for t in dfm3.columns :
       if x in t and t[len(x):len(x)+1] == "_" :
           ll2.append(t)

     dd[x]= ll2
  return dd


def merge_colunns2( dfm3,  l, x0 ) :
  dfz = pd.DataFrame( { 'easy_id' : dfm3['easy_id'].values })  
  for t in l :
     ix  =  t.rfind("_") 
     val = int( t[ix+1:])
     print(ix, t[ix+1:] )
     dfz[t] = dfm3[t].apply( lambda x : val if x > 0 else 0 )

  # print(dfz)
  dfz = dfz.set_index('easy_id')
  dfz[x0] = dfz.iloc[:,:].sum(1)
  for t in dfz.columns :
     if t != x0 :
         del dfz[t]
  return dfz








#####################################################################################
#####################################################################################
def pd_downsample(df, coltarget="y", n1max= 10000, n2max= -1, isconcat=1 ) :    
   """
      DownSampler      
   """
   #n0  = len( df1 )
   #l1  = np.random.choice( len(df1) , n1max, replace=False)
   df1 = df[ df[coltarget] == 0 ].sample(n= n1max)

   #df1   = df[ df[coltarget] == 1 ] 
   #n1    = len(df1 )
   #print(n1)
   n2max = len(  df[ df[coltarget] == 1 ]  ) if n2max == -1 else n2max
   #l1    = np.random.choice(len(df1) , n2max, replace=False)
   df0   = df[ df[coltarget] == 1 ].sample(n= n2max)
   #print(len(df0))

   if isconcat :
     df2 = pd.concat(( df1, df0 ))   
     df2 = df2.sample(frac=1.0, replace=True)
     return df2
    
   else :
     print("y=1", n1max, "y=0", n2max)
     return df0, df1




def ccol(df):
    return list(df.columns)




def plotxy(x,y, color=1, size=1, title= "", xrange=(0, 1000), yrange=(0, 1000), filename=None) :
  color = np.zeros(len(x)) if type(color) == int else color  
  np.append( x, [ xrange[0], xrange[1]] )
  np.append( y, [ yrange[0], yrange[1]] )  
  
  fig, ax = plt.subplots(figsize=(12, 10))
  plt.scatter( x , y,  c= color, cmap="Spectral", s=size)
  plt.xlim( xrange[0], xrange[1] )
  plt.ylim( yrange[0], yrange[1] )
  plt.title(   title, fontsize=11 )
  plt.show()

  if filename :
      print("saveiing on disk", filename)
      fig.savefig(filename, dpi=fig.dpi)


      

def get_dfna_col(dfm2) :
  ll = []
  for x in dfm2.columns :
     nn = dfm2[ x ].isnull().sum()     
     nn = nn + len(dfm2[ dfm2[x] == -1 ])
   
     ll.append(nn)
  dfna_col = pd.DataFrame( {'col': list(dfm2.columns), 'n_na': ll} )
  dfna_col['n_tot'] = len(dfm2)
  dfna_col['pct_na'] = dfna_col['n_na'] / dfna_col['n_tot']
  return dfna_col



def pd_na_per_row(dfm2, colid="easy_id",  n = 10**6, naflag=-1) :
  ll =[]
  for ii,x in dfm2.iloc[ :n, :].iterrows() :
     jj = 0
     for t in x:
       if pd.isna(t) or t == naflag :
         jj = jj +1
     ll.append( jj )
     
  #### nb of NA per row   
  dfna_row = pd.DataFrame( {colid  :  dfm2[colid].values[:n] , 
                            'n_na'  :  ll,
                            'n_ok'  :  len(dfm2.columns) - np.array(ll) } )
  return dfna_row





def get_stat_imbalance(df, colname=None):
 """
   Frequency of Max, Min values

 """   
 ll =  { x : []  for x in   [ 'col', 'xmin_freq', 'nunique', 'xmax_freq' ,'xmax' , 
                              'xmin',  'n', 'n_na', 'n_notna'  ]   }
 
 colname =   df.columns if colname is None else df.columns 
 nn = len(df)
 for x in  colname :
    try : 
     xmin = df[x].min()
     ll['xmin_freq'].append(  len(df[ df[x] < xmin + 0.01 ]) )
     ll['xmin'].append( xmin )
    
     xmax = df[x].max()
     ll['xmax_freq'].append( len(df[ df[x] > xmax - 0.01 ]) )
     ll['xmax'].append( xmax )
    
     n_notna = df[x].count()  
     ll['n_notna'].append(  n_notna   ) 
     ll['n_na'].append(   nn - n_notna  ) 
     ll['n'].append(   nn   ) 
     
     ll['nunique'].append(   df[x].nunique()   ) 
     ll['col'].append(x)
    except : pass


 ll = pd.DataFrame(ll)
 ll['xmin_ratio'] = ll['xmin_freq'] / nn
 ll['xmax_ratio'] = ll['xmax_freq'] / nn
 return ll
    



def cat_correl_cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))



def cat_correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator/denominator
    return eta



def theils_u(x, y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x
    
    
    



#### Calculate KAISO Limit  #########################################################
def get_kaiso_limit(dfm2, col_score='scoress', coldefault="y", ntotal_default=491, def_list=None, nblock=20.0) : 
  if def_list is None :
     def_list = np.ones(21) * ntotal_default / nblock       
  
  dfm2['scoress_bin'] = dfm2[ col_score ].apply(lambda x : np.floor( x / 1.0 ) * 1.0  )
  dfs5 = dfm2.groupby(  'scoress_bin' ).agg( { col_score :  'mean'  ,
                                               coldefault :  {'sum', 'count'}
  }).reset_index()
  dfs5.columns = [ x[0]  if x[0] == x[1] else  x[0] +'_'+ x[1]  for x in  dfs5.columns ] 
  dfs5 = dfs5.sort_values( col_score , ascending=False) 
  # return dfs5

  l2 =  []
  k  =  1
  ndef, nuser = 0 , 0
  for i, x in dfs5.iterrows() :
    if k > nblock : break  
    nuser =  nuser + x[ coldefault + '_count']    
    ndef  =  ndef  + x[ coldefault + '_sum']  
    pdi   =  ndef / nuser 
  
    if ndef > def_list[k-1] :
    #if  pdi > pdlist[k] :
      l2.append( [np.round( x[ col_score ], 1) ,  k, pdi,  ndef, nuser ] )      
      k = k + 1
      ndef, nuser = 0 , 0
  l2.append( [np.round( x[ col_score ], 1 ) ,  k, pdi,  ndef, nuser ] )  
  l2  = pd.DataFrame(l2, columns=[  col_score , 'kaiso3' , 'pd', 'ndef', 'nuser' ] )
  l2['nuser_pct'] = l2['nuser'] / l2['nuser'].sum()
  return l2




##### Get Kaiso limit ###############################################################
def get_kaiso2(x, l1) :
   for i in range(0, len(l1)) :
       if x >= l1[i]  :
           return i+1
   return i + 1+1




#### Histogram    ################################################################################## 
def np_histo(dfm2, bins=[], col0='diff', bin_last=10**9, bin_first=None ) :
  bin_used = copy.deepcopy( bins)
  bin_used[-1] = bin_last if bin_last is not None else bin_used[-1]
  # bin_used[0] =  bin_first if bin_last is not None else bin_used[0]     
      
  hh  = np.histogram( dfm2[ col0 ].values , 
                      bins=bin_used, range=None, normed=None, weights=None, density=None)
  
  hh2 = pd.DataFrame({ 'xall'    : hh[1][:-1] , 
                       'freqall' : hh[0] } )[[ 'xall', 'freqall' ]]
  hh2['densityall'] = hh2['freqall'] / hh2['freqall'].sum()    
  return hh2  




def np_histo2(dfm2, bins=50, col0='diff', col1='y') :
  hh  = np.histogram( dfm2[ col0 ].values , 
                     bins=bins, range=None, normed=None, weights=None, density=None)
  hh2 = pd.DataFrame({ 'xall' : hh[1][:-1] , 
                       'freqall' :hh[0] } )[[ 'xall', 'freqall' ]]
  hh2['densityall'] = hh2['freqall'] / hh2['freqall'].sum()    

        
  hh  = np.histogram( dfm2[ dfm2[ col1 ] == 0 ][ col0 ].values , 
                     bins=bins, range=None, normed=None, weights=None, density=None)
  hh2['x0'] = hh[1][:-1]
  hh2['freq0'] = hh[0]
  hh2['density0'] = hh2['freq0'] / hh2['freq0'].sum()

  
  hh  = np.histogram( dfm2[ dfm2[ col1 ] == 1 ][ col0 ].values , 
                     bins=bins, range=None, normed=None, weights=None, density=None)
  hh2['x1'] = hh[1][:-1]
  hh2['freq1'] = hh[0]
  hh2['density1'] = hh2['freq1'] / hh2['freq1'].sum()
  
  return hh2  


##### Drop duplicates
from collections import OrderedDict
def np_drop_duplicates(l1):
  l0 = list( OrderedDict((x, True) for x in l1 ).keys())
  return l0


def col_extract_colbin( cols2) :
 coln = []
 for ss in cols2 :
  xr = ss[ss.rfind("_")+1:]
  xl = ss[:ss.rfind("_")]
  if len(xr) < 3 :  # -1 or 1
    coln.append( xl )
  else :
    coln.append( ss )     
    
 coln = np_drop_duplicates(coln)
 return coln



def col_stats(df) :
  ll = { 'col' : [], 'nunique' : [] }
  for x in df.columns:
     ll['col'].append( x )
     ll['nunique'].append( df[x].nunique() )
  ll =pd.DataFrame(ll)
  n =len(df) + 0.0
  ll['ratio'] =  ll['nunique'] / n
  ll['coltype'] = ll['nunique'].apply( lambda x :  'cat' if x < 100 else 'num')
  
  return ll





def pd_feat_normalize(dfm2, colnum_log, colproba) :
  for x in [  'SP1b' , 'SP2b'  ] :
    dfm2[x] = dfm2[x] * 0.01

  dfm2['SP1b' ] = dfm2['SP1b' ].fillna(0.5)  
  dfm2['SP2b' ] = dfm2['SP2b' ].fillna(0.5)

  for x in colnum_log :
    try :  
      dfm2[x] =np.log( dfm2[x].values.astype(np.float64)  + 1.1 )
      dfm2[x] = dfm2[x].replace(-np.inf, 0)
      dfm2[x] = dfm2[x].fillna(0)
      print(x, dfm2[x].min(), dfm2[x].max() )
      dfm2[x] = dfm2[x] / dfm2[x].max()
    except :
      pass
          
  
  for x in colproba :
    print(x)  
    dfm2[x] = dfm2[x].replace(-1, 0.5)
    dfm2[x] = dfm2[x].fillna(0.5)
    
  return dfm2



def pd_feat_check( dfm2 ) :
 # print(dfm2['segmentc'].values[0]) 
 for x in dfm2.columns :
    if len( dfm2[x].unique() ) > 2 and dfm2[x].dtype  != np.dtype('O'):
        print(x, len(dfm2[x].unique())  ,  dfm2[x].min() , dfm2[x].max()  )



def split_train(df1, ntrain=10000, ntest=100000, colused=None ) :
  n1  = len( df1[ df1['y'] == 0 ] )
  dft = pd.concat(( df1[ df1['y'] == 0 ].iloc[  np.random.choice( n1 , ntest, False), : ]  , 
                    df1[ (df1['y'] == 1) & (df1['def'] > 201803 ) ].iloc[ : , :]  ))

  X_test = dft[ colused ].values 
  y_test = dft[ 'y'  ].values
  print('test', sum(y_test))

  ######## Train data   
  n1   = len( df1[ df1['y'] == 0 ] )
  dft2 = pd.concat(( df1[ df1['y'] == 0 ].iloc[  np.random.choice( n1 , ntrain, False), : ]  , 
                      df1[ ( df1['y'] == 1) & (df1['def'] > 201703 ) & (df1['def'] < 201804 )  ].iloc[ : , :]  ))
  dft2 = dft2.iloc[ np.random.choice( len(dft2) , len(dft2), False) , : ]

  X_train = dft2[ colused ].values 
  y_train = dft2[ 'y' ].values
  print('train', sum(y_train))
  return X_train, X_test, y_train, y_test 



def split_train2(df1, ntrain=10000, ntest=100000, colused=None, nratio =0.4 ) :
  n1  =  len( df1[ df1['y'] == 0 ] )
  n2  =  len( df1[ df1['y'] == 1 ] ) 
  n2s =  int(n2*nratio)  # 80% of default
      
  #### Test data
  dft = pd.concat(( df1[ df1['y'] == 0 ].iloc[  np.random.choice( n1 , ntest, False), : ]  , 
                    df1[ (df1['y'] == 1)  ].iloc[: , :]  ))

  X_test = dft[ colused ].values 
  y_test = dft[ 'y'  ].values
  print('test', sum(y_test))

  ######## Train data   
  n1   = len( df1[ df1['y'] == 0 ] )
  dft2 = pd.concat(( df1[ df1['y'] == 0 ].iloc[  np.random.choice( n1 , ntrain, False), : ]  , 
                     df1[ (df1['y'] == 1)  ].iloc[ np.random.choice( n2 , n2s, False) , :]   ))
  dft2 = dft2.iloc[ np.random.choice( len(dft2) , len(dft2), False) , : ]

  X_train = dft2[ colused ].values 
  y_train = dft2[ 'y' ].values
  print('train', sum(y_train))
  return X_train, X_test, y_train, y_test 





def feature_impt_logis2(dfm2, cols2) :   
 '''
  Randomized for feature impt.
 '''   
 X_train, _, y_train, _  = split_train2(  dfm2[ cols2 + ['y', 'def', 'score_kaisob',  'SP2b' ]], 
                                          ntrain=60000, ntest=2000,
                                          colused = cols2[:], nratio =0.8 ) 

 clf = sk.linear_model.LogisticRegression( penalty='l1', class_weight= 'balanced' )
 clf.fit(  X_train , y_train  )       
 dfeatures = pd.DataFrame( { 'feature' :  cols2  ,  'coef' :   clf.coef_[0]  ,
                             'coef_abs' : np.abs(  clf.coef_[0]  )  }).sort_values('coef_abs', ascending=False)    
 dfeatures['rank'] = np.arange(0, len(dfeatures))
 dfeatures['origin'] = 1
 
 
 clf = sk.linear_model.LogisticRegressionCV( cv=2,  class_weight= 'balanced' )
 clf.fit(  X_train , y_train  )       
 dfeatures2 = pd.DataFrame( { 'feature' :  cols2  ,  'coef' :   clf.coef_[0]  ,
                             'coef_abs' : np.abs(  clf.coef_[0]  )  }).sort_values('coef_abs', ascending=False)    
 dfeatures2['rank'] = np.arange(0, len(dfeatures2))
 dfeatures2['origin'] = 2
 dfeatures = pd.concat((dfeatures, dfeatures2), axis=0)


 X_train, _, y_train, _  = split_train2(  dfm2[ cols2 + ['y', 'def', 'score_kaisob',  'SP2b' ]], 
                                          ntrain=20000, ntest=2000,
                                          colused = cols2[:], nratio =0.8 ) 

 clf = sk.linear_model.LogisticRegression( penalty='l1', class_weight= 'balanced' )
 clf.fit(  X_train , y_train  )       
 dfeatures2 = pd.DataFrame( { 'feature' :  cols2  ,  'coef' :   clf.coef_[0]  ,
                             'coef_abs' : np.abs(  clf.coef_[0]  )  }).sort_values('coef_abs', ascending=False)    
 dfeatures2['rank'] = np.arange(0, len(dfeatures2))
 dfeatures2['origin'] = 3
 dfeatures = pd.concat((dfeatures, dfeatures2), axis=0) 
 
 clf = sk.linear_model.LogisticRegressionCV( cv=2,  class_weight= 'balanced' )
 clf.fit(  X_train , y_train  )       
 dfeatures2 = pd.DataFrame( { 'feature' :  cols2  ,  'coef' :   clf.coef_[0]  ,
                             'coef_abs' : np.abs(  clf.coef_[0]  )  }).sort_values('coef_abs', ascending=False)    
 dfeatures2['rank'] = np.arange(0, len(dfeatures2))
 dfeatures2['origin'] = 4
 dfeatures = pd.concat((dfeatures, dfeatures2), axis=0)

 return dfeatures




#####################################################################################
def pd_remove(df, cols) :
 for x in  cols :
   try :   
    del df[ x ]
   except : pass
 return df

def sk_showconfusion( Y_train,Y_pred, isprint=True):
  cm = sk.metrics.confusion_matrix(Y_train, Y_pred); cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  if isprint: print(( cm_norm[0,0] + cm_norm[1,1])); print(cm_norm); print(cm)
  return cm, cm_norm, cm_norm[0,0] + cm_norm[1,1]


def sk_feature_importance(clf, cols) :
  dfcol = pd.DataFrame( {  'col' :  cols,  'feat':   clf.feature_importances_
                     } ).sort_values('feat', ascending=0).reset_index()
  dfcol['cum'] = dfcol['feat'].cumsum(axis = 0) 
  colsel = list( dfcol[ dfcol['cum'] < 0.9 ]['col'].values )
  return dfcol, colsel


def sk_showmetrics(y_test, ytest_pred, ytest_proba,target_names=['0', '1']) :
  #### Confusion matrix
  mtest  = sk_showconfusion( y_test, ytest_pred, isprint=False)
  # mtrain = sk_showconfusion( y_train , ytrain_pred, isprint=False)
  auc  =  roc_auc_score( y_test, ytest_proba)   # 
  gini =  2*auc -1
  acc  =  accuracy_score(  y_test, ytest_pred )
  f1macro = sk.metrics.f1_score(y_test, ytest_pred, average='macro')
  
  print('Test confusion matrix')
  print(mtest[0]) ; print(mtest[1])
  print('auc ' + str(auc) )
  print('gini '+str(gini) )
  print('acc ' + str(acc) )
  print('f1macro ' + str(f1macro) )
  print('Nsample ' + str(len(y_test)) )
  
  print(classification_report(y_test, ytest_pred, target_names=target_names))

  # calculate roc curve
  fpr, tpr, thresholds = roc_curve(y_test, ytest_proba)
  plt.plot([0, 1], [0, 1], linestyle='--')
  plt.plot(fpr, tpr, marker='.')
  plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.title('ROC curve')
  plt.show()



def sk_showmetrics2(y_test, ytest_pred, ytest_proba,target_names=['0', '1']) :
  #### Confusion matrix
  # mtest  = sk_showconfusion( y_test, ytest_pred, isprint=False)
  # mtrain = sk_showconfusion( y_train , ytrain_pred, isprint=False)
  auc  =  roc_auc_score( y_test, ytest_proba)   # 
  gini =  2*auc -1
  acc  =  accuracy_score(  y_test, ytest_pred )
  return auc, gini, acc



def clf_prediction_score(clf, df1 , cols, outype='score') :
  def score_calc(yproba , pnorm = 1000.0 ) :
    yy =  np.log( 0.00001 + (1 - yproba )  / (yproba + 0.001) )   
    # yy =  (yy  -  np.minimum(yy)   ) / ( np.maximum(yy) - np.minimum(yy)  )
    # return  np.maximum( 0.01 , yy )    ## Error it bias proba
    return yy


  X_all = df1[ cols ].values 

  yall_proba   = clf.predict_proba(X_all)[:, 1]
  yall_pred    = clf.predict(X_all)
  try :
    y_all = df1[ 'y'  ].values
    sk_showmetrics(y_all, yall_pred, yall_proba)
  except : pass

  yall_score   = score_calc( yall_proba )
  yall_score   = 1000 * ( yall_score - np.min( yall_score ) ) / (  np.max(yall_score) - np.min(yall_score) )

  if outype == 'score' :
      return yall_score
  if  outype == 'proba' :
      return yall_proba, yall_pred




def col_extract(colbin) :
 '''
    Column extraction 
 '''   
 colnew = []
 for x in colbin :
    if len(x) > 2 :
      if   x[-2] ==  "_" :
          if x[:-2] not in colnew : 
            colnew.append(  x[:-2] ) 
          
      elif x[-2] ==  "-"   :
          if  x[:-3] not in colnew :
            colnew.append(  x[:-3]  )
          
      else :
          if x not in colnew :
             colnew.append( x ) 
 return colnew



def col_remove(cols, colsremove) :
  #cols = list(df1.columns)
  '''
  colsremove = [
     'y', 'def',
     'segment',  'flg_target', 'SP1', 'SP2', 'SP3', 'SP4', 'SP5',        
     'score' ,   'segment2' ,
     'scoreb', 'score_kaisob', 'segmentb', 'def_test'
  ]
  colsremove = colsremove + [ 'SP6',  ' score_kaiso'  ]
  '''
  for x in colsremove :
    try :     cols.remove(x)
    except :  pass
  return cols




def col_remove_fuzzy(cols, colsremove) :
  #cols = list(df1.columns)
  '''
  colsremove = [
     'y', 'def',
     'segment',  'flg_target', 'SP1', 'SP2', 'SP3', 'SP4', 'SP5',        
     'score' ,   'segment2' ,
     'scoreb', 'score_kaisob', 'segmentb', 'def_test'
  ]
  colsremove = colsremove + [ 'SP6',  ' score_kaiso'  ]
  '''
  cols3 = []
  for t in cols :
    flag = 0   
    for x in colsremove :
        if x in t :
            flag = 1
            break
    if flag == 0 :
      cols3.append(t)
  return cols3





def pd_filter(dfxx , cols ) :      
  df1  = copy.deepcopy( dfxx[cols + [ 'def' , 'y'] ] )
  df1  = df1[  df1['def'] < 201905  ] 
  df1  = df1[ (df1['def'] > 201703) | (df1['def'] == -1 )  ] 
  return  df1




# Based on Kaggle kernel by Scirpus
def convert_to_ffm(df, fileout, numerics,categories, coltarget='y'):
    currentcode = len(numerics)
    catdict = {}
    catcodes = {}
    # Flagging categorical and numerical fields
    for x in numerics:
         catdict[x] = 0
    for x in categories:
         catdict[x] = 1
    
    nrows    = df.shape[0]
    ncolumns = len(  numerics ) + len( categories )
    with open(str(fileout) + "_ffm.txt", "w") as text_file:

     # Looping over rows to convert each row to libffm format
     for n, r in enumerate(range(nrows)):
         datastring = ""
         datarow = df.iloc[r].to_dict()
         datastring += str(int(datarow[coltarget]))
         
         # For numerical fields, we are creating a dummy field here
         for i, x in enumerate(catdict.keys()):
             if(catdict[x]==0):
                 datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
             else:
         # For a new field appearing in a training example
                 if(x not in catcodes):
                     catcodes[x] = {}
                     currentcode +=1
                     catcodes[x][datarow[x]] = currentcode #encoding the feature
         # For already encoded fields
                 elif(datarow[x] not in catcodes[x]):
                     currentcode +=1
                     catcodes[x][datarow[x]] = currentcode #encoding the feature
                     
                 code = catcodes[x][datarow[x]]
                 datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"

         datastring += '\n'
         text_file.write(datastring)
         
       
import gc       
gc.collect()
# GLB = globals()





#####################################################################################
########## CDNA processing ##########################################################
def pt_segment(x):
 try :    
  if x > 3000 : return 3000
  if x > 1000 : return 1000
  if x > 500 :  return 500  
  if x > 100 :  return 100  
  if x > 50 :  return 50  
  if x > 20 :  return 20  
  if x > 10 :  return 10  
  if x >  5 :  return 5  
  else : return 1
 except :
     return -1









"""

# coding: utf-8
# pylint: disable = invalid-name, C0111
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

print('Loading data...')
# load or create your dataset
df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

print('Starting training...')
# train
gbm = lgb.LGBMRegressor(num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=20)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=5)

print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# feature importances
print('Feature importances:', list(gbm.feature_importances_))


# self-defined eval metric
# f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
# Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_true, y_pred):
    return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False


print('Starting training with custom eval function...')
# train
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=rmsle,
        early_stopping_rounds=5)


# another self-defined eval metric
# f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
# Relative Absolute Error (RAE)
def rae(y_true, y_pred):
    return 'RAE', np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(np.mean(y_true) - y_true)), False


print('Starting training with multiple custom eval functions...')
# train
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=lambda y_true, y_pred: [rmsle(y_true, y_pred), rae(y_true, y_pred)],
        early_stopping_rounds=5)

print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# eval
print('The rmsle of prediction is:', rmsle(y_test, y_pred)[1])
print('The rae of prediction is:', rae(y_test, y_pred)[1])

# other scikit-learn modules
estimator = lgb.LGBMRegressor(num_leaves=31)

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}

gbm = GridSearchCV(estimator, param_grid, cv=3)
gbm.fit(X_train, y_train)

print('Best parameters found by grid search are:', gbm.best_params_)


"""





