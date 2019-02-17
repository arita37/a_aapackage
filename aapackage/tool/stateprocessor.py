# -*- coding: utf-8 -*-
"""
Utilities to calculate States for Stock Selection 
"""
try :     n= len(symfull); del n
except : symfull= []
  
try :     n= len(dateref); del n
except :  dateref= []
  
  
  



#####################################################################################
####################  Stock Selection ###############################################
def sort(x, col, asc): return   util.sortcol(x, col, asc)
def perf(close, t0, t1):  return  100*( close2[:,t1] / close2[:,t0] -1)
def and2(tuple1):  return np.logical_and.reduce(tuple1)



def ff(x) : return util.find(x, symfull)  # 607  

def dd(t) :
 x= util.find(t, dateref)  
 if x==-1 : print('Error date not found') ; return np.nan
 else :  return x


def gap(close, t0, t1, lag):   
 ret= pf.getret_fromquotes(close[:,t0:t1], lag)
 rmin= 100*np.amin(ret, axis=1)
 return rmin

def process_stock(stkstr, show1=1) :
 stklist= stkstr.split(",") 
 for k,x in  enumerate(stklist) : stklist[k]= x.strip().replace('\n','')
 v= list(set(stklist))
 v.sort(); aux=""
 for x in v :  aux= aux+x+"," 
 if aux[0] == ',' : aux= aux[1:]
  
 if show1 : 
  # print  aux + "\n", "   "
  print('http://finviz.com/screener.ashx?v=211&t='+aux)
 return v

def printn(ss) :
 ss= util.sortcol(ss, 71, asc=True) ;
 aux2=[];  aux=""; aux3=""
 for k in range(0, len(ss)) :
   kx= int(ss[k,0]); 
   aux= aux + symfull[kx] +","
   aux3=  aux3 + "'" +  symfull[kx] + "',"
   aux2.append(  symfull[kx] )
  # print  kx, symfull[kx], round(s1[kx,70],2), round(s1[kx,71],2), round(s1[kx,45],1)

 #print "["+aux3[:-1]+"]", "\n"
 print('\n----------------------\n')
 # print aux, "\n"
 print('http://finviz.com/screener.ashx?v=211&t='+aux)
 return aux2


def show(ll): 
 if type(ll)==str: ll= [ll]
 for x in ll :
  k=util.find(x, symfull)
  if k==-1 : print('error') 
  else :
   print(k, x, s1[k,45],  s1[k,42])  
   print("Trend:", s1[k,54] ,  s1[k,58] ,   s1[k,60] , s1[k,62], s1[k,64], s1[k,57] , s1[k,65])    #Trend t+5,10,15 
   print("Ret 5 days:",  s1[k,1], s1[k,24], s1[k,25], s1[k,26], s1[k,2])  #Return
   print("Daily 5 days:", s1[k,1], s1[k,27], s1[k,28], s1[k,29], s1[k,30])  #Return

   print("DistFromMinMax: ",  s1[k,68], s1[k,69])  
 
   print("Ret 20 days:",   s1[k,3],s1[k,4], s1[k,5],   s1[k,6] , s1[k,7],   s1[k,8])   #Return

   print("Trend Max: 200d",  s1[k,70], s1[k,71])  
   print("Trend Min: 200d",  s1[k,72], s1[k,73])  
   print("LowerBand, Price, TopBand 120 days", s1[k,91], s1[k,45], s1[k,90]) 
   print("LowerBand, Price, TopBand 200 days", s1[k,94], s1[k,45], s1[k,93]) 
   print("LowerBand, Price, TopBand 300 days", s1[k,97], s1[k,45], s1[k,96]) 
   print("MA20,MA50, RMI ", s1[k,30], s1[k,31], s1[k,32]) 
   
   print("Trend Max: 100d",  s1[k,74], s1[k,75])  
   print("Trend Min: 100d",  s1[k,76], s1[k,77], "\n-----")   

#################################################################################




#####################################################################################
######################  Decision Tree  For Stock Selection ##########################
def get_treeselect(stk, s1, xnewdata=None, newsample=5, show1=1, nbtree=5, depthtree=10, debug=0):
 ll1= process_stock(stk, show1)
 select1= [  util.find(x, symfull) for x in ll1]  
 Xtrain=  s1[:,1:]  
 Ytrain=  np.array([ 1 if  i in select1  else 0 for i in range(0, np.shape(s1)[0])] )

 if not xnewdata is None :
   for vv in xnewdata :
     Xtrain= np.row_stack((Xtrain, vv[:,1:]))
     ynew= np.ones(np.shape(vv)[0])
     Ytrain= np.concatenate((Ytrain, ynew))

 if debug: print(np.max(Xtrain), np.min(Xtrain))

 clfrfmin, cmin, c0= util.sk_tree(Xtrain, Ytrain, nbtree, depthtree, 0)
 errmin=20; diversity= 0; divermax= newsample #max(10, np.sum(Ytrain) /3)
 for k in range(0,500):
   clfrf, c1, c0= util.sk_tree(Xtrain, Ytrain, nbtree,depthtree, 0, 1)
   Ystock= clfrf.predict(Xtrain)

   if c1[0,1] >= diversity and c1[0,1] < divermax :  #choose more stock
     diversity= c1[0,1]
     if c1[1,0]  <= errmin :   #choose same stocks
       errmin= c1[1,0]; clfrfmin= clfrf; cmin= c1
       #print(cmin)
 print(cmin)
 return clfrfmin


def store_patternstate(tree, sym1, theme) :
 lstate=[]
 for x in sym1 :
   kid= util.find(x, symfull)
   lstate.append(s1[kid,:])

 lstate= np.array(lstate, dtype=np.float32)
 name1= 'stat_stk_pattern_'+theme+'_'+ str(dateref[-1])
 
 aux= (tree, dateref[-1],sym1, lstate) 
 util.save_obj(aux, name1)
 print(name1)


def load_patternstate(name1):
  tree, date, sym1, lstate= util.load_obj(name1)
  return tree, date, sym1, lstate
  

def get_stocklist(clf, s11, initial, show1=1):
 ll0= process_stock(initial, show1=0)
 Ystock= clf.predict(s11[:,1:])
 aux=''; laux=[]
 for k,x in enumerate(Ystock):
  if x != 0.0 : 
    aux= aux  + str(symfull[k]) +','
    laux.append(str(symfull[k]))    
  
 if show1: print('Full_List: ');  print(aux)
 aux2= list(set(laux).difference(ll0))
 aux2.sort()
 if show1: print('\nNew ones:');   print(",".join(list(aux2)))
 aux3= ",".join(list(aux2))
 return aux3

#####################################################################################
#####################################################################################






############### Database  ###########################################################
#------------------------------------------------------------------------------------

def add_position(longlist, shortlist, name1= 'stock_select_history_v1') :
  t1= util.date_now()
  select_hist= util.load_obj(name1)
  kid= util.find(None, select_hist[:,0])  
  
  select_hist[kid,0]= kid
  select_hist[kid,1]= int(t1)
  select_hist[kid,2]= int(t1)
  select_hist[kid,4]= "long"
  select_hist[kid,3]= longlist  #Buy List  

  kid= util.find(None, select_hist[:,0])  
  
  select_hist[kid,0]= kid
  select_hist[kid,1]= int(t1)
  select_hist[kid,2]= int(t1)
  select_hist[kid,4]= "short"
  select_hist[kid,3]= shortlist  #Buy List  

  util.save_obj(select_hist, name1)
  
  print((kid, t1, name1))
  return select_hist  
  


def add_stock(l1, type1='',comment="", name1= 'stock_select_history_v1', date1= dateref):
  t1= util.date_now()
  select_hist= util.load_obj(name1)
  kid= util.find(None, select_hist[:,0])  
  
  select_hist[kid,0]= kid
  select_hist[kid,1]= int(t1)
  select_hist[kid,2]= int(date1[-1])
  select_hist[kid,4]= type1
  select_hist[kid,5]= comment
  
  if isinstance(l1,str) :
    l1= l1.split(',')
  else: 
    l1= np.reshape(l1,(len(l1)/3,3))   
  select_hist[kid,3]= copy.deepcopy(l1)    #Buy List

  util.save_obj(select_hist, name1)
  
  print((kid, t1, name1))
  return select_hist
 
 
def show_select_history( type1='buy'):
 aux=""
 for vv in select_hist :
   if vv is not None :
    if vv[4] == type1 : 
      print(vv[2],":", ",".join(vv[3]).replace('\n',''))
      aux=  aux + ",".join(vv[3]).replace('\n','')
 
 aux2= list(set(aux.split(',')))
 print("\n", ",".join(aux2))


 
def generate_stock_list(l1): 
 for i in range(0,15) :
  tree_buy= get_treeselect(l1 , s1, newsample=25, nbtree=2, depthtree=7)    
  l1= l1 +","+get_stocklist(tree_buy, s1, l1,show1=0)                
  ll1= l1.split(",")
  if len(ll1) > 30 : break


def export_tocsv(l1) :
  print() 
  l2= l1.split(',')
  l2= [ x.replace('\n','').strip() for x in l2]
  l2= util.pd_createdf(l2, ['symbol'])
  l2.to_csv('E:\desktop/__stock_upload.csv')
  print('E:\desktop/__stock_upload.csv')



#####################################################################################



#####################################################################################
####################  Pqttern Detection in Past dta #################################
from numba import jit, int32, float32

@jit 
def cond_buy(v,t) : 
 pt= v[t]  
 if ( 0.02 < v[t+3]/pt-1 < 0.15 and v[t+5]/pt-1 > 0.06 and   v[t+10] / v[t] -1.0 > 0.07   ) :
   return True
   
@jit 
def cond_buy2(v,t) : 
 pt= v[t]  
 if ( 0.02 < v[t+3]/pt-1 < 0.06 and v[t+5]/pt-1 > 0.04 and  v[t+10] / v[t] -1.0 > 0.05   ) :
   return True


@jit 
def cond_buy3(v,t) : 
 pt= v[t]  
 if ( 0.02 < v[t+3]/pt-1 < 0.08 and v[t+5]/pt-1 > 0.03 and  0.04 < v[t+7] / v[t] -1.0 < 0.20   ) :
   return True
 else: return False
 
   
@jit  
def selection_stock(close2, criteria1):
 masset, tmax= np.shape(close2)
 stk_select=[]
 for t in range(0,tmax-10):
  for k in range(0, masset):
     if criteria1(close2[k,:],t) : stk_select.append([t,k])
 return np.array(stk_select)


def getprediction(Ytest, symfull):
 aux=''
 for k,x in enumerate(Ytest):
  if x != 0 :  
     aux= aux+ symfull[k] + ','
 print(aux)
# return aux
 
def dd(t) :
 x= util.find(t, dateref)  # 607
 if x==-1 : print('Error date not found') ; return np.nan
 else :  return x

 
def getYpositivecase(stk_select, t, dateref) :
  t1= util.find(t, dateref)
  stkselect_t= stk_select[stk_select[:,0]==t1 ]
  Ytrain= np.zeros(np.shape(statt)[0], dtype= np.float16)
  Ytrain[stkselect_t]=1
  return Ytrain

#####################################################################################
#####################################################################################
#---------------------             --------------------






























