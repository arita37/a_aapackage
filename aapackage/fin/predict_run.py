# -*- coding: utf-8 -*-
"""Machine Learning  -- Forecast Model    """
# %load_ext autoreload
# autoreload 2
import os
import sys

import numpy as np
import pandas as pd

import datanalysis
import portfolio as pf
import util

runfile(os.getcwd() + "/aapackage/allmodule.py", wdir="D:/_devs/Python01/project27")


# ---------------------   Data List          --------------------------------
# execfile('D:/_devs/Python01/aapackage/alldata.py')


sym = ["SPY"]
now1 = util.date_now(2)
# qlist, sym= pf.imp_yahoo_getquotes(sym, start="19950110", end=now1)
qlist, sym = pf.imp_sql_getquotes(
    sym, start1="19950110", end1=now1, table1="daily_us_etf"
)  # .imp_yahoo_getquotes(sym, start="19950110", end=now1)
for k, q in enumerate(qlist):
    print(sym[k], q.date.values[0], q.date.values[-1])
# util.save_obj(qlist,'qlist_spy_prediction_1_1995')


# qlist= util.load_obj('qlist_spy_prediction_1_1995')


# ------------  ADD New date ------------------------------------------
"""
q= qlist[0];  row1=  np.array([[
 util.datetime_todate(datetime(2016,07,14)), 2016,7,14,
  1, 215.0, 215.0, 215.0, 215.0, 10000, 215.0],
  [util.datetime_todate(datetime(2016,07,15)), 2016,7,15,
   1, 215.0, 215.0, 215.0, 215.0, 10000, 215.0]], dtype=np.object)
qlist[0]= util.pd_insertrow(q,row1); del q
"""

# Calculate all the Technical Indicators
for k in range(0, len(qlist)):
    q = qlist[k]
    q["date"] = util.datetime_todate(util.dateint_todatetime(q["date"].values))
    q.reset_index(drop=True, inplace=True)
    qlist[k] = pf.rsk_calc_all_TA(q)

util.save_obj(qlist, "qlist_spy_prediction_2_SPY_1995")
del q

# qlist= util.load_obj('qlist_spy_prediction_2_SPY_1995')


# Intersection of dates
date0 = util.pd_date_intersection(qlist)

# Check of the dates
for k, q in enumerate(qlist):
    print(sym[k], date0[0], date0[-1])

# Indicator and Index Selection
select_ind = [
    "MA_200_dist",
    "MA_50_dist",
    "MA_20_dist",
    "MA_5_dist",
    "MA_3_dist",
    "RET_1",
    "RET_2",
    "RET_5",
    "RET_20",
    "RET_60",
    "RMI_14_10",
    "RMI_7_5",
    "STD_120",
    "ndaylow_252",
    "ndistlow_252",
    "ndayhigh_252",
    "ndisthigh_252",
    "qearning_per",
    "qearning_day",
    "optexpiry_day",
]

select_idx = ["SPY"]

# Create State Dataframe with indicator
qspy = qlist[0]
qspy = qspy[qspy.date.isin(date0)]
state = util.pd_array_todataframe(qspy[["date"]].values, ["date"])

for j, qiname in enumerate(select_idx):
    i = util.find(qiname, sym)
    qi = qlist[i]
    for k, colkname in enumerate(select_ind):
        qaux = qi[qi.date.isin(date0)]
        state = util.pd_insertcol(
            state, qiname.replace("=X", "") + "_" + colkname, qaux[colkname].values
        )
del qaux, qlist
util.a_cleanmemory()

print("Save Data in Fille :")
util.save_obj(state, "prediction1/state_spy_prediction_5_onlySPY_19950411")

################################################################################
##################Training Model ###############################################
# -----------Model Storage ------------------------------------------------------
# smodel= np.empty((3000, 10), dtype=np.object)
# smodel= util.np_addrow(smodel, 2000)
# util.save_obj(smodel,'prediction1/smodel_SPY_singleFactor')
# util.save_obj(smodel2,'prediction1/smodel2_SPY_singleFactor')

smodel = util.load_obj("prediction1/smodel_SPY_singleFactor")
smodel2 = util.load_obj("prediction1/smodel2_SPY_singleFactor")


def sk_create_outputclass(q, tt0, predlag):
    tmax = len(q.MA_3.values)
    Yret = q.MA_3[tt0 + predlag :].values / q.MA_3[tt0 : (tmax - predlag)].values - 1
    y2 = [-1 if y0 < retlevel else 1 for y0 in Yret]
    return np.array(y2, dtype=np.float)


############################Loop to find Best Ensemble Model ########################
###### Update/Refresh  with New Data:
# Best Ensemble Voting Classifier:  Optimize the weights
def and1(x, y):
    return np.logical_and(x, y)


def fun_obj(ww):
    Ypred2, _ = datanalysis.sk_votingpredict(estlist[selectk], 0, ww, X_test)
    cm = sk.metrics.confusion_matrix(Y_test, Ypred2)
    # print len(cm),cm
    cm = np.nan_to_num(cm)
    try:
        aux = -(cm[0, 0] + cm[1, 1] - cmref[0, 0] - cmref[1, 1])
        return aux
    except:
        return 5


ttlag = 1  # 1  2  3  4  5  6   7  8  9  10   15  20  40
retlevel = -0.02  # [ -0.01, -0.005,  0.0,  0.005,  0.01  ]
tt0 = 200
ttest = len(qspy.date.values) - tt0 - 20 * 6  # Last 6 months
print("Calibrate the Models :")
for predlag in [1, 2, 3, 4, 5, 10, 15, 20]:
    for retlevel in [-0.04, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.04]:
        vv = np.copy(
            util.np_sortbycol(
                smodel2[and1(smodel2[:, 4] == predlag, smodel2[:, 9] == retlevel)], 3, asc=False
            )
        )
        # for kk in range(0, len(vv) ) :  #util.find(None, vv[:,0])
        #  clfrf= vv[kk,1]
        #  nest= 1 if str(type(clfrf))[8:-2] == 'sklearn.tree.tree.DecisionTreeClassifier'    else  clfrf.n_estimators
        #  print kk, vv[kk,9], vv[kk,4],  vv[kk,3], str(type(clfrf))[16:-2], nest
        #  print "     ",  vv[kk,8],  vv[kk,2][1][0,0], vv[kk,2][1][1,1]

        if len(vv) > 2:
            # -------Create ensemble Model -------------------------------------------------------
            X = state.values
            tmax = np.shape(X)[0]
            X = np.array(X[(tt0) : (tmax - predlag), 1:], dtype=np.float)
            Y = sk_create_outputclass(qspy, tt0, predlag)
            X_test = X[ttest:, :]
            Y_test = Y[ttest:]

            # Voting Classifier
            estlist2, estww = datanalysis.sk_gen_ensemble_weight(vv, 4, 1.0)  # Took 7 best
            estlist = np.copy(estlist2)
            del estlist2

            # Best Single classifier
            Ypredref = estlist[0].predict(X_test)
            cmref = sk.metrics.confusion_matrix(Y_test, Ypredref)
            cm_normref = cmref.astype("float") / cmref.sum(axis=1)[:, np.newaxis]
            if len(cmref) > 1:
                print((predlag, retlevel, cm_normref[0, 0] + cm_normref[1, 1]))
                print(cm_normref)
                print(cmref)

            # Best Voting
            fmin = 1
            for kk in range(2, len(estlist) + 1):  # 1elet --> All 7 Elts
                selectk = np.arange(0, kk)
                bounds = [(0, 1) for k in selectk]
                result = sci.optimize.differential_evolution(fun_obj, bounds, maxiter=25)
                if result.fun < fmin:
                    fmin = result.fun
                    estwwbest = result.x
                    selectkbest = selectk

            Ypred2, _ = datanalysis.sk_votingpredict(estlist[selectkbest], 0, estwwbest, X_test)
            cm = sk.metrics.confusion_matrix(Y_test, Ypred2)
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            if len(cm_norm) > 1 and len(cmref) > 1:
                print(
                    (cm[0, 0] + cm[1, 1] - cmref[0, 0] - cmref[1, 1], cm_norm[0, 0] + cm_norm[1, 1])
                )
                print(cm_norm)
                print((cm), selectk)
                print(selectkbest)
                estwwbest

            # Save Data
            vv[0, 8] = (selectkbest, estwwbest)
            vv[1, 8] = (selectkbest, estwwbest)
            util.save_obj(
                vv, "prediction1/predictor_spy_" + str(predlag) + "day_" + str(retlevel) + "_ret"
            )

#####################################################################################


#####################################################################################
################## Predict with Sub-Class
# -------------------Initialize -----------------------------------------------------
print("Do the Prediction")
state = util.load_obj("prediction1/state_spy_prediction_5_onlySPY_19950411")

Ypredict = util.pd_array_todataframe(
    qspy[["date", "close", "MA_3"]].values[5282:],
    ["date", "close", "MA_3"],
    index1=qspy.index[5282:].values,
)

predlaglist = [10, 15, 20]
rpct = [-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04]
for k in predlaglist:
    aux = []
    for i in rpct:
        aux.append(str(k) + "d" + str(i))
    Ypredict = util.pd_addcol(Ypredict, ["MA3_" + str(k)])
    Ypredict = util.pd_addcol(Ypredict, ["MA3p_" + str(k)])
    Ypredict = util.pd_addcol(Ypredict, ["MA3pr_" + str(k)])
    Ypredict = util.pd_addcol(Ypredict, aux)

predlaglist = [1, 2, 3, 4, 5]
rpct = [-0.01, -0.005, -0.01, 0.0, 0.005, 0.01]
for k in predlaglist:
    aux = []
    for i in rpct:
        aux.append(str(k) + "d" + str(i))
    Ypredict = util.pd_addcol(Ypredict, ["MA3_" + str(k)])
    Ypredict = util.pd_addcol(Ypredict, ["MA3p_" + str(k)])
    Ypredict = util.pd_addcol(Ypredict, ["MA3pr_" + str(k)])
    Ypredict = util.pd_addcol(Ypredict, aux)

# Insert Date
datefut = util.datetime_todate(
    util.date_generatedatetime(
        start=util.datetime_tostring(util.date_add_bdays(state.date.values[-1], 1)), nbday=90
    )
)
Ytab = np.empty((len(datefut), len(Ypredict.columns.values)), dtype=np.object)
Ytab[:, 0] = datefut
Ypredict = util.pd_insertrow(Ypredict, Ytab)
del Ytab, datefut


# ----------------------Prediction ------------------------------------
# Bin Votes methodology, Probability, remove outlier
def addvote(x, k, nmax):
    val = 1
    if k == 0:
        val = 10  # Extreme: Strong Predictor
    if k == nmax:
        val = 7  # Extreme: Strong Predictor  right
    if x == 1:  # Right
        if k == nmax:
            aux = [val if i >= k else 0 for i in range(0, nmax + 1)]
        else:
            aux = [val if i > k else 0 for i in range(0, nmax + 1)]
    elif x == -1:
        aux = [val if i <= k else 0 for i in range(0, nmax + 1)]
    return aux


def findret(r1, kk, rpct):
    if np.isnan(np.sum(r1)):
        return 0
    nmax = len(rpct)
    vote = np.zeros(nmax + 1)
    for k in range(0, nmax):
        vote = vote + addvote(r1[k], k, nmax)
    # print vote
    kmax1, vmax1 = util.np_find_maxpos(vote)

    if kmax1 != nmax:
        kmax2, vmax2 = util.np_find_maxpos(vote[kmax1 + 1 :])
        if vmax2 == vmax1:
            kmax = (kmax1 + kmax2 + kmax1 + 1) / 2
        else:
            kmax = kmax1
    else:
        kmax = kmax1

    if kmax == 0:
        ret0 = median2(-15, rpct[0])  # extreme left
    elif kmax == nmax:
        ret0 = median2(rpct[-1], 15)  # extreme right
    else:
        ret0 = median2(rpct[kmax - 1], rpct[kmax])
    return ret0


# ----------Insert the Prediction Lag, Return Matrix---------------------------------------------------
predlaglist = [10, 15, 20]
rpct = [-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04]
for predlag in predlaglist:
    kcol = util.find("MA3_" + str(predlag), Ypredict.columns.values)
    aux = pf.getret_fromquotes(Ypredict.MA_3.fillna(0).values, predlag)
    Ypredict.iloc[predlag : (predlag + len(aux)), kcol] = aux * 100

    for retlevel in rpct:
        # predlag= 1;  retlevel= -0.005

        tt0 = 200
        ttest = 5282 - tt0
        vv = util.load_obj(
            "prediction1/predictor_spy_" + str(predlag) + "day_" + str(retlevel) + "_ret"
        )
        (selectkbest, estwwbest) = vv[0, 8]

        X = state.values
        tmax = np.shape(X)[0]
        X = np.array(X[(tt0):(tmax), 1:], dtype=np.float)
        X_test = X[ttest:, :]
        Ypred, _ = datanalysis.sk_votingpredict(vv[selectkbest, 1], 0, estwwbest, X_test)

        # Insert predlag-  prediction:
        kcol = util.find(str(predlag) + "d" + str(retlevel), Ypredict.columns.values)
        Ypredict.iloc[predlag : (predlag + len(Ypred)), kcol] = Ypred  # Check in Excel the lag

# Calculate Prediction
qma3 = qspy.MA_3.fillna(0).values


def median2(x1, x2):
    return np.mean(qma3_r1[and1(qma3_r1 > x1, qma3_r1 < x2)])


nmax = len(Ypredict.MA_3.values)
col = Ypredict.columns.values
for t in predlaglist:
    qma3_r1 = pf.getret_fromquotes(qma3, predlag)[5282:]
    for k in range(t, nmax):
        kcol = util.find(str(t) + "d" + str(rpct[0]), col)
        r1 = findret(Ypredict.iloc[k, kcol : (kcol + len(rpct))].values, k, rpct)

        try:
            val = Ypredict.iloc[k - t, 2] * (1 + r1)
            Ypredict.iloc[k, util.find("MA3p_" + str(t), col)] = val
            Ypredict.iloc[k, util.find("MA3pr_" + str(t), col)] = r1 * 100
        except:
            pass

# ----------Insert the Prediction Lag, Return Matrix---------------------------------------------------
predlaglist = [1, 2, 3, 4, 5]
rpct = [-0.01, -0.005, 0.0, 0.005, 0.01]
for predlag in predlaglist:
    kcol = util.find("MA3_" + str(predlag), Ypredict.columns.values)
    aux = pf.getret_fromquotes(Ypredict.MA_3.fillna(0).values, predlag)
    Ypredict.iloc[predlag : (predlag + len(aux)), kcol] = aux * 100

    for retlevel in rpct:
        tt0 = 200
        ttest = 5282 - tt0
        vv = util.load_obj(
            "prediction1/predictor_spy_" + str(predlag) + "day_" + str(retlevel) + "_ret"
        )
        (selectkbest, estwwbest) = vv[0, 8]

        X = state.values
        tmax = np.shape(X)[0]
        X = np.array(X[(tt0):(tmax), 1:], dtype=np.float)
        X_test = X[ttest:, :]
        Ypred, _ = datanalysis.sk_votingpredict(vv[selectkbest, 1], 0, estwwbest, X_test)

        # Insert predlag-  prediction:
        kcol = util.find(str(predlag) + "d" + str(retlevel), Ypredict.columns.values)
        Ypredict.iloc[predlag : (predlag + len(Ypred)), kcol] = Ypred  # Check in Excel the lag

# Calculate Prediction
qma3 = qspy.MA_3.fillna(0).values


def median2(x1, x2):
    return np.mean(qma3_r1[and1(qma3_r1 > x1, qma3_r1 < x2)])


nmax = len(Ypredict.MA_3.values)
col = Ypredict.columns.values
for t in predlaglist:
    qma3_r1 = pf.getret_fromquotes(qma3, predlag)[5282:]
    for k in range(t, nmax):
        kcol = util.find(str(t) + "d" + str(rpct[0]), col)
        r1 = findret(Ypredict.iloc[k, kcol : (kcol + len(rpct))].values, k, rpct)

        try:
            val = Ypredict.iloc[k - t, 2] * (1 + r1)
            Ypredict.iloc[k, util.find("MA3p_" + str(t), col)] = val
            Ypredict.iloc[k, util.find("MA3pr_" + str(t), col)] = r1 * 100
        except:
            pass

# Get Prediction in Table
Ypredict2 = util.pd_array_todataframe(
    Ypredict[["date", "close", "MA_3"]].values,
    ["date", "close", "MA_3"],
    index1=Ypredict.index.values,
)
for t in [1, 2, 3, 4, 5, 10, 15, 20]:
    q = Ypredict.iloc[:, util.find("MA3p_" + str(t), col)].values
    Ypredict2 = util.pd_insertcol(Ypredict2, "MA3p_" + str(t), q)
del q
Ypredict2 = Ypredict2.fillna(0)

# Show Prediction Level :
for predlag in [1, 2, 3, 4, 5, 10, 15, 20]:  # [1,2,3,4,5] :
    for retlevel in [
        -0.04,
        -0.02,
        -0.01,
        -0.005,
        0.0,
        0.005,
        0.01,
        0.02,
        0.04,
    ]:  # [ -0.01, -0.005,  0.0,  0.005,  0.01  ] :
        try:
            tt0 = 200
            ttest = 5282 - tt0
            vv = util.load_obj(
                "prediction1/predictor_spy_" + str(predlag) + "day_" + str(retlevel) + "_ret"
            )
            (selectkbest, estwwbest) = vv[0, 8]

            X = state.values
            tmax = np.shape(X)[0]
            X = np.array(X[(tt0) : (tmax - predlag), 1:], dtype=np.float)
            Y = sk_create_outputclass(qspy, tt0, predlag)
            X_test = X[ttest:, :]
            Y_test = Y[ttest:]

            Ypred, _ = datanalysis.sk_votingpredict(vv[selectkbest, 1], 0, estwwbest, X_test)
            cm = sk.metrics.confusion_matrix(Y_test, Ypred)
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            print(
                (
                    "Lag:" + str(predlag),
                    "Ret:" + str(retlevel),
                    round(cm_norm[0, 0] + cm_norm[1, 1], 3),
                    cm[0, 0],
                    cm[0, 1],
                    cm[1, 0],
                    cm[1, 1],
                )
            )
        except:
            pass
#####################################################################################

print("Saving the models")
util.save_obj(Ypredict, "SPYpredict_table1_" + util.date_now())
util.save_obj(Ypredict2, "SPYpredict_table2_" + util.date_now())

try:
    del X, Y, X_test, Y_test, Ypred, Ypred2, Ypredref, aux, cmref, cm_norm, cm_normref
    del estlist, estww, estwwbest, cm, col, colkname, fmin, kcol, nmax, predlag
    del smodel, smodel2, date0, selectk
    del selectkbest, select_ind, select_idx, sym, t
    del qi, qiname, r1, tt0, ttest, ttlag, val, vv, i, j, k, kk
    del qma3, qma3_r1, result, bounds, retlevel, tmax, state
except:
    pass
