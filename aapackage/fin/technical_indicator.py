# -*- coding: utf-8 -*-
######################Technical Indicator ###########################################
import calendar
import datetime
import operator
from datetime import datetime

import numpy as np
import pandas as pd
import scipy as sci


def np_find(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1


def np_find_minpos(values):
    min_index, min_value = min(enumerate(values), key=operator.itemgetter(1))
    return min_index, min_value


def np_find_maxpos(values):
    max_index, max_value = max(enumerate(values), key=operator.itemgetter(1))
    return max_index, max_value


def date_earningquater(t1):  # JP Morgan Qearing date
    quater = qdate = None
    if (t1.month == 10 and t1.day >= 14) or (t1.month == 1 and t1.day < 14) or t1.month in [11, 12]:
        if t1.month in [10, 11, 12]:
            qdate = datetime(t1.year + 1, 1, 14)
        else:
            qdate = datetime(t1.year, 1, 14)
        quater = 4

    if (t1.month == 1 and t1.day >= 14) or (t1.month == 4 and t1.day < 14) or t1.month in [2, 3]:
        qdate = datetime(t1.year, 4, 14)
        quater = 1

    if (t1.month == 4 and t1.day >= 14) or (t1.month == 7 and t1.day < 14) or t1.month in [5, 6]:
        qdate = datetime(t1.year, 7, 14)
        quater = 2

    if (t1.month == 7 and t1.day >= 14) or (t1.month == 10 and t1.day < 14) or t1.month in [8, 9]:
        qdate = datetime(t1.year, 10, 14)
        quater = 3

    nbday = (qdate - t1).days
    return quater, nbday, qdate


def date_option_expiry(date):
    day = 21 - (calendar.weekday(date.year, date.month, 1) + 2) % 7
    if date.day <= day:
        nbday = day - date.day
        datexp = datetime(date.year, date.month, day)
    else:
        if date.month == 12:
            day = 21 - (calendar.weekday(date.year + 1, 1, 1) + 2) % 7
            datexp = datetime(date.year + 1, 1, day)
        else:
            day = 21 - (calendar.weekday(date.year, date.month + 1, 1) + 2) % 7
            datexp = datetime(date.year, date.month + 1, day)

        nbday = (datexp - date).days

    return nbday, datexp


def linearreg(a, *args):
    x = args[1]
    y = args[2]
    b = args[0]
    v = a * x + b - y
    return np.sum(v * v)


def np_sortbycolumn(arr, colid, asc=True):
    df = pd.DataFrame(arr)
    arr = df.sort(colid, ascending=asc)
    return arr.values


def np_findlocalmax(v):
    n = len(v)
    v2 = np.zeros((n, 10))
    if n > 2:
        for i, x in enumerate(v):
            if i < n - 1 and i > 0:
                if x > v[i - 1] and x > v[i + 1]:
                    v2[i, 0] = i
                    v2[i, 1] = x
        v2 = np_sortbycolumn(v2, 1, asc=False)
        return v2
    else:
        max_index, max_value = np_find_maxpos(v)
        return [[max_index, max_value]]


def findhigher(item, vec):
    for i in range(len(vec)):
        if item < vec[i]:
            return i
    return -1


def findlower(item, vec):
    for i in range(len(vec)):
        if item > vec[i]:
            return i
    return -1


def np_findlocalmin(v):
    n = len(v)
    v2 = np.zeros((n, 10))
    if n > 2:
        i2 = 0
        for i, x in enumerate(v):
            if i < n - 1 and i > 0:
                if x < v[i - 1] and x < v[i + 1]:
                    v2[i2, 0] = i
                    v2[i2, 1] = x
                    i2 += 1
        v2 = np_sortbycolumn(v2[:i2], 0, asc=True)
        return v2
    else:

        max_index, max_value = np_find_minpos(v)
        return [[max_index, max_value]]


#####################################################################################
# Finviz Style Support and Resistance
# noinspection PyTypeChecker,PyUnresolvedReferences
def supportmaxmin1(df1):
    df = df1.close.values
    qmax = np_findlocalmax(df)
    t1 = len(df)
    tmax, _ = np_find_maxpos(df)
    # Classification of the Local Max
    for k in range(0, len(qmax)):
        kmax = qmax[k, 0]
        kmaxl = findhigher(qmax[k, 1], df[:kmax][::-1])  # Find same level of max
        kmaxr = findhigher(qmax[k, 1], df[kmax + 1 :])

        kmaxl = 0 if kmaxl == -1 else kmax - kmaxl
        kmaxr = t1 if kmaxr == -1 else kmaxr + kmax

        qmax[k, 2] = np.abs(kmaxr - kmaxl)  # Range
        qmax[k, 3] = np.abs(kmax - tmax)  # Range of the Max After
        qmax[k, 4] = 0  # Range of the Max After
        qmax[k, 5] = kmax - kmaxl
        qmax[k, 6] = kmaxr - kmax

    qmax = np_sortbycolumn(qmax, 1, asc=False)
    tmax = qmax[0, 0]
    pmax = qmax[0, 1]

    # Trend Line Left:  t=0 to tmax
    qmax2 = qmax[qmax[:, 2] > 20, :]  # Range of days where max
    qmax2 = qmax2[qmax2[:, 0] <= tmax, :]  # Time BEfore

    if len(qmax2) > 10:
        qmax2 = qmax2[qmax2[:, 5] >= 10, :]  # Time After
        qmax2 = qmax2[qmax2[:, 6] >= 10, :]  # Time After

    qmax2 = np_sortbycolumn(qmax2, 2, asc=False)  # Order by Time
    qmax2 = qmax2[0:3]  # Only Top 3 Max Value
    qmax2 = np_sortbycolumn(qmax2, 0, asc=True)  # Order by Time

    if np.shape(qmax2)[0] > 1:
        tt = np.arange(0, tmax)
        b = pmax
        res = sci.optimize.fmin(linearreg, 0.1, (b, qmax2[:, 0] - tmax, qmax2[:, 1]), disp=False)
        a = min((df[0] - b) / (0 - tmax), res[0])  # Constraint on the Slope
        suppmax1 = a * (tt - tmax) + b
    else:
        suppmax1 = np.zeros(tmax) + pmax

    # Trend Line Right Side from tmax and t1
    qmax2 = qmax[qmax[:, 2] > 20, :]  # Level of days where max is max
    qmax2 = qmax2[qmax2[:, 0] >= qmax2[0, 0], :]  # Time After
    qmax2 = qmax2[qmax2[:, 5] >= 10, :]  # Time After
    qmax2 = qmax2[qmax2[:, 6] >= 10, :]  # Time After
    qmax2 = np_sortbycolumn(qmax2, 2, asc=False)  # Order by Time

    qmax2 = qmax2[0:3]  # Only Top 3 Max Value
    qmax2 = np_sortbycolumn(qmax2, 0, asc=True)  # Order by Time

    if np.shape(qmax2)[0] > 1:
        tt = np.arange(tmax, t1)
        b = pmax
        res = sci.optimize.fmin(linearreg, 0.1, (b, qmax2[:, 0] - tmax, qmax2[:, 1]), disp=False)
        a = max((df[t1 - 1] - b) / (t1 - tmax), res[0])  # Constraint on the Slope
        suppmax2 = a * (tt - tmax) + b
    else:
        suppmax2 = np.zeros(t1 - tmax) + qmax2[-1, 1]

    suppmax = np.zeros(t1)
    suppmax[0:tmax] = suppmax1
    suppmax[tmax:] = suppmax2

    ##############################################################################
    ##### Local Min Trend Line
    qmin = np_findlocalmin(df)
    t1 = len(df)
    tmin, _ = np_find_minpos(df)

    # Classification of the Local min
    for k in range(0, len(qmin)):
        if qmin[k, 1] != 0.0:
            kmin = qmin[k, 0]
            kminl = findlower(qmin[k, 1], df[:kmin][::-1])  # Find same level of min
            kminr = findlower(qmin[k, 1], df[kmin + 1 :])

            kminl = 0 if kminl == -1 else kmin - kminl
            kminr = t1 if kminr == -1 else kminr + kmin

            qmin[k, 2] = np.abs(kminr - kminl)  # Range
            qmin[k, 3] = np.abs(kmin - tmin)  # Range of the min After
            qmin[k, 4] = 0  # Range of the min After
            qmin[k, 5] = kmin - kminl
            qmin[k, 6] = kminr - kmin

    qmin = np_sortbycolumn(qmin, 1, asc=True)
    tmin = qmin[0, 0]
    pmin = qmin[0, 1]

    # Trend Line Left:  t=0 to tmin
    qmin2 = qmin[qmin[:, 2] > 20, :]  # Range of days where min
    qmin2 = qmin2[qmin2[:, 0] <= tmin, :]  # Time BEfore

    if len(qmin2) > 10:
        qmin2 = qmin2[qmin2[:, 5] >= 10, :]  # Time After
        qmin2 = qmin2[qmin2[:, 6] >= 10, :]  # Time After

    qmin2 = np_sortbycolumn(qmin2, 2, asc=False)  # Order by Time
    qmin2 = qmin2[0:3]  # Only Top 3 min Value
    qmin2 = np_sortbycolumn(qmin2, 0, asc=True)  # Order by Time

    if np.shape(qmin2)[0] > 1:
        tt = np.arange(0, tmin)
        b = pmin
        res = sci.optimize.fmin(linearreg, 0.1, (b, qmin2[:, 0] - tmin, qmin2[:, 1]), disp=False)
        a = max((df[0] - b) / (0 - tmin), res[0])  # Constraint on the Slope
        suppmin1 = a * (tt - tmin) + b
    else:
        suppmin1 = np.zeros(0, tmin) + qmin2[-1, 1]

    # Trend Line Right Side from tmin and t1
    qmin2 = qmin[qmin[:, 2] > 20, :]  # Level of days where min is min
    qmin2 = qmin2[qmin2[:, 0] >= qmin2[0, 0], :]  # Time After
    qmin2 = qmin2[qmin2[:, 5] >= 10, :]  # Time After
    qmin2 = qmin2[qmin2[:, 6] >= 10, :]  # Time After
    qmin2 = np_sortbycolumn(qmin2, 2, asc=False)  # Order by Time

    qmin2 = qmin2[0:3]  # Only Top 3 min Value
    qmin2 = np_sortbycolumn(qmin2, 0, asc=True)  # Order by Time

    if np.shape(qmin2)[0] > 1:
        tt = np.arange(tmin, t1)
        b = pmin
        res = sci.optimize.fmin(linearreg, 0.1, (b, qmin2[:, 0] - tmin, qmin2[:, 1]), disp=False)
        a = min((df[t1 - 1] - b) / (t1 - tmin), res[0])  # Constraint on the Slope
        suppmin2 = a * (tt - tmin) + b
    else:
        suppmin2 = np.zeros(t1 - tmin) + qmin2[0, 1]

    suppmin = np.zeros(t1)
    suppmin[0:tmin] = suppmin1
    suppmin[tmin:] = suppmin2

    suppmax = pd.Series(suppmax, name="suppmax_1")
    df1 = df1.join(suppmax)
    suppmin = pd.Series(suppmin, name="suppmin_1")
    df1 = df1.join(suppmin)

    return df1


# RETURN
def ret(df, n):
    n = n + 1
    m = df["close"].diff(n - 1)
    n2 = df["close"].shift(n - 1)
    roc = pd.Series(100 * m / n2, name="RET_" + str(n - 1))
    df = df.join(roc)
    return df


def qearning_dist(df):
    d1 = df["date"].values
    quarter = np.zeros(len(d1))
    nbday = np.zeros(len(d1))
    for i, t in enumerate(d1):
        q1, nday, _ = date_earningquater(datetime(t.year, t.month, t.day))
        quarter[i] = q1
        nbday[i] = nday
    smin1 = pd.Series(quarter, name="qearning_per")
    smin2 = pd.Series(nbday, name="qearning_day")
    df = df.join(smin1)
    df = df.join(smin2)
    return df


def optionexpiry_dist(df):
    d1 = df["date"].values
    nbday = np.zeros(len(d1))
    for i, t in enumerate(d1):
        nday, _ = date_option_expiry(datetime(t.year, t.month, t.day))
        nbday[i] = nday
    smin2 = pd.Series(nbday, name="optexpiry_day")
    df = df.join(smin2)
    return df


def nbtime_reachtop(df, n, trigger=0.005):
    """nb of days from 1 year low """
    close = df["close"].values
    nnbreach = np.zeros(len(close))
    for i in range(n, len(close)):
        kid, max1 = np_find_maxpos(close[i - n : i])
        dd = np.abs(close[i - n : i] / max1 - 1)
        nnbreach[i] = -np.sum(np.sign(np.minimum(0, dd - trigger)))

    smin1 = pd.Series(nnbreach, name="nbreachigh_" + str(n))
    df = df.join(smin1)
    return df


def distance_day(df, tk, tkname):
    tk = datetime.date(tk)
    date1 = df["date"].values
    dist = np.zeros(len(date1))
    for i in range(0, len(date1)):
        dist[i] = (date1[i] - tk).days
    dist = pd.Series(dist, name="days_" + tkname)
    df = df.join(dist)
    return df


def distance(df, ind):
    df2 = pd.Series(100 * (df["close"] / df[ind] - 1), name=ind + "_dist")
    df = df.join(df2)
    return df


# Moving Average
def ma(df, n):
    ma = pd.Series(pd.rolling_mean(df["close"], n), name="MA_" + str(n))
    df = df.join(ma)
    return df


# Exponential Moving Average
def ema(df, n):
    ema = pd.Series(pd.ewma(df["close"], span=n, min_periods=n - 1), name="EMA_" + str(n))
    df = df.join(ema)
    return df


# Momentum
def mom(df, n):
    m = pd.Series(df["close"].diff(n), name="Momentum_" + str(n))
    df = df.join(m)
    return df


# Rate of Change
def roc(df, n):
    m = df["close"].diff(n - 1)
    n = df["close"].shift(n - 1)
    roc = pd.Series(m / n, name="ROC_" + str(n))
    df = df.join(roc)
    return df


# Average True Range
def atr(df, n):
    i = 0
    tr_l = [0]
    while i < df.index[-1]:
        tr = max(df.get_value(i + 1, "high"), df.get_value(i, "close")) - min(
            df.get_value(i + 1, "low"), df.get_value(i, "close")
        )
        tr_l.append(tr)
        i = i + 1
    tr_s = pd.Series(tr_l)
    atr = pd.Series(pd.ewma(tr_s, span=n, min_periods=n), name="ATR_" + str(n))
    df = df.join(atr)
    return df


# Bollinger Bands
# noinspection PyTypeChecker
def bbands(df, n):
    ma = pd.Series(pd.rolling_mean(df["close"], n))
    msd = pd.Series(pd.rolling_std(df["close"], n))
    b1 = 4 * msd / ma
    b12 = pd.Series(b1, name="BollingerB_" + str(n))
    df = df.join(b12)
    b2 = (df["close"] - ma + 2 * msd) / (4 * msd)
    b22 = pd.Series(b2, name="Bollinger%b_" + str(n))
    df = df.join(b22)
    return df


# Pivot Points, Supports and Resistances
# noinspection PyTypeChecker
def ppsr(df):
    pp = pd.Series((df["high"] + df["low"] + df["close"]) / 3)
    r1 = pd.Series(2 * pp - df["low"])
    s1 = pd.Series(2 * pp - df["high"])
    r2 = pd.Series(pp + df["high"] - df["low"])
    s2 = pd.Series(pp - df["high"] + df["low"])
    r3 = pd.Series(df["high"] + 2 * (pp - df["low"]))
    s3 = pd.Series(df["low"] - 2 * (df["high"] - pp))
    psr = {"PP": pp, "R1": r1, "S1": s1, "R2": r2, "S2": s2, "R3": r3, "S3": s3}
    psr = pd.DataFrame(psr)
    df = df.join(psr)
    return df


# Stochastic oscillator %K
def stok(df):
    s_ok = pd.Series((df["close"] - df["low"]) / (df["high"] - df["low"]), name="SO%k")
    df = df.join(s_ok)
    return df


# Stochastic oscillator %D
def sto(df, n):
    s_ok = pd.Series((df["close"] - df["low"]) / (df["high"] - df["low"]), name="SO%k")
    s_od = pd.Series(pd.ewma(s_ok, span=n, min_periods=n - 1), name="SO%d_" + str(n))
    df = df.join(s_od)
    return df


# Trix
def trix(df, n):
    ex1 = pd.ewma(df["close"], span=n, min_periods=n - 1)
    ex2 = pd.ewma(ex1, span=n, min_periods=n - 1)
    ex3 = pd.ewma(ex2, span=n, min_periods=n - 1)
    i = 0
    roc_l = [0]
    while i + 1 <= df.index[-1]:
        roc = (ex3[i + 1] - ex3[i]) / ex3[i]
        roc_l.append(roc)
        i = i + 1
    trix = pd.Series(roc_l, name="Trix_" + str(n))
    df = df.join(trix)
    return df


# Average Directional Movement Index
def adx(df, n, n_adx):
    i = 0
    up_i = []
    do_i = []
    while i + 1 <= df.index[-1]:
        up_move = df.get_value(i + 1, "high") - df.get_value(i, "high")
        do_move = df.get_value(i, "low") - df.get_value(i + 1, "low")
        if up_move > do_move and up_move > 0:
            up_d = up_move
        else:
            up_d = 0
        up_i.append(up_d)
        if do_move > up_move and do_move > 0:
            do_d = do_move
        else:
            do_d = 0
        do_i.append(do_d)
        i = i + 1
    i = 0
    tr_l = [0]
    while i < df.index[-1]:
        tr = max(df.get_value(i + 1, "high"), df.get_value(i, "close")) - min(
            df.get_value(i + 1, "low"), df.get_value(i, "close")
        )
        tr_l.append(tr)
        i = i + 1
    tr_s = pd.Series(tr_l)
    atr = pd.Series(pd.ewma(tr_s, span=n, min_periods=n))
    up_i = pd.Series(up_i)
    do_i = pd.Series(do_i)
    pos_di = pd.Series(pd.ewma(up_i, span=n, min_periods=n - 1) / atr)
    neg_di = pd.Series(pd.ewma(do_i, span=n, min_periods=n - 1) / atr)
    adx = pd.Series(
        pd.ewma(abs(pos_di - neg_di) / (pos_di + neg_di), span=n_adx, min_periods=n_adx - 1),
        name="ADX_" + str(n) + "_" + str(n_adx),
    )
    df = df.join(adx)
    return df


# macd, macd Signal and macd difference
def macd(df, n_fast, n_slow):
    ema_fast = pd.Series(pd.ewma(df["close"], span=n_fast, min_periods=n_slow - 1))
    ema_slow = pd.Series(pd.ewma(df["close"], span=n_slow, min_periods=n_slow - 1))
    macd = pd.Series(ema_fast - ema_slow, name="MACD_" + str(n_fast) + "_" + str(n_slow))
    macd_sign = pd.Series(
        pd.ewma(macd, span=9, min_periods=8), name="MACDsign_" + str(n_fast) + "_" + str(n_slow)
    )
    macd_diff = pd.Series(macd - macd_sign, name="MACDdiff_" + str(n_fast) + "_" + str(n_slow))
    df = df.join(macd)
    df = df.join(macd_sign)
    df = df.join(macd_diff)
    return df


# Mass Index
def mass_i(df):
    range = df["high"] - df["low"]
    ex1 = pd.ewma(range, span=9, min_periods=8)
    ex2 = pd.ewma(ex1, span=9, min_periods=8)
    mass = ex1 / ex2
    mass_i = pd.Series(pd.rolling_sum(mass, 25), name="Mass Index")
    df = df.join(mass_i)
    return df


# vortex Indicator: http://www.vortexindicator.com/VFX_VORTEX.PDF
def vortex(df, n):
    i = 0
    tr = [0]
    while i < df.index[-1]:
        range = max(df.get_value(i + 1, "high"), df.get_value(i, "close")) - min(
            df.get_value(i + 1, "low"), df.get_value(i, "close")
        )
        tr.append(range)
        i = i + 1
    i = 0
    vm = [0]
    while i < df.index[-1]:
        range = abs(df.get_value(i + 1, "high") - df.get_value(i, "low")) - abs(
            df.get_value(i + 1, "low") - df.get_value(i, "high")
        )
        vm.append(range)
        i = i + 1
    vi = pd.Series(
        pd.rolling_sum(pd.Series(vm), n) / pd.rolling_sum(pd.Series(tr), n), name="Vortex_" + str(n)
    )
    df = df.join(vi)
    return df


# kst Oscillator
def kst(df, r1, r2, r3, r4, n1, n2, n3, n4):
    m = df["close"].diff(r1 - 1)
    n = df["close"].shift(r1 - 1)
    roc1 = m / n
    m = df["close"].diff(r2 - 1)
    n = df["close"].shift(r2 - 1)
    roc2 = m / n
    m = df["close"].diff(r3 - 1)
    n = df["close"].shift(r3 - 1)
    roc3 = m / n
    m = df["close"].diff(r4 - 1)
    n = df["close"].shift(r4 - 1)
    roc4 = m / n
    kst = pd.Series(
        pd.rolling_sum(roc1, n1)
        + pd.rolling_sum(roc2, n2) * 2
        + pd.rolling_sum(roc3, n3) * 3
        + pd.rolling_sum(roc4, n4) * 4,
        name="KST_"
        + str(r1)
        + "_"
        + str(r2)
        + "_"
        + str(r3)
        + "_"
        + str(r4)
        + "_"
        + str(n1)
        + "_"
        + str(n2)
        + "_"
        + str(n3)
        + "_"
        + str(n4),
    )
    df = df.join(kst)
    return df


# Relative Strength Index
def rsi(df, n=14):
    # If the rsi rises above 30, buy signal, rsi falls under 70, a sell signal occurs.
    i = 0
    up_i = [0]
    do_i = [0]
    while i + 1 <= df.index[-1]:
        up_move = df.get_value(i + 1, "high") - df.get_value(i, "high")
        do_move = df.get_value(i, "low") - df.get_value(i + 1, "low")
        if up_move > do_move and up_move > 0:
            up_d = up_move
        else:
            up_d = 0
        up_i.append(up_d)
        if do_move > up_move and do_move > 0:
            do_d = do_move
        else:
            do_d = 0
        do_i.append(do_d)
        i = i + 1
    up_i = pd.Series(up_i)
    do_i = pd.Series(do_i)
    pos_di = pd.Series(pd.ewma(up_i, span=n, min_periods=n - 1))
    neg_di = pd.Series(pd.ewma(do_i, span=n, min_periods=n - 1))
    rsi = pd.Series(pos_di / (pos_di + neg_di), name="RSI_" + str(n))
    df = df.join(rsi)
    return df


# Relative Momentum Index
def rmi(df, n=14, m=10):
    # http://www.csidata.com/?page_id=797  , FinVIZ rmi 10
    i = m
    up_i = list(np.zeros(m))  # Switch by m values
    do_i = list(np.zeros(m))

    while i <= df.index[-1]:
        up_move = df.get_value(i, "high") - df.get_value(i - m, "high")
        do_move = df.get_value(i - m, "low") - df.get_value(i, "low")
        if up_move > do_move and up_move > 0:
            up_d = up_move
        else:
            up_d = 0
        up_i.append(up_d)
        if do_move > up_move and do_move > 0:
            do_d = do_move
        else:
            do_d = 0
        do_i.append(do_d)
        i = i + 1
    up_i = pd.Series(up_i)
    do_i = pd.Series(do_i)

    pos_di = pd.Series(pd.ewma(up_i, span=n, min_periods=n - 1))
    neg_di = pd.Series(pd.ewma(do_i, span=n, min_periods=n - 1))

    # noinspection PyTypeChecker
    rsi = pd.Series(100 * pos_di / (pos_di + neg_di), name="RMI_" + str(n) + "_" + str(m))
    df = df.join(rsi)
    return df


# True Strength Index
def tsi(df, r, s):
    m = pd.Series(df["close"].diff(1))
    a_m = abs(m)
    ema1 = pd.Series(pd.ewma(m, span=r, min_periods=r - 1))
    a_ema1 = pd.Series(pd.ewma(a_m, span=r, min_periods=r - 1))
    ema2 = pd.Series(pd.ewma(ema1, span=s, min_periods=s - 1))
    a_ema2 = pd.Series(pd.ewma(a_ema1, span=s, min_periods=s - 1))
    tsi = pd.Series(ema2 / a_ema2, name="TSI_" + str(r) + "_" + str(s))
    df = df.join(tsi)
    return df


# Accumulation/Distribution
def accdist(df, n):
    ad = (2 * df["close"] - df["high"] - df["low"]) / (df["high"] - df["low"]) * df["volume"]
    m = ad.diff(n - 1)
    n = ad.shift(n - 1)
    roc = m / n
    ad = pd.Series(roc, name="Acc/Dist_ROC_" + str(n))
    df = df.join(ad)
    return df


# chaikin Oscillator
def chaikin(df):
    ad = (2 * df["close"] - df["high"] - df["low"]) / (df["high"] - df["low"]) * df["volume"]
    chaikin = pd.Series(
        pd.ewma(ad, span=3, min_periods=2) - pd.ewma(ad, span=10, min_periods=9), name="chaikin"
    )
    df = df.join(chaikin)
    return df


# Money Flow Index and Ratio
# noinspection PyTypeChecker
def mfi(df, n):
    pp = (df["high"] + df["low"] + df["close"]) / 3
    i = 0
    pos_mf = [0]
    while i < df.index[-1]:
        if pp[i + 1] > pp[i]:
            pos_mf.append(pp[i + 1] * df.get_value(i + 1, "volume"))
        else:
            pos_mf.append(0)
        i = i + 1
    pos_mf = pd.Series(pos_mf)
    tot_mf = pp * df["volume"]
    mfr = pd.Series(pos_mf / tot_mf)
    mfi = pd.Series(pd.rolling_mean(mfr, n), name="MFI_" + str(n))
    df = df.join(mfi)
    return df


# On-balance Volume
def obv(df, n):
    i = 0
    obv = [0]
    while i < df.index[-1]:
        if df.get_value(i + 1, "close") - df.get_value(i, "close") > 0:
            obv.append(df.get_value(i + 1, "volume"))
        if df.get_value(i + 1, "close") - df.get_value(i, "close") == 0:
            obv.append(0)
        if df.get_value(i + 1, "close") - df.get_value(i, "close") < 0:
            obv.append(-df.get_value(i + 1, "volume"))
        i = i + 1
    obv = pd.Series(obv)
    obv_ma = pd.Series(pd.rolling_mean(obv, n), name="OBV_" + str(n))
    df = df.join(obv_ma)
    return df


# Force Index
def force(df, n):
    f = pd.Series(df["close"].diff(n) * df["volume"].diff(n), name="Force_" + str(n))
    df = df.join(f)
    return df


# Ease of Movement
def eom(df, n):
    eo_m = (df["high"].diff(1) + df["low"].diff(1)) * (df["high"] - df["low"]) / (2 * df["volume"])
    eom_ma = pd.Series(pd.rolling_mean(eo_m, n), name="EoM_" + str(n))
    df = df.join(eom_ma)
    return df


# Commodity Channel Index
def cci(df, n):
    pp = (df["high"] + df["low"] + df["close"]) / 3
    cci = pd.Series((pp - pd.rolling_mean(pp, n)) / pd.rolling_std(pp, n), name="CCI_" + str(n))
    df = df.join(cci)
    return df


# Coppock Curve
def copp(df, n):
    m = df["close"].diff(int(n * 11 / 10) - 1)
    n = df["close"].shift(int(n * 11 / 10) - 1)
    roc1 = m / n
    m = df["close"].diff(int(n * 14 / 10) - 1)
    n = df["close"].shift(int(n * 14 / 10) - 1)
    roc2 = m / n
    copp = pd.Series(pd.ewma(roc1 + roc2, span=n, min_periods=n), name="Copp_" + str(n))
    df = df.join(copp)
    return df


# Keltner Channel
def kelch(df, n):
    kel_ch_m = pd.Series(
        pd.rolling_mean((df["high"] + df["low"] + df["close"]) / 3, n), name="KelChM_" + str(n)
    )
    kel_ch_u = pd.Series(
        pd.rolling_mean((4 * df["high"] - 2 * df["low"] + df["close"]) / 3, n),
        name="KelChU_" + str(n),
    )
    kel_ch_d = pd.Series(
        pd.rolling_mean((-2 * df["high"] + 4 * df["low"] + df["close"]) / 3, n),
        name="KelChD_" + str(n),
    )
    df = df.join(kel_ch_m)
    df = df.join(kel_ch_u)
    df = df.join(kel_ch_d)
    return df


# Ultimate Oscillator
def ultosc(df):
    i = 0
    tr_l = [0]
    bp_l = [0]
    while i < df.index[-1]:
        tr = max(df.get_value(i + 1, "high"), df.get_value(i, "close")) - min(
            df.get_value(i + 1, "low"), df.get_value(i, "close")
        )
        tr_l.append(tr)
        bp = df.get_value(i + 1, "close") - min(
            df.get_value(i + 1, "low"), df.get_value(i, "close")
        )
        bp_l.append(bp)
        i = i + 1
    ult_o = pd.Series(
        (4 * pd.rolling_sum(pd.Series(bp_l), 7) / pd.rolling_sum(pd.Series(tr_l), 7))
        + (2 * pd.rolling_sum(pd.Series(bp_l), 14) / pd.rolling_sum(pd.Series(tr_l), 14))
        + (pd.rolling_sum(pd.Series(bp_l), 28) / pd.rolling_sum(pd.Series(tr_l), 28)),
        name="Ultimate_Osc",
    )
    df = df.join(ult_o)
    return df


# Donchian Channel
def donch(df, n):
    i = 0
    dc_l = []
    while i < n - 1:
        dc_l.append(0)
        i = i + 1
    i = 0
    while i + n - 1 < df.index[-1]:
        dc = max(df["high"].ix[i : i + n - 1]) - min(df["low"].ix[i : i + n - 1])
        dc_l.append(dc)
        i = i + 1
    don_ch = pd.Series(dc_l, name="Donchian_" + str(n))
    don_ch = don_ch.shift(n - 1)
    df = df.join(don_ch)
    return df


# Standard Deviation
def stddev(df, n):
    df = df.join(pd.Series(pd.rolling_std(df["close"], n), name="STD_" + str(n)))
    return df


def rwi(df, nn, natr):
    return 0


# First we compute rwi for maxima:
# RWImax = [(day‘s high - (day's low * number of days))] / [(Average True Range * number of days * square root of number of days)]
#
# Similarly, we compute the rwi for minima:
# RWImin = [(day's high *number of days - (day's low))] / [(Average True Range * number of days * square root of number of days)]
#
# True range = higher of (day's high, previous close ) - lower of (day's low, previous close)


def nbday_low(df, n):
    """nb of days from 1 year low """
    close = df["close"].values
    ndaylow = np.zeros(len(close))
    distlow = np.zeros(len(close))
    for i in range(n, len(close)):
        kid, min1 = np_find_minpos(close[i - n : i])
        ndaylow[i] = n - kid
        distlow[i] = close[i] - min1

    smin1 = pd.Series(ndaylow, name="ndaylow_" + str(n))
    smin2 = pd.Series(distlow, name="ndistlow_" + str(n))
    df = df.join(smin1)
    df = df.join(smin2)
    return df


def nbday_high(df, n):
    """nb of days from 1 year low """
    close = df["close"].values
    ndaylow = np.zeros(len(close))
    distlow = np.zeros(len(close))
    for i in range(n, len(close)):
        kid, min1 = np_find_maxpos(close[i - n : i])
        ndaylow[i] = n - kid
        distlow[i] = close[i] - min1

    smin1 = pd.Series(ndaylow, name="ndayhigh_" + str(n))
    smin2 = pd.Series(distlow, name="ndisthigh_" + str(n))
    df = df.join(smin1)
    df = df.join(smin2)
    return df


"""
26 responses
 171bfddace1cb03c836e2f6054f1b9c8  Lionel  Mar 17, 2015
This is great , thank you for sharing this :)

Lionel.

 A31bd5accd3432cbf5a8e3dc84a04928  Ambooj Mittal  Mar 22, 2015
this seems great

 B95a5c9a32a4c951f474cc75181e5297  Andrea D'Amore  Mar 25, 2015
There's little need to see an indicator implementation, the whole point of using external libraries is delegating implementation details to someone else, and ta-lib is well tested in that regard. Moreover by using a python implementation you're possibly not using acceleration on numpy or panda's side.

It is IMHO better to understand and indicator's rationale and proper usage by reading the author's note or some specific article about it than looking at the code.

Also those are just methods from the pandas implemented module of pyTaLib with packages import rather than wildcards, I'd put the author's name back in the body (due to copyright) and remove your name from it since the file is essentially unmodified.

 B7dd86784deedc047d5ad0fb30378bdd  Peter Bakker  Mar 25, 2015
Your opinion granted. For me it actually helps to look at what happens and as I don't think I'm unique so that's why I put the code up.

The author Bruno is mentioned so I don't think that's an issue

The code is changed here and there and I added a few functions but I omitted to record what I added. As in my humble opinion it's useful I'll keep the post live.

 B7dd86784deedc047d5ad0fb30378bdd  Peter Bakker  Mar 25, 2015
In addition to that. Above code has quite a few indicators that talib does not have so only for that reason it's useful to have it around.

 D4b55852454c34c05c0d55b91c8575b1  Robby F  Mar 25, 2015
This is sick. I have spent way too much time trying to python talib functions for custom oscillators. Thanks.

edit: What is the good/standard way to build a dataframe with high/low/open/close/volume? I typically have/use individual history dataframes for each but I would like to know how to use the code as it is in the original post.

Thanks again

 409a3be6af196fecbaf1329aef6b50c2  Tarik Fehmi  Apr 24, 2015
Thank you!

 0c2b51aac4540dbc179f7583f7a231ea  Ethan Adair  Apr 24, 2015
Robby, here is some code from one of my algos I use to build an OHLC dataframe.. It works but I don't like the method. If anyone has a more streamlined approach I would be grateful to see it.
EDIT: you may want to modify it, I made it in such a way that it appends securities in the same data columns, this was useful for the way I was using the securities data

#Define Window  
    trail = 200  
    #store OHLCV data  
    open_hist = history(trail,'1d','open_price',ffill=False)  
    close_hist = history(trail,'1d','close_price',ffill=False)  
    high_hist = history(trail,'1d','high',ffill=False)  
    low_hist = history(trail,'1d','low',ffill=False)  
    volume_hist = history(trail,'1d','volume',ffill=False)  
    opencol = []  
    closecol = []  
    highcol = []  
    lowcol = []  
    volumecol = []  
    #trinsmit OHLCV to continuous numpy arrays  
    for sec in context.secs:  
        opencol = np.concatenate((opencol, open_hist[sec][:]))  
        closecol = np.concatenate((closecol, close_hist[sec][:]))  
        highcol = np.concatenate((highcol, high_hist[sec][:]))  
        lowcol = np.concatenate((lowcol, low_hist[sec][:]))  
        volumecol = np.concatenate((volumecol, volume_hist[sec][:]))  
    print("putting in pandas")  
    #Combine arrays into dataframe  
    df = pd.DataFrame({'O': opencol,  
                      'H': highcol,  
                      'L': lowcol,  
                      'C': closecol,  
                      'V': volumecol})  






"""
