"""
    Input Historical prices to SQLite Database.
"""
import csv
import datetime
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import sqlalchemy as sql

import portfolio as pf
import util


class dailyDataStore(object):
    def __init__(self, db_full_path=r"sqlite:///aaserialize/store/yahoo.db", source="yahoo"):
        self.dbname = db_full_path
        self.con = sql.create_engine(self.dbname)
        self.data_tablename = util.sql_get_dbschema(dbengine=self.con, isprint=0)
        self.source = source  # origin of data
        """
          daily_us_etf
          daily_us_stock
          daily_us_stock2
          daily_us_index
          daily_us_fund
          daily_us_fundamental
        """

        ## output data

    def download_todb(self, sym_list, table1="daily", start1="20140120", end1="20160805"):
        """         """
        for i, sub_list in enumerate(self._break_list_to_sub_list(sym_list)):
            # print 'processing sub list'  , sub_list
            if self.source == "yahoo":
                qlist, sym = pf.imp_yahoo_getquotes(sub_list, start=start1, end=end1)
            if self.source == "google":
                qlist, sym = pf.imp_googleQuoteList(sub_list, start=start1, end=end1)

            print(("Batch " + str(i), len(sub_list), qlist[0].date.values[-1]))
            # print(len(sym),  sym[0])

            for k, df in enumerate(qlist):
                df = util.pd_insertcol(df, "sym", [sym[k]] * df.shape[0])
                # df['date'] = df['date'].map(lambda x: int(x.replace('-', '')))  # in int value
                df.to_sql(
                    table1,
                    self.con,
                    if_exists="append",
                    index=False,
                    index_label=None,
                    chunksize=None,
                )
            del qlist, sym

    def get_tablelist(self, isprint=1):
        listtable = util.sql_get_dbschema(dbengine=self.con, isprint=isprint)
        return listtable

    def get_symlist(self, table1):
        """ Retrieve the stocklist from db """
        sql_str = "SELECT DISTINCT sym FROM %s order by sym asc " % table1
        df = pd.read_sql_query(sql_str, self.con)
        return np.array([str(x[0]) for x in df.values])

    def get_sql(
        self,
        sqls='SELECT * FROM daily_us_stock where sym in ("SPY") order by date  asc',
        stock_list=[],
        table1="daily_us_etf",
        start1="",
        end1="",
        split_df=1,
    ):
        """ Retrieved a list of stocks covering the target date range select_all (bool):  Will pull all the stock symbol"""
        sym_str = "".join(['"' + n + '",' for n in stock_list])
        sym_str = sym_str[:-1]

        df = pd.read_sql_query(sql_str, self.con)
        df.reset_index(drop=True, inplace=True)
        df["date"] = df["date"].map(lambda x: int(x.replace("-", "")))  # in int value
        if start1 != "" and end1 != "":
            df = df[
                (df["date"].values >= int(start1)) & (df["date"].values <= int(end1))
            ]  # Date Interval creation

        if split_df:
            qlist, sym = self._splitdf_inlist(df)
            return qlist, sym
        else:
            return df, df.sym.uniques

    def get_histo(self, stock_list=[], table1="daily_us_etf", start1="", end1="", split_df=1):
        """ Retrieved a list of stocks covering the target date range select_all (bool):  Will pull all the stock symbol"""
        sym_str = "".join(['"' + n + '",' for n in stock_list])
        sym_str = sym_str[:-1]

        if len(stock_list) > 0:
            sql_str = "SELECT * FROM %s where sym in (%s) order by date  asc  " % (table1, sym_str)
        else:
            sql_str = "SELECT * FROM %s order by date asc  " % (table1)

        df = pd.read_sql_query(sql_str, self.con)
        df.reset_index(drop=True, inplace=True)  # Reset to 0...565

        df["date"] = df["date"].map(lambda x: int(x.replace("-", "")))  # in int value
        if start1 != "" and end1 != "":
            df = df[
                (df["date"].values >= int(start1)) & (df["date"].values <= int(end1))
            ]  # Date Interval creation

        if split_df:
            qlist, sym = self._splitdf_inlist(df)
            return qlist, sym
        else:
            return df

    def update_table(self, table1, colname):
        df = dstore.get_histo([], table1=table1, start1="", end1="", split_df=0)
        df[colname] = df[colname].map(lambda x: int(x.replace("-", "")))
        util.sql_delete_table(table1, self.dbname)

        # Batch upload for memory
        nchunk = 500000
        nn = len(df.index)
        for j in range(0, int(nn / nchunk) + 1):
            i = nchunk * j
            df2 = df[i : i + nchunk]
            df2.to_sql(
                table1, self.con, if_exists="append", index=False, index_label=None, chunksize=None
            )

    def clean_table(self, table1):
        print("Remove Duplicate")
        sql = (
            "DELETE FROM "
            + table1
            + " WHERE rowid NOT  IN(SELECT  min(rowid) FROM "
            + table1
            + " GROUP BY sym, date )"
        )
        with self.con.connect() as con:
            rs = con.execute(sql)

    def _splitdf_inlist(self, df):
        UniqueNames = df.sym.unique()
        l1 = []  # create unique list of names
        for key in UniqueNames:
            df2 = df[:][df.sym == key]
            df.reset_index(drop=True, inplace=True)
            l1.append(df2)
        return l1, UniqueNames

    def _break_list_to_sub_list(self, full_list, chunk_size=200):
        if chunk_size < 1:
            chunk_size = 1
        return [full_list[i : i + chunk_size] for i in range(0, len(full_list), chunk_size)]


class intradayDataStore(object):
    def __init__(
        self, db_full_path="D:/_devs/Python01/project27/aaserialize/store/yahoo.db", source="yahoo"
    ):
        self.dbname = "sqlite://" + db_full_path
        self.con = sql.create_engine(self.dbname)
        self.data_tablename = util.sql_get_dbschema(dbengine=self.con)
        self.source = source  # origin of data
        """
          daily_us_risk1   :      Vol, Correlation, state calculation
          daily_us_risk2   :      Technical, Vol, Correlation, state calculation
        """

        ## output data

    def _break_list_to_sub_list(self, full_list, chunk_size=100):
        if chunk_size < 1:
            chunk_size = 1
        return [full_list[i : i + chunk_size] for i in range(0, len(full_list), chunk_size)]

    def download_todb(self, sym_list, table1="daily", start1="20140120", end1="20160805"):
        """         """
        for i, sub_list in enumerate(self._break_list_to_sub_list(sym_list)):
            # print 'processing sub list'  , sub_list
            qlist, sym = pf.imp_yahoo_getquotes(sub_list, start=start1, end=end1)
            print(("Batch " + str(i), len(sub_list), qlist[0].date.values[-1]))
            # print(len(sym),  sym[0])

            for k, df in enumerate(qlist):
                df = util.pd_insertcol(df, "sym", [sym[k]] * df.shape[0])
                # df['date'] = df['date'].map(lambda x: int(x.replace('-', '')))  # in int value
                df.to_sql(
                    table1,
                    self.con,
                    if_exists="append",
                    index=False,
                    index_label=None,
                    chunksize=None,
                )
            del qlist, sym

    def get_tablelist(self):
        listtable = util.sql_get_dbschema(dbengine=self.con)
        return listtable

    def get_symlist(self, table1):
        """ Retrieve the stocklist from db """
        sql_str = "SELECT DISTINCT sym FROM %s order by sym asc " % table1
        df = pd.read_sql_query(sql_str, self.con)
        return np.array([str(x[0]) for x in df.values])

    def get_histo(self, stock_list=[], table1="daily_us_etf", start1="", end1="", split_df=1):
        """ Retrieved a list of stocks covering the target date range select_all (bool):  Will pull all the stock symbol"""
        stock_sym_str = "".join(['"' + n + '",' for n in stock_list])
        stock_sym_str = stock_sym_str[:-1]

        if len(stock_list) > 0:
            sql_str = "SELECT * FROM %s where sym in (%s) order by date  asc  " % (
                table1,
                stock_sym_str,
            )
            df = pd.read_sql_query(sql_str, self.con)
        else:
            sql_str = "SELECT * FROM %s order by date asc  " % (table1)
            df = pd.read_sql_query(sql_str, self.con)

        if start1 != "" and end1 != "":  # Date Interval creation
            # df['date'] = df['date'].map(lambda x: int(x.replace('-', '')))       #in int value
            df = df[(df["date"].values >= int(start1)) & (df["date"].values <= int(end1))]

        if split_df == 1:
            UniqueNames = df.sym.unique()  # create unique list of names
            l1 = []
            for key in UniqueNames:
                l1.append(df[:][df.sym == key])
            return l1, UniqueNames
        else:
            return df

    def update_table(self, table1, colname):
        df = dstore.get_histo([], table1=table1, start1="", end1="", split_df=0)
        df[colname] = df[colname].map(lambda x: int(x.replace("-", "")))
        util.sql_delete_table(table1, self.dbname)

        df.to_sql(
            table1, self.con, if_exists="append", index=False, index_label=None, chunksize=None
        )

    def clean_table(self, table1):
        print("Remove Duplicate")
        sql = (
            "DELETE FROM "
            + table1
            + " WHERE rowid NOT  IN(SELECT  min(rowid) FROM "
            + table1
            + " GROUP BY sym, date )"
        )
        with self.con.connect() as con:
            rs = con.execute(sql)


class riskDataStore(object):
    def __init__(
        self,
        db_full_path="D:/_devs/Python01/project27/aaserialize/store/risk1.db",
        listtable=["daily", "dividend"],
    ):
        """ Set the link to the database that store the information. """
        self.dbname = "sqlite://" + db_full_path
        self.con = lite.create_engine("sqlite://" + db_full_path)

        self.hist_data_tablename = listtable[0]  # differnt table store in database

        ## output data

    def _break_list_to_sub_list(self, full_list, chunk_size=200):
        if chunk_size < 1:
            chunk_size = 1
        return [full_list[i : i + chunk_size] for i in range(0, len(full_list), chunk_size)]
