# -*- coding: utf-8 -*-
"""
Project : Dataset and Code Manager into centralized framework.
"""
import blaze as bz, pandas as pd, numpy as np, copy
import re, sys, os, time, datetime, csv, sqlalchemy as sql
from datetime import datetime
import util

class project(object):
    ''' Structure
       data --->  In memory / On Disk with in memory mapping / on Disk only file
       DB <----> SQLAlchemy <----> Pandas   <---->  Python_memory
       Methods:
          db_  : SQL DB related
          project_ :  related to project in pandas + connection
          table_   :  related to table in pandas + connection
          code_    :  related to code source serialization
             sessiondata: for in memory variables
             sessioncode : for in memory code source / import package /....
             
      Complex Class serialization :
         https://news.ycombinator.com/item?id=10788814

       JSON Schema vs SQL Alchemy
       https://marshmallow-sqlalchemy.readthedocs.io/en/latest/index.html

       http://stackoverflow.com/questions/25613543/how-to-use-dill-to-serialize-a-class-definition

       Pickle would blow up on the above.
       If you don't want dill to serialize the class explicitly,
       and to do what pickle does, then you can ask dill to pickle by
       reference with dill.dumps(Foo, byref=True).

    '''

    def __init__(self, name='', folder='', user_id='', version=-1, db_name = 'sqlite:///aaserialize/store/project_reftable.sqlite', path='D:/_devs/Python01/project27/aaserialize/session/', verbose=0 ):
        self.name, self.folder, self.user_id, self.version = name, folder, user_id, version
        self.db_name = db_name   # db=r'sqlite:///aaserialize/store/yahoo.db'
        self.db_con = sql.create_engine(self.db_name, execution_options={'sqlite_raw_colnames':True})
        self.path= os.getcwd() if path=='' else path
        self.verbose= verbose

        self.project_initialize(name, folder, user_id, version)
        self.ptable = self._db_table_get()
        self.mcol_table, self.mcol_project=   self.ptable.shape[1], self.pproject.shape[1]

        if self.project_id==-1 : self.table_add({'name':'sessiondata_main',  'datatype':'spydata',  'summary':'Spydata Main session file'})
          
 
        
    # DB access ------------------------------------------------------------------------------------------------
    def db_sql(self, sqls, type1='pandas', isprint=0):
        df = pd.read_sql_query(sqls, self.db_con)
        if isprint: print(sqls)
        df.reset_index(drop=True, inplace=True)
        if type1 == 'pandas':        return df
        if type1 == 'numpy':         return df.values, df.columns.values

    def _db_criteria_tosql(self, c):
        ''' Generic criteria request to  SQL using AST: Abstract Tree '''
        pass


    def _db_table_get(self, project_id=0):
        # find project in master database, otherwise create one
        project_id= self.project_id if project_id == 0 else project_id
        # print self.project_id, project_id,   project_id != ''
        ss=  'select t.* from  ptable t where  t.project_id=  ' + str(project_id)
        df = self.db_sql(ss, 'pandas')
        df['date_create'] = pd.to_datetime(df['date_create']).astype(datetime)
        df['date_update'] = pd.to_datetime(df['date_update']).astype(datetime)
        return df


    def db_search(self, where1='project/table/session', name='', folder='', user_id='', version=-1, isprint=1 ):
        ss=  'select t.* from  pproject p where  p.name like "'+ name +'" p.folder like "'+ name +'" p.user_id like "' + user_id +'" '
        df = self.ptable_sql(ss, 'pandas')
        if isprint:
           util.pprint(df['id','name', 'folder', 'user_id', 'version', 'summary']);return None
        else :
           return df

        ss=  'select t.* from  ptable p where  p.name like "'+ name +'" p.folder like "'+ name +'" p.user_id like "' + user_id +'" '
        df = self.ptable_sql(ss, 'pandas')
        if isprint:
           util.pprint(df['id','name', 'folder', 'user_id', 'version', 'summary']); return None
        else :
           return df


    # Project -----------------------------------------------------------------------------------------
    def project_initialize(self, name='project123', folder='folder1', user_id='my_name_123', version=-1):
      # Find project in master database, otherwise create one

      #Key= name+folder+user_id
      ss= 'select * from project where name="'+name+'" and folder="'+folder+'" and user_id="'+user_id+ '" order by version desc'
      pproject1 = self.db_sql(ss, 'pandas')
      self.pprojecthisto= pproject1
      self.version_last=  pproject1['version'].values[0]

      if pproject1.shape[0] > 0: # name+folder+user_id  Found 
        if version != -1 :  
           pproject=  copy.copy(pproject1[ pproject1['version']== version])
           if pproject.shape[0] == 0 : 
              print(('Use Version -1. Cannot find version '+str(version))); sys.exit()
        else :   # Use Last version
           pproject= copy.copy(pproject1[:1])

        pproject['date_create']= pd.to_datetime(pproject['date_create']).astype(datetime)
        pproject['date_update']= pd.to_datetime(pproject['date_update']).astype(datetime)
        self.pproject=     pproject
        self.project_id=   pproject['id'].values[0]
        self.version=      pproject['version'].values[0]


      else: #Create a new project
        print(('New Creation of '+name))
        pproject= util.pd_insertrow(pproject1, [''] * len(pproject1.columns))
        pproject['name']=         name
        pproject['folder']=       folder
        pproject['user_id']=      user_id
        pproject['date_update']=  datetime.now()
        pproject['date_create']=  datetime.now()
        pproject['version']= 0
        pproject['user_update']=  os.getenv('username')
        self.pproject= pproject;  self.project_id= -1
        self.version_last= 1



    def search(self, name='project123', folder='folder1', user_id='my_name_123', version=1):
       pass



    def project_save(self):
      ''' save version/commit in db
          If updated from earlier version_5 ---> version_24 (create new_version)
      '''
      # ptable id	project_id version	name	source	uri	uri_source	datatype	sourcetype	group1	group2	group3	size	item1	item2	date_creation	date_update	user_update
      # project id	version	name	folder	user_id	summary	group_user	group_project	group1	group2	group3	date_create	date_update	user_update
       # (id, version, name, folder, user_id, summary, group_user, group_project, group1, group2, group3, date_create, date_update, user_update)   '

      #Save/Insert project :
      self.pproject['date_update']= datetime.now()
      self.pproject['summary']=     self.pproject['summary'] + '\n Updated from version '+ str(self.version) + ';'
      self.version_last=            1+self.version_last
      self.version=                 self.version_last
      self.pproject['version']=     self.version_last
      self.pproject['user_update']= os.getenv('username')

      res= util.sql_insert_df(self.pproject, 'project', self.db_con, ['id']  )
      self.project_id= res.lastrowid
      self.pproject.ix[-1:,'id']= self.project_id
      self.pprojecthisto= pd.concat((self.pproject,self.pprojecthisto), axis=0, ignore_index=True)

      # self.project_initialize(name='project123', folder='folder1', user_id='my_name_123', version= self.version)

      #Save/Insert Table
      self.ptable.ix[:,'project_id']=  self.project_id
      self.ptable.ix[:,'date_update']= datetime.now()

      res= util.sql_insert_df(self.ptable, 'ptable', self.db_con, ['id']  )
      # self.ptable['id']= res.lastrowid
      self.ptable = self._db_table_get()     #Get table back


    def project_export_meta(self, type1='tuple/dict/list/full'  ,location=''):
      ''' Export Project meta-data in Pandas/CSV : ptable, pproject, phisto'''

      if type1== 'list' :   return  [self.ptable, self.pprojecthisto]
      if type1== 'tuple' :  return    self.ptable, self.pprojecthisto
      if type1== 'dict' :
         d={}
         d['ptable']= self.project_id;       d['phisto']= self.project_id
         return d


    def project_archive(self,name='project123', folder='folder1', user_id='my_name_123', version=-1):
       ''' Archive a series of projec into project_archive / project_table / Cannot delete Project  '''
       pass

    def project_update(self, name=None, folder=None, user_id=None, summary=None, groupuser=None, grouproject=None, group1=None, group2=None ):
      if name is not None :      self.pproject['name']= name
      if folder is not None :    self.pproject['folder']= folder
      if user_id is not None :   self.pproject['user_id']= user_id
      if summary is not None :   self.pproject['summary']=  self.pproject['summary'] + '\n' + summary
      if groupuser is not None :    self.pproject['group_user']= groupuser
      if grouproject is not None :   self.pproject['group_project']= grouproject


    #Table ----------------------------------------------------------------------------------------------
    def table(self, table_name1) :
      ''' Get table details into a dataframe '''
      return self.ptable[self.ptable.name== table_name1]


    def t(self, table_name1) :
      ''' Get table details into a dataframe '''
      return self.ptable[self.ptable.name== table_name1]


    def table_add(self, vec=['table_name', 'uri', 'summary']):
      # dict1 = {'uri_table': 'table_name': '', 'python_name':}
      name = vec[0] if  type(vec) == list else vec['name']

      if util.find(name, self.ptable['name'].values) != -1 :
        print('Table name already exist, please choose new one')
        return None

      self.ptable= util.pd_insertrow(self.ptable, [''] * self.mcol_table, None, isreset=1)
      #return self.ptable
      
      kid= self.ptable.index.values[-1]
      if kid == 0 :
         self.ptable.loc[kid,'id']=            0
         self.ptable.loc[kid,'project_id']=   -1   
      else :
         self.ptable.loc[kid,'id']=       1 + self.ptable.loc[kid-1,'id']
         self.ptable.loc[kid,'project_id']=   self.ptable.loc[kid-1,'project_id']

      self.ptable.loc[kid,'version']=      0
      self.ptable.loc[kid,'date_create']= datetime.now()
      self.ptable.loc[kid,'date_update']= datetime.now() 
      self.ptable.loc[kid,'user_update']= os.getenv('username')

      if  type(vec) == list :        
        self.ptable.loc[kid,'name']=    vec[0]
        self.ptable.loc[kid,'uri']=     vec[1]
        self.ptable.loc[kid,'summary']= vec[2]
                
      if  type(vec) == dict :
        for key, x in list(vec.items()) :   self.ptable.loc[kid, key] = x
        self.ptable.loc[kid,'date_create']= datetime.now()


    def table_update(self, table_name='', vdict={'item_name': 'table_name'}):
        ''' {'uri':  'new_one', 'dshape': 'newone'}) '''
        kid=  util.find(table_name, self.ptable['name'].values) 
        #Implicif  Index: 0...5
        
        if kid==-1 :
          print('Table name does not exist');   return None          
        
        type1= type(vdict)
        self.ptable.loc[kid,'date_update']= datetime.now()
        self.ptable.loc[kid,'version']= 1 +  int(self.ptable.loc[kid,'version'])
        self.ptable.loc[kid,'user_update']=  os.getenv('username')

        if type1 == list :
          self.ptable.loc[kid,'uri']=    vdict[0]

        if type1 == dict :
          for key, x in list(vdict.items()) :   self.ptable.loc[kid, key] = x


    def table_delete(self, table_name=''):
        ''' Delete table '''
        kid=  util.find(table_name, self.ptable['name'].values)
        if kid==-1 : print('Table name does not exist')
        else :
           self.ptable.drop(self.ptable.index[kid], inplace=True)
           print('Table deleted')


    def table_load(self, tablename, exportype1='blaze' , dshape1=None, python_name=''):  # Load get the table reference
        ''' Export table URI into memory variable
           df = bz.data('sqlite:///%s::iris' % bz.utils.example('iris.db'))
           df = bz.data('my-small-file.csv')
           df = bz.data('myfile-2014-01-*.csv.gz')
           df = bz.data('myfile.hdf5::/mydataset')
           df = bz.data('impala://54.81.251.205/default::reddit_parquet')
           df = bz.data(sql.create_engine('postgresql://myusername:mypassword@localhost:5432/mydatabase'))
           odo('hdfstore://path_store_1::table_name', 'hdfstore://path_store_new_name::table_name')

        '''
        kid = util.find(tablename, self.ptable['name'].values)
        table_uri, source_uri = self.ptable.loc[kid, 'uri'], self.ptable.loc[kid, 'uri_source']

        #In memory Identifier -----------------------------------------------------------------------
        df= bz.data(data_source=table_uri, dshape=dshape1)
        if exportype1== 'blaze' : pass

        if exportype1=='pandas' : df= bz.odo(df, pd.DataFrame)

        if exportype1=='hdfs' :  df= bz.odo(df, pd.HDFStore)

        if exportype1=='numpy' : df= bz.odo(df, np.ndarray)

        if python_name == '' :        return df
        else :
             main1= sys.modules['__main__']
             setattr(main1, python_name, df)



    def table_loadall(tablename, type1='blaze' ,python_name='table'):  # Load get the table reference
        if isall:
            main1 = sys.modules['__main__']
            for i in self.table.shape[0]:
                var1 = self.table_get(uri=self.ptable.ic[i, 'table_uri'])
                table_list.append((self.ptable[i, 'python_name'], var1))
        else:
            table_list = [(python_name, table_name)].append((self.ptable[i, 'python_name'], var1))

        for python_n, var1 in table_list:
            setattr(main1, pythno_n, var1.copy())


    def table_sessiondata_load(self,  name='sessiondata_main', file1=''):
        ''' Session SPYDATA from load, '''
        if file1 ==  '' :
           kid= util.find(name, self.ptable['name'].values)
           file1= self.ptable.loc[kid, 'uri']
        util.session_load(file1)


    def table_sessiondata_save(self, global1=None, file1='',  sessioname='sessiondata_main', isnewversion=False):
       # action_Item_Sub_definition, into single file
        if file1=='' : 
          file1=   self.path+'/sessiondata_' + self.pproject['name'].values[-1] 
          file1+=  '_'+ self.pproject['folder'].values[-1] + '_' + self.pproject['user_id'].values[-1]
          ver1= self.version_last if isnewversion else  self.version
          file1+=  '_'+ str(ver1)

        file1= file1+'.spydata' if not util.os_file_exist(file1)  else file1 + '_' + str(np.random.randint(1000, 9999)) + '.spydata'
        util.session_save(file1, global1, use_cwd_path=0)
        self.table_update(sessioname, [file1])


    def table_printall(self):  # print all the table in memory
        pass


    def table_uri_check(self, ptable=None):
       ''' Check all the link Exist / broken'''
       pass


    def zhelp(self):
     print('''
       ds=  dshape1= dshape("var * {name: string[20, 'ascii'], amount: float64}"
       df = bz.data('sqlite:///%s::iris' % bz.utils.example('iris.db'))
       df = bz.data('my-small-file.csv')
       df = bz.data('myfile-2014-01-*.csv.gz')
       engine = sql.create_engine('postgresql://%s:%s@localhost:5432/%s' %(myusername, mypassword, mydatabase))
       df     = bz.data(engine)
Paths to files on disk
.csv
.json
:.csv.gz/json.gz
.hdf5
.hdf5::/datapath
.bcolz
.xls(x)
.sas7bdat

Collections of files on disk
*.csv

SQLAlchemy strings
sqlite:////absolute/path/to/myfile.db::tablename
sqlite:////absolute/path/to/myfile.db (specify a particular table)
postgresql://username:password@hostname:port
impala://hostname (uses impyla)

MongoDB Connection strings
mongodb://username:password@hostname:port/database_name::collection_name

Remote locations via SSH, HDFS and Amazon’s S3
ssh://user@hostname:/path/to/data
hdfs://user@hostname:/path/to/data
s3://path/to/data

     ''')



    '''
    #Code ----------------------------------------------------------------------------------------------
    def code(self, code_name1):
         pass


    def code_add(self, vec=['table_name', 'uri', 'summary']):
      # dict1 = {'uri_table': 'table_name': '', 'python_name':}

      if util.find(vec[0], self.ptable['name'].values) != -1 :
        print('Table name already exist, please choose new one')
        return None

      mcol= self.ptable.shape[1]
      self.ptable= util.pd_insertrow(self.ptable, ['']*mcol, None,   isreset=1 )
      #return self.ptable

      kid= self.ptable.index.values[-1]
      if kid == 0 :
         self.ptable.loc[kid,'id']=            0
         self.ptable.loc[kid,'project_id']=   -1
      else :
         self.ptable.loc[kid,'id']=       1 + self.ptable.loc[kid-1,'id']
         self.ptable.loc[kid,'project_id']=   self.ptable.loc[kid-1,'project_id']

      self.ptable.loc[kid,'version']=      0
      self.ptable.loc[kid,'date_create']= datetime.now()
      self.ptable.loc[kid,'date_update']= datetime.now()

      if  type(vec) == list :
        self.ptable.loc[kid,'name']=    vec[0]
        self.ptable.loc[kid,'uri']=     vec[1]
        self.ptable.loc[kid,'summary']= vec[2]

      if  type(vec) == dict :
        for key, x in vec.items() :   self.ptable.loc[kid, key] = x
        self.ptable.loc[kid,'date_create']= datetime.now()


    def code_update(self, table_name='', vdict={'item_name': 'table_name'}):
        # {'uri':  'new_one', 'dshape': 'newone'})
        kid=  util.find(table_name, self.ptable['name'].values)
        #Implicif  Index: 0...5

        if kid==-1 :
          print('Table name does not exist');   return None

        type1= type(vdict)
        self.ptable.loc[kid,'date_update']= datetime.now()
        self.ptable.loc[kid,'version']= 1 +  int(self.ptable.loc[kid,'version'])
        self.pproject.loc[kid,'user_update']=  os.getenv('username')


        if type1 == list :
          self.ptable.loc[kid,'uri']=    vdict[0]

        if type1 == dict :
          for key, x in vdict.items() :   self.ptable.loc[kid, key] = x


    def code_delete(self, table_name=''):
        # {'uri':  'new_one', 'dshape': 'newone'})
        kid=  util.find(table_name, self.ptable['name'].values)
        #Implicif  Index: 0...5

        if kid==-1 : print('Table name does not exist');   return None

        type1= type(vdict)
        self.ptable.loc[kid,'date_update']= datetime.now()
        self.ptable.loc[kid,'version']= 1 +  int(self.ptable.loc[kid,'version'])
        self.pproject.loc[kid,'user_update']=  os.getenv('username')

        self.ptable.loc[kid,'uri']=    vdict[0]
        for key, x in vdict.items() :
           self.ptable.loc[kid, key] = x


    def code_load(name='mytable5', table_uri='',  python_name='table', inplace=0):  # Load get the table reference
        # Export table URI into memory variable
        # Identifier of the table
        if table_uri != '':
            kid = util.find(name, self.ptable['name'].values)
            table_uri, source_uri = self.ptable.ic[kid, 'table_uri'], self.ptable.ic[kid, 'source_uri']

        # Blaze identifier
        df = bz.data(data_source=table_uri, dshape=None)

        if isall:
            main1 = sys.modules['__main__']
            for i in self.table.shape[0]:
                var1 = self.table_get(uri=self.ptable.ic[i, 'table_uri'])
                table_list.append((self.ptable[i, 'python_name'], var1))
        else:
            table_list = [(python_name, table_name)].append((self.ptable[i, 'python_name'], var1))

        for python_n, var1 in table_list:
            setattr(main1, pythno_n, var1.copy())




        return df


    def code_sessiondata_load(self, name='session', version=-1):
        # Session SPYDATA from load, 4*|mlhgfdsq654
        #v SQ'(-[_]@

        if file1 ==  '' :
           kid= util.find('spydata', self.ptable['datatype'].values)
           file1= self.ptable.loc[kid, 'uri']
        util.session_load(file1)


    def code_sessiondata_save(self, global1=None, file1=''):
       # action_Item_Sub_definition, into single file
        if file1=='' : file1= '/session_data_'+str(np.random.randint(99999))+'.spydata'
        util.session_save(file1, global1)
        self.table_update('session', [self.session_path+file1])
    '''





############################################################################
#---------------------             --------------------







############################################################################











############################################################################
#---------------------             --------------------




############################################################################





















############################################################################
#---------------------             --------------------






























