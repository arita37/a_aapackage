# -*- coding: utf-8 -*-
"""
Quandl Data calculation 
"""
%load_ext autoreload
%autoreload 2

import util
import Quandl as quandl
import copy



'''
util.aa_getmodule_doc('Quandl')

API token dffTSNJ4JHbRE1Csd7zZ
'''

#-----------------------Get Data From Quandl ---------------------------------------------

l1= []

data = quandl.get("EUREKA/481")


import numpy as np
np.

srch1= quandl.search("EUREKAHEDGE", )


ldata=np.empty(500, dtype=np.object)
for i, v in enumerate(srch1) :
  name1= v['code']
  
  if 
  try: 
   data= quandl.get(name1)
   print(name1+'_ok')
   v['data']= copy.deepcopy(data)
   ldata[i]= copy.deepcopy(v)
  except: 
    print(name1+'_notok')
    pass



util.save_obj((srch1,ldata),'QUANDL_EUREKA_'+util.date_now())



x=  util.load_obj('QUANDL_EUREKA_'+util.date_now())








Name                       :        Eurekahedge Fund of Funds Index                   
Quandl Code                :        EUREKA/265                                        
Description                :        <p>Eurekahedge Fund of Funds Index. Eurekahedge is the world's largest independent data provider and research house dedicated to the collation, development and continuous improvement of alternative investment data.</p>
Frequency                  :        monthly                                           
Column Names               :        [u'Date', u'Returns']                             








quandl.ApiConfig.api_key = 'dffTSNJ4JHbRE1Csd7zZ'



quandl.bulkdownload("EUREKA",download_type="partial",filename="./ZEA.zip")





import util
  
util.aa_getmodule_doc("quandl")


!pip install fastnumbers


########################################################################################


quandl.



qd.search("EUREKA")




import quandl
mydata = quandl.get("FRED/GDP")


########################################################################################

data = quandl.get("WIKI/FB")




data = quandl.get("EUREKA")







########################################################################################


Quandl.Quandl
Quandl.version

 
 
Quandl.Quandl.HTTPError.info(self) 
Quandl.Quandl.HTTPError.__init__(self, fp, headers, url, code) 
Quandl.Quandl.HTTPError.__str__(self) 
Quandl.Quandl.HTTPError.__init__(self, url, code, msg, hdrs, fp) 
Quandl.Quandl.Request.set_proxy(self, host, type) 
Quandl.Quandl.Request.get_host(self) 
Quandl.Quandl.Request.is_unverifiable(self) 
Quandl.Quandl.Request.get_data(self) 
Quandl.Quandl.Request.add_data(self, data) 
Quandl.Quandl.Request.header_items(self) 
Quandl.Quandl.Request.__getattr__(self, attr) 
Quandl.Quandl.Request.has_proxy(self) 
Quandl.Quandl.Request.add_header(self, key, val) 
Quandl.Quandl.Request.get_type(self) 
Quandl.Quandl.Request.get_selector(self) 
Quandl.Quandl.Request.__init__(self, url, data, headers, origin_req_host, unverifiable) 
Quandl.Quandl.Request.get_header(self, header_name, default) 
Quandl.Quandl.Request.add_unredirected_header(self, key, val) 
Quandl.Quandl.Request.get_full_url(self) 
Quandl.Quandl.Request.get_origin_req_host(self) 
Quandl.Quandl.Request.has_header(self, header_name) 
Quandl.Quandl.Request.get_method(self) 
Quandl.Quandl.Request.has_data(self) 
Quandl.Quandl._append_query_fields(url) 
Quandl.Quandl._download(url) 
Quandl.Quandl._getauthtoken(token, text) 
Quandl.Quandl._htmlpush(url, raw_params) 
Quandl.Quandl._parse_dates(date) 
Quandl.Quandl._pushcodetest(code) 
Quandl.Quandl.genfromtxt(fname, dtype, comments, delimiter, skip_header, skip_footer, converters, missing_values, filling_values, usecols, names, excludelist, deletechars, replace_space, autostrip, case_sensitive, defaultfmt, unpack, usemask, loose, invalid_raise, max_rows) 
Quandl.Quandl.get(dataset) 
Quandl.Quandl.push(data, code, name, authtoken, desc, override, verbose, text) 
Quandl.Quandl.search(query, source, page, authtoken, verbose, prints) 
Quandl.Quandl.urlencode(query, doseq) 
Quandl.Quandl.urlopen(url, data, timeout, cafile, capath, cadefault, context) 

 
 

 
 
Quandl.Quandl.HTTPError.info(self) 
Quandl.Quandl.HTTPError.__init__(self, fp, headers, url, code) 
Quandl.Quandl.HTTPError.__str__(self) 
Quandl.Quandl.HTTPError.__init__(self, url, code, msg, hdrs, fp) 
Quandl.Quandl.Request.set_proxy(self, host, type) 
Quandl.Quandl.Request.get_host(self) 
Quandl.Quandl.Request.is_unverifiable(self) 
Quandl.Quandl.Request.get_data(self) 
Quandl.Quandl.Request.add_data(self, data) 
Quandl.Quandl.Request.header_items(self) 
Quandl.Quandl.Request.__getattr__(self, attr) 
Quandl.Quandl.Request.has_proxy(self) 
Quandl.Quandl.Request.add_header(self, key, val) 
Quandl.Quandl.Request.get_type(self) 
Quandl.Quandl.Request.get_selector(self) 
Quandl.Quandl.Request.__init__(self, url, data, headers, origin_req_host, unverifiable) 
Quandl.Quandl.Request.get_header(self, header_name, default) 
Quandl.Quandl.Request.add_unredirected_header(self, key, val) 
Quandl.Quandl.Request.get_full_url(self) 
Quandl.Quandl.Request.get_origin_req_host(self) 
Quandl.Quandl.Request.has_header(self, header_name) 
Quandl.Quandl.Request.get_method(self) 
Quandl.Quandl.Request.has_data(self) 
Quandl.Quandl._append_query_fields(url)  
       None 

Quandl.Quandl._download(url)  
       None 

Quandl.Quandl._getauthtoken(token, text)  
       Return and save API token to a pickle file for reuse. 

Quandl.Quandl._htmlpush(url, raw_params)  
       None 

Quandl.Quandl._parse_dates(date)  
       None 

Quandl.Quandl._pushcodetest(code)  
       None 

Quandl.Quandl.genfromtxt(fname, dtype, comments, delimiter, skip_header, skip_footer, converters, missing_values, filling_values, usecols, names, excludelist, deletechars, replace_space, autostrip, case_sensitive, defaultfmt, unpack, usemask, loose, invalid_raise, max_rows)  
       Load data from a text file, with missing values handled as specified.
       
       Each line past the first `skip_header` lines is split at the `delimiter`
       character, and characters following the `comments` character are discarded.
       
       Parameters
       ----------
       fname : file, str, list of str, generator
           File, filename, list, or generator to read.  If the filename
           extension is `.gz` or `.bz2`, the file is first decompressed. Mote
           that generators must return byte strings in Python 3k.  The strings
           in a list or produced by a generator are treated as lines.
       dtype : dtype, optional
           Data type of the resulting array.
           If None, the dtypes will be determined by the contents of each
           column, individually.
       comments : str, optional
           The character used to indicate the start of a comment.
           All the characters occurring on a line after a comment are discarded
       delimiter : str, int, or sequence, optional
           The string used to separate values.  By default, any consecutive
           whitespaces act as delimiter.  An integer or sequence of integers
           can also be provided as width(s) of each field.
       skiprows : int, optional
           `skiprows` was removed in numpy 1.10. Please use `skip_header` instead.
       skip_header : int, optional
           The number of lines to skip at the beginning of the file.
       skip_footer : int, optional
           The number of lines to skip at the end of the file.
       converters : variable, optional
           The set of functions that convert the data of a column to a value.
           The converters can also be used to provide a default value
           for missing data: ``converters = {3: lambda s: float(s or 0)}``.
       missing : variable, optional
           `missing` was removed in numpy 1.10. Please use `missing_values`
           instead.
       missing_values : variable, optional
           The set of strings corresponding to missing data.
       filling_values : variable, optional
           The set of values to be used as default when the data are missing.
       usecols : sequence, optional
           Which columns to read, with 0 being the first.  For example,
           ``usecols = (1, 4, 5)`` will extract the 2nd, 5th and 6th columns.
       names : {None, True, str, sequence}, optional
           If `names` is True, the field names are read from the first valid line
           after the first `skip_header` lines.
           If `names` is a sequence or a single-string of comma-separated names,
           the names will be used to define the field names in a structured dtype.
           If `names` is None, the names of the dtype fields will be used, if any.
       excludelist : sequence, optional
           A list of names to exclude. This list is appended to the default list
           return,file,print. Excluded names are appended an underscore:
           for example, `file` would become `file_`.
       deletechars : str, optional
           A string combining invalid characters that must be deleted from the
           names.
       defaultfmt : str, optional
           A format used to define default field names, such as "f%i" or "f_%02i".
       autostrip : bool, optional
           Whether to automatically strip white spaces from the variables.
       replace_space : char, optional
           Character(s) used in replacement of white spaces in the variables
           names. By default, use a _.
       case_sensitive : {True, False, upper, lower}, optional
           If True, field names are case sensitive.
           If False or upper, field names are converted to upper case.
           If lower, field names are converted to lower case.
       unpack : bool, optional
           If True, the returned array is transposed, so that arguments may be
           unpacked using ``x, y, z = loadtxt(...)``
       usemask : bool, optional
           If True, return a masked array.
           If False, return a regular array.
       loose : bool, optional
           If True, do not raise errors for invalid values.
       invalid_raise : bool, optional
           If True, an exception is raised if an inconsistency is detected in the
           number of columns.
           If False, a warning is emitted and the offending lines are skipped.
       max_rows : int,  optional
           The maximum number of rows to read. Must not be used with skip_footer
           at the same time.  If given, the value must be at least 1. Default is
           to read the entire file.
       
           .. versionadded:: 1.10.0
       
       Returns
       -------
       out : ndarray
           Data read from the text file. If `usemask` is True, this is a
           masked array.
       
       See Also
       --------
       numpy.loadtxt : equivalent function when no data is missing.
       
       Notes
       -----
       * When spaces are used as delimiters, or when no delimiter has been given
         as input, there should not be any missing data between two fields.
       * When the variables are named (either by a flexible dtype or with `names`,
         there must not be any header in the file (else a ValueError
         exception is raised).
       * Individual values are not stripped of spaces by default.
         When using a custom converter, make sure the function does remove spaces.
       
       References
       ----------
       .. 1 Numpy User Guide, section `I/O with Numpy
              <http://docs.scipy.org/doc/numpy/user/basics.io.genfromtxt.html>`_.
       
       Examples
       ---------
       >>> from io import StringIO
       >>> import numpy as np
       
       Comma delimited file with mixed dtype
       
       >>> s = StringIO("1,1.3,abcde")
       >>> data = np.genfromtxt(s, dtype=(myint,i8),(myfloat,f8),
       ... (mystring,S5), delimiter=",")
       >>> data
       array((1, 1.3, abcde),
             dtype=(myint, <i8), (myfloat, <f8), (mystring, |S5))
       
       Using dtype = None
       
       >>> s.seek(0) # needed for StringIO example only
       >>> data = np.genfromtxt(s, dtype=None,
       ... names = myint,myfloat,mystring, delimiter=",")
       >>> data
       array((1, 1.3, abcde),
             dtype=(myint, <i8), (myfloat, <f8), (mystring, |S5))
       
       Specifying dtype and names
       
       >>> s.seek(0)
       >>> data = np.genfromtxt(s, dtype="i8,f8,S5",
       ... names=myint,myfloat,mystring, delimiter=",")
       >>> data
       array((1, 1.3, abcde),
             dtype=(myint, <i8), (myfloat, <f8), (mystring, |S5))
       
       An example with fixed-width columns
       
       >>> s = StringIO("11.3abcde")
       >>> data = np.genfromtxt(s, dtype=None, names=intvar,fltvar,strvar,
       ...     delimiter=1,3,5)
       >>> data
       array((1, 1.3, abcde),
             dtype=(intvar, <i8), (fltvar, <f8), (strvar, |S5)) 

Quandl.Quandl.get(dataset)  
       Return dataframe of requested dataset from Quandl.
       
       :param dataset: str or list, depending on single dataset usage or multiset usage
               Dataset codes are available on the Quandl website
       :param str authtoken: Downloads are limited to 10 unless token is specified
       :param str trim_start, trim_end: Optional datefilers, otherwise entire
              dataset is returned
       :param str collapse: Options are daily, weekly, monthly, quarterly, annual
       :param str transformation: options are diff, rdiff, cumul, and normalize
       :param int rows: Number of rows which will be returned
       :param str sort_order: options are asc, desc. Default: `asc`
       :param str returns: specify what format you wish your dataset returned as,
           either `numpy` for a numpy ndarray or `pandas`. Default: `pandas`
       :param bool verbose: specify whether to print output text to stdout, default is False.
       :param str text: Deprecated. Use `verbose` instead.
       :returns: :class:`pandas.DataFrame` or :class:`numpy.ndarray`
       
       Note that Pandas expects timeseries data to be sorted ascending for most
       timeseries functionality to work.
       
       Any other `kwargs` passed to `get` are sent as field/value params to Quandl
       with no interference. 

Quandl.Quandl.push(data, code, name, authtoken, desc, override, verbose, text)  
       Upload a pandas Dataframe to Quandl and returns link to the dataset.
       
       :param data: (required), pandas ts or numpy array
       :param str code: (required), Dataset code
                    must consist of only capital letters, numbers, and underscores
       :param str name: (required), Dataset name
       :param str authtoken: (required), to upload data
       :param str desc: (optional), Description of dataset
       :param bool verbose: specify whether to print output text to stdout, default is False.
       :param str text: Deprecated. Use `verbose` instead.
       
       :returns: :str: link to uploaded dataset 

Quandl.Quandl.search(query, source, page, authtoken, verbose, prints)  
       Return array of dictionaries of search results.
       :param str query: (required), query to search with
       :param str source: (optional), source to search
       :param +ve int: (optional), page number of search 
       :param str authotoken: (optional) Quandl auth token for extended API access
       :returns: :array: search results 

Quandl.Quandl.urlencode(query, doseq)  
       Encode a sequence of two-element tuples or dictionary into a URL query string.
       
       If any values in the query arg are sequences and doseq is true, each
       sequence element is converted to a separate parameter.
       
       If the query arg is a sequence of two-element tuples, the order of the
       parameters in the output will match the order of parameters in the
       input. 

Quandl.Quandl.urlopen(url, data, timeout, cafile, capath, cadefault, context)  
       None 


 
 

 
 
 
 -----------------------------------------------------------------------------

 
Module: Quandl.Quandl-------------------------------------------------
    
   +Class: CallLimitExceeded
        (No members)
    
   +Class: CodeFormatError
        (No members)
    
   +Class: DatasetNotFound
        (No members)
    
   +Class: DateNotRecognized
        (No members)
    
   +Class: ErrorDownloading
        (No members)
    
   +Class: HTTPError
          +  info(self)
          +  __init__(self, fp, headers, url, code)
        	  	  Default_Args:(code, None)
          +  __str__(self)
          +  __init__(self, url, code, msg, hdrs, fp)
    
   +Class: MissingToken
        (No members)
    
   +Class: MultisetLimit
        (No members)
    
   +Class: ParsingError
        (No members)
    
   +Class: QuandlError
        (No members)
    
   +Class: Request
          +  set_proxy(self, host, type)
          +  get_host(self)
          +  is_unverifiable(self)
          +  get_data(self)
          +  add_data(self, data)
          +  header_items(self)
          +  __getattr__(self, attr)
          +  has_proxy(self)
          +  add_header(self, key, val)
          +  get_type(self)
          +  get_selector(self)
          +  __init__(self, url, data, headers, origin_req_host, unverifiable)
        	  	  Default_Args:(data, None), (headers, {}), (origin_req_host, None), (unverifiable, False)
          +  get_header(self, header_name, default)
        	  	  Default_Args:(default, None)
          +  add_unredirected_header(self, key, val)
          +  get_full_url(self)
          +  get_origin_req_host(self)
          +  has_header(self, header_name)
          +  get_method(self)
          +  has_data(self)
    
   +Class: TokenError
        (No members)
    
   +Class: UnknownError
        (No members)
    
   +Class: WrongFormat
        (No members)
      +Func: _append_query_fields(url)
    	   Keyword_Args: kwargs
      +Func: _download(url)
      +Func: _getauthtoken(token, text)
      +Func: _htmlpush(url, raw_params)
      +Func: _parse_dates(date)
      +Func: _pushcodetest(code)
      +Func: genfromtxt(fname, dtype, comments, delimiter, skip_header, skip_footer, converters, missing_values, filling_values, usecols, names, excludelist, deletechars, replace_space, autostrip, case_sensitive, defaultfmt, unpack, usemask, loose, invalid_raise, max_rows)
    	  	  Default_Args:(dtype, <type float>), (comments, #), (delimiter, None), (skip_header, 0), (skip_footer, 0), (converters, None), (missing_values, None), (filling_values, None), (usecols, None), (names, None), (excludelist, None), (deletechars, None), (replace_space, _), (autostrip, False), (case_sensitive, True), (defaultfmt, f%i), (unpack, None), (usemask, False), (loose, True), (invalid_raise, True), (max_rows, None)
      +Func: get(dataset)
    	   Keyword_Args: kwargs
      +Func: push(data, code, name, authtoken, desc, override, verbose, text)
    	  	  Default_Args:(authtoken, u), (desc, u), (override, False), (verbose, False), (text, None)
      +Func: search(query, source, page, authtoken, verbose, prints)
    	  	  Default_Args:(source, None), (page, 1), (authtoken, None), (verbose, True), (prints, None)
    
   +Class: unicode
        (No members)
      +Func: urlencode(query, doseq)
    	  	  Default_Args:(doseq, 0)
      +Func: urlopen(url, data, timeout, cafile, capath, cadefault, context)
    	  	  Default_Args:(data, None), (timeout, <object object at 0x0000000003787400>), (cafile, None), (capath, None), (cadefault, False), (context, None)

 
Module: Quandl.version-------------------------------------------------
    (No members)



Eurekahedge Multi–Factor Risk Premia Index
Eurekahedge Multi:Factor Risk Premia Index. Eurekahedge is the world's largest independent data provider and research house dedicated to the collation development and continuous improvement of alternative investment data.
CODE: EUREKA/MEI27
Latest: 2016-08-31
Monthly, since 2010
MEBI Zero Beta Strategy L1 w/DRC
MEBI Zero Beta Strategy L1 w/DRC. Eurekahedge is the world's largest independent data provider and research house dedicated to the collation, development and continuous improvement of alternative investment data.
CODE: EUREKA/MEI26
Latest: 2016-08-31
Monthly, since 2015
MEBI Zero Beta Strategy L1
MEBI Zero Beta Strategy L1. Eurekahedge is the world's largest independent data provider and research house dedicated to the collation, development and continuous improvement of alternative investment data.
CODE: EUREKA/MEI25
Latest: 2016-08-31
Monthly, since 1994
MEBI Maximum Sharpe Ratio Strategy L1
MEBI Maximum Sharpe Ratio Strategy L1. Eurekahedge is the world's largest independent data provider and research house dedicated to the collation development and continuous improvement of alternative investment data.
CODE: EUREKA/MEI24
Latest: 2016-08-31
Monthly, since 1991
Eurekahedge Asia ex Japan Multi–Strategy Hedge Fund Index
Eurekahedge Asia ex Japan Multi:Strategy Hedge Fund Index. Eurekahedge is the world's largest independent data provider and research house dedicated to the collation, development and continuous improvement of alternative investment data.
CODE: EUREKA/8
Latest: 2016-07-31
Monthly, since 2000
Eurekahedge Asia Pacific Long Short Equities Fund of Funds Index
Eurekahedge Asia Pacific Long Short Equities Fund of Funds Index. Eurekahedge is the world's largest independent data provider and research house dedicated to the collation, development and continuous improvement of alternative investment data.
CODE: EUREKA/180
Latest: 2016-07-31
Monthly, since 2000
Eurekahedge CTA/Managed Futures Fund of Funds Index
Eurekahedge CTA/Managed Futures Fund of Funds Index. Eurekahedge is the world's largest independent data provider and research house dedicated to the collation, development and continuous improvement of alternative investment data.
CODE: EUREKA/256
Latest: 2016-07-31
Monthly, since 2000
Eurekahedge Medium Hedge Fund Index (US$100m – US$500m)
Eurekahedge Medium Hedge Fund Index (US$100m : US$500m). Eurekahedge is the world's largest independent data provider and research house dedicated to the collation, development and continuous improvement of alternative investment data.
CODE: EUREKA/487
Latest: 2016-07-31
Monthly, since 2000
Mizuho–Eurekahedge Asia Pacific Multi–Strategy Index – USD
Mizuho:Eurekahedge Asia Pacific Multi:Strategy Index : USD. Eurekahedge is the world's largest independent data provider and research house dedicated to the collation, development and continuous improvement of alternative investment data.
CODE: EUREKA/MEI17
Latest: 2016-07-31
Monthly, since 2003
Eurekahedge Small Latin American Hedge Fund Index (< US$25m)
Eurekahedge Small Latin American Hedge Fund Index (< US$25m). Eurekahedge is the world's largest independent data provider and research house dedicated to the collation, development and continuous improvement of alternative investment data.
CODE: EUREKA/303
Latest: 2016-07-31
Monthly, since 2000
Eurekahedge Medium Absolute Return Fund Index (US$50m – US$250m)
Eurekahedge Medium Absolute Return Fund Index (US$50m : US$250m). Eurekahedge is the world's largest independent data provider and research house dedicated to the collation, development and continuous improvement of alternative investment data.
CODE: EUREKA/334
Latest: 2016-07-31
Monthly, since 2000
Eurekahedge Emerging Markets Multi–Strategy Hedge Fund Index
Eurekahedge Emerging Markets Multi:Strategy Hedge Fund Index. Eurekahedge is the world's largest independent data provider and research house dedicated to the collation, development and continuous improvement of alternative investment data.
CODE: EUREKA/467
Latest: 2016-07-31
Monthly, since 2000
Eurekahedge Asia inc Japan Relative Value Hedge Fund Index
Eurekahedge Asia inc Japan Relative Value Hedge Fund Index. Eurekahedge is the world's largest independent data provider and research house dedicated to the collation, development and continuous improvement of alternative investment data.
CODE: EUREKA/498
Latest: 2016-07-31
Monthly, since 2000
Eurekahedge Islamic Fund Middle East/Africa Fixed Income Index
Eurekahedge Islamic Fund Middle East/Africa Fixed Income Index. Eurekahedge is the world's largest independent data provider and research house dedicated to the collation, development and continuous improvement of alternative investment data.
CODE: EUREKA/543
Latest: 2016-07-31
Monthly, since 2000
Mizuho–Eurekahedge Asia Pacific ex Japan Multi–Strategy Index – USD
Mizuho:Eurekahedge Asia Pacific ex Japan Multi:Strategy Index : USD. Eurekahedge is the world's largest independent data provider and research house dedicated to the collation, development and continuous improvement of alternative investment data.
CODE: EUREKA/MEI20
Latest: 2016-07-31
Monthly, since 2003
Eurekahedge Latin American Event Driven Hedge Fund Index
Eurekahedge Latin American Event Driven Hedge Fund Index. Eurekahedge is the world's largest independent data provider and research house dedicated to the collation, development and continuous improvement of alternative investment data.
CODE: EUREKA/149
Latest: 2016-07-31
Monthly, since 2003
Eurekahedge Global Multi–Strategy Fund of Funds Index
Eurekahedge Global Multi:Strategy Fund of Funds Index. Eurekahedge is the world's largest independent data provider and research house dedicated to the collation, development and continuous improvement of alternative investment data.
CODE: EUREKA/215
Latest: 2016-07-31
Monthly, since 2000
Eurekahedge Arbitrage Fund of Funds Index
Eurekahedge Arbitrage Fund of Funds Index. Eurekahedge is the world's largest independent data provider and research house dedicated to the collation development and continuous improvement of alternative investment data.
CODE: EUREKA/255
Latest: 2016-07-31
Monthly, since 2000
Eurekahedge Long Short Equities Fund of Funds Index
Eurekahedge Long Short Equities Fund of Funds Index. Eurekahedge is the world's largest independent data provider and research house dedicated to the collation, development and continuous improvement of alternative investment data.
CODE: EUREKA/258
Latest: 2016-07-31
Monthly, since 2000
Eurekahedge Event Driven Fund of Funds Index
Eurekahedge Event Driven Fund of Funds Index. Eurekahedge is the world's largest independent data provider and research house dedicated to the collation development and continuous improvement of alternative investment data.
CODE: EUREKA/259























