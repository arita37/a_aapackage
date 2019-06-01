# -*- coding: utf-8 -*-
import os

import util


# Google Drive


def googledrive_get():
    """
   https://github.com/ctberthiaume/gdcp
   ... I am using this now to transfer thousands of mp3 files from a ubuntu vps to google drive.


http://olivermarshall.net/how-to-upload-a-file-to-google-drive-from-the-command-line/
https://github.com/prasmussen/gdrive  : Super Complete

gdrive [global] upload [options] <path>

global:
  -c, --config <configDir>         Application path, default: /Users/<user>/.gdrive
  --refresh-token <refreshToken>   Oauth refresh token used to get access token (for advanced users)
  --access-token <accessToken>     Oauth access token, only recommended for short-lived requests
                                   because of short lifetime (for advanced users)

options:
  -r, --recursive           Upload directory recursively
  -p, --parent <parent>     Parent id, used to upload file to a specific directory, can be
                            specified multiple times to give many parents
  --name <name>             Filename
  --no-progress             Hide progress
  --mime <mime>             Force mime type
  --share                   Share file
  --delete                  Delete local file when upload is successful
  --chunksize <chunksize>   Set chunk size in bytes, default: 8388608

   :return:
   """
    pass


def googledrive_put():
    """
  100 GB: 2USD,  1TB: 10USD
  https://gsuite.google.com/intl/en/pricing.html

  :return:
  """
    pass


def googledrive_list():
    pass


def os_processify_fun(func):
    """Decorator to run a function as a process.
    Be sure that every argument and the return value is *pickable*.
    The created process is joined, so the code does not  run in parallel.
    @processify

    def test():
      return os.getpid()

    @processify
    def test_deadlock():
      return range(30000)

   @processify
   def test_exception():
     raise RuntimeError('xyz')

   def test():
     print os.getpid()
     print test_function()
     print len(test_deadlock())
     test_exception()

   if __name__ == '__main__':
     test()

    """
    import sys
    import traceback
    from functools import wraps
    from multiprocessing import Process, Queue

    def process_func(q, *args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except Exception:
            ex_type, ex_value, tb = sys.exc_info()
            error = ex_type, ex_value, "".join(traceback.format_tb(tb))
            ret = None
        else:
            error = None

        q.put((ret, error))

    # register original function with different name
    # in sys.modules so it is pickable
    process_func.__name__ = func.__name__ + "processify_func"
    setattr(sys.modules[__name__], process_func.__name__, process_func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        q = Queue()
        p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
        p.start()
        ret, error = q.get()
        # p.join()

        if error:
            ex_type, ex_value, tb_str = error
            message = "%s (in subprocess)\n%s" % (ex_value.message, tb_str)
            raise ex_type(message)

        return ret

    return wrapper


@os_processify_fun
def ztest_processify():
    return os.getpid()


def a_module_codesample(module_str="pandas"):
    dir1 = "D:/_devs/Python01/project27/docs/"
    file1 = dir1 + "/" + module_str + "/codesample.py"
    txt = util.os_file_read(file1)
    util.os_gui_popup_show(txt)


def a_module_doc(module_str="pandas"):
    dir1 = "D:/_devs/Python01/project27/docs/"
    file1 = dir1 + "/" + module_str + "/doc.py"
    txt = util.os_file_read(file1)
    util.os_gui_popup_show(txt)


# Object Class Introspection
def obj_getclass_of_method(meth):
    import inspect

    for cls in inspect.getmro(meth.__self__.__class__):
        if meth.__name__ in cls.__dict__:
            return cls
    return None


def obj_getclass_property(pfi):
    for prop, value in vars(pfi).items():
        print(prop, ": ", value)


# Object
def sql_getdate():
    pass


"""
solver1= load_obj('batch/solver_ELVIS_USD_9assets_3_regime_perf_20160906_059')

solver1.x
solver1.convergence
._calculate_population_energies
solver1.next()  #One generation evolution
solver1.solve()  #Solve the pronlem

solver1.population           # Get the list of population sol
solver1.population_energies  # Population energy

aux= solver1.population


solver1.next()  #One generation evolution


def np_runsolver(name1, niter):

pfi= copy.deepcopy( get_class_that_defined_method(solver1.func) )

obj_getclass_property(pfi)

solver1.x



id : int
date:
time:
name: str
solver
xbest
fbest
population
population_energies
params1
params2
params3
params4
params5
details

storage_optim= np.empty((1000, 15), dtype= np.object)


  id= ['id','date', 'time','name','solver','xbest', 'fbest', 'population','pop_energies',
  'params1', 'params2', 'params3', 'params4', 'params5', 'details']      )

def tostore('store/'+dbname, dbid, vv) :
 storage_optim=  load_obj(dbname)

 storage_optim[0,0]= 0
 storage_optim[0,1]= date_now()
 storage_optim[0,2]= copy.deepcopy(solver1)

 copyfile(dbname)
 save_obj(storage_optim, 'store/'+dbname)


aux= (solver1.population ,  solver1.population_energies  )
save_obj(aux, 'batch/elvis_usd_9assets_3_regime_perf_best_population_01')


"""


# XML / HTML processing
"""
https://pypi.python.org/pypi/RapidXml/
http://pugixml.org/benchmark.html

"""


# -------- PDF processing ------------------------------------------------------------
def print_topdf():
    import datetime
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.
    with PdfPages("multipage_pdf.pdf") as pdf:
        plt.figure(figsize=(3, 3))
        plt.plot(list(range(7)), [3, 1, 4, 1, 5, 9, 2], "r-o")
        plt.title("Page One")
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        plt.rc("text", usetex=True)
        plt.figure(figsize=(8, 6))
        x = np.arange(0, 5, 0.1)
        plt.plot(x, np.sin(x), "b-")
        plt.title("Page Two")
        pdf.savefig()
        plt.close()

        plt.rc("text", usetex=False)
        fig = plt.figure(figsize=(4, 5))
        plt.plot(x, x * x, "ko")
        plt.title("Page Three")
        pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
        plt.close()

        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d["Title"] = "Multipage PDF Example"
        d["Author"] = "Jouni K. Sepp\xe4nen"
        d["Subject"] = "How to create a multipage pdf file and set its metadata"
        d["Keywords"] = "PdfPages multipage keywords author title subject"
        d["CreationDate"] = datetime.datetime(2009, 11, 13)
        d["ModDate"] = datetime.datetime.today()


# --------CSV processing -------------------------------------------------------------
# Put Excel and CSV into Database / Extract CSV from database


"""
http://www.ibm.com/developerworks/library/l-pyint/

spam.upper().lower()

spam.strip()
spam.lstrip()
spam.rstrip()  Performs both lstrip() and rstrip() on string

count(str, beg= 0,end=len(string))  Counts how many times str occurs in string or in a substring of
find(str, beg=0 end=len(string))  str occurs in string or in a substring of strindex if found
-1 otherwise.
replace(old, new [, max])   Replaces all occurrences of old in string with new or at most
max occurrences if max given.

isdecimal()   Returns true if a unicode string contains only decimal characters and false otherwise.
isalnum()  Returns true if string has at least 1 character and all characters are alphanumeric and
false otherwise.
salpha()   Returns true if string has at least 1 character and all characters are alphabetic and
false otherwise.
isdigit()  Returns true if string contains only digits and false otherwise.
islower()  Returns true if string has at least 1 cased character and all cased characters are in
lowercase and false otherwise.
isnumeric() Returns true if a unicode string contains only numeric characters and false otherwise.
isspace()  Returns true if string contains only whitespace characters and false otherwise.
istitle()  Returns true if string is properly "titlecased" and false otherwise.

lower()   Converts all uppercase letters in string to lowercase.
capitalize()   Capitalizes first letter of string
center(width, fillchar)  Returns space-padded string with string centered  w
title()  Returns "titlecased" version of string, that is, all words begin with uppercase and the
rest are lowercase.
upper()   Converts lowercase letters in string to uppercase.

join(seq)  Merges (concatenates) the string representations of elements in sequence seq into a
string, with separator string.

split(str="", num=string.count(str))   Splits string according to delimiter str  and returns list
of substrings;
splitlines( num=string.count('\n'))  Splits string at all (or num) NEWLINEs and returns a list of
each line with NEWLINEs removed.

ljust(width[, fillchar])   Returns a space-padded string with the original string left-justified to
a total of width columns.

maketrans()  Returns a translation table to be used in translate function.

index(str, beg=0, end=len(string))   Same as find(), but raises an exception if str not found.
rfind(str, beg=0,end=len(string))   Same as find(), but search backwards in string.

rindex( str, beg=0, end=len(string))   Same as index(), but search backwards in string.

rjust(width,[, fillchar])  Returns a space-padded string with string right-jus to a total of width
columns.

startswith(str, beg=0,end=len(string))
Determines if string or a substring of string (if starting index beg and ending index end are given)
starts with substring str; returns true if so and false otherwise.

decode(encoding='UTF-8',errors='strict')
Decodes the string using the codec registered for encoding. encoding defaults to the default string
encoding.

encode(encoding='UTF-8',errors='strict')
Returns encoded string version of string; on error, default is to raise a ValueError

endswith(suffix, beg=0, end=len(string))
Determines if string or a substring of string  ends with suffix; returns true if so and false
otherwise.

expandtabs(tabsize=8)
Expands tabs in string to multiple spaces; defaults to 8 spaces per tab if tabsize not provided.

zfill (width)
Returns original string leftpadded with zeros to a total of width characters; intended for numbers,
zfill() retains any sign given (less one zero).

"""
