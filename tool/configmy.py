# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from importlib import import_module
from pkgutil import walk_packages
import builtins, operator, inspect, future
####################################################################################################

import regex, past, ast, re,   os, sys, glob, platform, arrow
from collections import OrderedDict
from attrdict import AttrDict as dict2


####################################################################################################
#__path__= '/'
#__version__= "1.0.0"
#__file__= "configmy.py"
__all__ = ['get_environ_details', 'get_config_from_environ', 'get', 'set']
CONFIGMY_ROOT_FILE= "CONFIGMY_ROOT_FILE"

try:
   import configmy
   PACKAGE_PATH= configmy.__path__[0] + "/"
except:
   PACKAGE_PATH= ""


####################################################################################################
def zdoc() :
 source= inspect.getsource(ztest)
 print(source)





def os_file_replacestring1(findStr, repStr, filePath):
    "replaces all findStr by repStr in file filePath"
    import fileinput
    file1= fileinput.FileInput(filePath, inplace=True, backup='.bak')
    for line in file1:
       line= line.replace(findStr,  repStr)
       sys.stdout.write(line)
    file1.close()
    print(("OK: "+format(filePath)))



def os_popen(cmd):
      result = subprocess.Popen(cmd, shell=True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      output,error = result.communicate()
      return output


def os_grep(search, folder='/home/ubuntu/', exclude="" ):
    '''
      LC_ALL=C fgrep -Irsl --exclude-dir=\*ubuntu\*  --exclude=\*.{py}   'DIRCWD=' /
    '''
    cmd= "fgrep  -Irsl '" +search+"' "+ folder
    a= os.popen( cmd ).read()
    print(a)
    # grep -rs  "export CONDA_ROOT" /home/ubuntu/




####################################################################################################
def get_environ_details(isprint=0):
 '''
  Calculate environnment details
  platform.system() #returns the base system, in your case Linux
  platform.release() #returns release version
  Dynamic release and data
 '''
 CFG   = {'os': sys.platform[:3],
          'username': os.path.expanduser('~').split("\\")[-1].split("/")[-1],
          "pythonversion":      str(sys.version_info.major),
          "pythonversion_full": str(sys.version_info),
          'os_release' :        str(platform.release()),
          'processor' :         str(platform.processor())
          # print subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT)

          }

 if isprint :  print(CFG); return CFG
 else :
   # CFG= dict2(CFG);        
   return CFG



def get_config_from_environ(CFG, dd_config_all, method0=["os", "username"]):
    '''
    Create unique id from method, using environnment details
    :param CFG: 
    :param dd_config_all: 
    :param method0:    method specify in os release, os, username, 
     method0=["os_", "username_"]  :           "win+MyUser1'    
     method0=["os_", "username_", "test"]  :   "win+MyUser1+test'
     method0=["os_", "username_", "prod"]  :   "win+MyUser1+prod'
     
     Order of preference :
       1) argument method0
       2) CONFIGMY_ROOT_FILE["configmy"]["method0"]
       3) Default method os_username    
    '''

    #Check configmy_root_file for method0
    method_default= "os_username"
    if "_".join(method0) == method_default :
       try :     method0= dd_config_all["configmy"]["method0"]
       except :  pass


    config_unique_id= ""   #  os_env_id
    for x in method0 :
       try :     key= str(CFG[x])  # add value of config
       except :  key= x            # add prefix

       if config_unique_id == "": config_unique_id= key
       else :                     config_unique_id=  config_unique_id + "+" + key

    dd_config= dd_config_all[config_unique_id]
    return dd_config



def get(config_file="_ROOT", method0=["os", "username"], output=["_CFG", "DIRCWD",]):
    ''' Get the config dictionary.
    method0:       os, username, pythonversion  
    config_file:  _ROOT: os.environ["CONFIGMY_ROOT_FILE"]  /   CONFIGMY_ROOT_FILE.py
    outputs:      _ALL: full file, _CFG : Config File,  DIRCWD root directory 
    '''
    try :
        if config_file == "_ROOT" :  config_file= os.environ[CONFIGMY_ROOT_FILE]
    except Exception as e :
        print(e);  print("Cannot Find os.environ['CONFIGMY_ROOT_FILE'] ")
        return None


    try :
      with   open(config_file) as f1 :
        dd_config_all = ast.literal_eval(f1.read())
    except Exception as e:
      print(e) ;   print("Incorrect config file Dictionnary format")
      return None


    CFG=       get_environ_details()
    dd_config= get_config_from_environ(CFG, dd_config_all, method0=method0)
    dd_config.update(CFG)    #Add system parameters os, system
    dd_config= dict2(dd_config)


    dd_config_all.update(CFG)
    dd_config_all= dict2(dd_config_all)


    #################################################################################
    output_tuple= ()
    for x in output :
      if x[0] != "_" :  output_tuple= (output_tuple + (dd_config[x],) )   #Ask directly items argument
      if x == "_CFG" :  output_tuple= (output_tuple + (dd_config,))
      if x == "_ALL" :  output_tuple= (output_tuple + (dd_config_all,))


    return output_tuple




def set(configmy_root_file="") :
    '''
    Do Command Line to set configmy root file in   os.environ['CONFIGMY_ROOT_FILE'] 
    '''
    print("Setuping PERMANENTLY   'CONFIGMY_ROOT_FILE' in Env Var ")
    CFG=      get_environ_details(isprint=0)
    env_var = CONFIGMY_ROOT_FILE

    os.environ[env_var] = configmy_root_file # visible in this process + all children

    if CFG["os"] == "win" :
      os.system("SETX {0} {1} /M".format(env_var, configmy_root_file))
      os.system("SETX {0} {1} ".format(env_var, configmy_root_file))
      print("You need to reboot Windows to get the Env Var visible, permanently")


    if CFG["os"] == "lin" :
      with open(os.path.expanduser("~/.bashrc"), "a") as outfile:
         print(" ./.bashrc")
         outfile.write("export  "+env_var + "="+configmy_root_file)
      os.system("source ~/.bashrc")  #Reload the bash

      with open("sudo /etc/environment", "a") as outfile:
         print(" /etc/environment")
         outfile.write(env_var + "="+configmy_root_file)

      os.system("source /etc/environnment")  #Reload the bash

      print("""
      Please setup manually here by sudo user
       sudo nano /etc/environment
       /etc/environment - 
       This file is specifically meant for system-wide environment variable settings. 
       It is not a script file, but rather consists of assignment expressions, one per line. 
       Specifically, this file stores the system-wide locale and path settings.
       Needed when launching Jupyter from crontab -e

       Not recommended:
       /etc/profile - 
       This file gets executed whenever a bash login shell is entered (e.g. when logging in from the console or over ssh), as well as by the DisplayManager when the desktop session loads. This is probably the file you will get referred to when asking veteran UNIX system administrators about environment variables. In Ubuntu, however, this file does little more then invoke the /etc/bash.bashrc file.

       /etc/bash.bashrc - This is the system-wide version of the ~/.bashrc file. 
       Ubuntu is configured by default to execute this file whenever a user enters a shell or the desktop environment.


       #### System Variables == os.environ  Ubuntu
       nano   ./.bashrc
       source ./.bashrc                           #  Reload file
       export CONFIGMY_ROOT_FILE=/home/ubuntu/myconfig.file

       nano /etc/profile                   #  Reload file
       source /etc/profile  
       export CONFIGMY_ROOT_FILE=/home/ubuntu/myconfig.file

      """)

    if CFG["os"] == "mac" :
       pass






#####################################################################################################
def conda_env_list() :
 import os
 a= os.popen('conda env list').read()
 ll= [ x.split(" ")[0]   for x in  a.split("\n")  if  x.split(" ")[0] not in ["#", ""]  ]
 return ll


def conda_install(package_list=['configmy==0.13.5'], condaenv_list=[], install_tool="pip/conda") :
    '''
    #  /anaconda/envs/venv_name/bin/pip install
    Cannot install on root, please install manually, No-deps, no update
    '''
    import os, sys
    if   os.__file__.find("envs") > -1 :  DIRANA= os.__file__.split("envs")[0]        # Anaconda Directory
    elif os.__file__.find("lib") > -1 :   DIRANA= os.__file__.split("lib")[0]   +"/"  # Anaconda from root
    elif os.__file__.find("Lib") > -1 :   DIRANA= os.__file__.split("Lib")[0]   +"/"  # Anaconda from root

    os_name= sys.platform[:3]

    if install_tool == "pip" :
      for package0 in package_list :
       for condaenv in condaenv_list :
         if condaenv != 'root' :

          if os_name == 'lin' :   DIR1=  DIRANA +'/envs/'+ condaenv +'/bin/'
          elif os_name== "win" :  DIR1=  DIRANA +'/envs/'+ condaenv +'/Scripts/'

          cmd= DIR1 +'pip install  --upgrade --no-deps --force-reinstall '+package0
          print("condaenv: " + condaenv + " , "+package0)
          print("     " + cmd)
          sys.stdout.flush()
          #condaenv= 'sandbox'
          try :
            a= os.popen( cmd ).read()
            print("     "+a)
          except Exception as e:print(e)
          sys.stdout.flush()


    if install_tool == "conda" :
      for package0 in package_list :
       for condaenv in condaenv_list :
         if condaenv != 'root' :

          if   os_name == 'lin' :  DIR2=  DIRANA +'/bin/'
          elif os_name==  "win" :  DIR2=  DIRANA +'/Scripts/'

          cmd= DIR2+'conda install --no-deps --no-update-deps --verbose  --yes -n '+condaenv + ' '  +package0

          print("condaenv: " + condaenv + " , "+package0)
          print("     " + cmd);   sys.stdout.flush()
          try :
            a= os.popen( cmd ).read()
            print("     "+a)
          except Exception as e:print(e)
          sys.stdout.flush()




def conda_uninstall(package_list=['configmy==0.13.5'], condaenv_list=[], install_tool="pip/conda") :
    '''
    #  /anaconda/envs/venv_name/bin/pip install
    Cannot install on root, please install manually, No-deps, no update
    '''
    import os, sys
    if   os.__file__.find("envs") > -1 :  DIRANA= os.__file__.split("envs")[0]        # Anaconda Directory
    elif os.__file__.find("lib") > -1 :   DIRANA= os.__file__.split("lib")[0]   +"/"  # Anaconda from root
    elif os.__file__.find("Lib") > -1 :   DIRANA= os.__file__.split("Lib")[0]   +"/"  # Anaconda from root

    os_name= sys.platform[:3]

    if install_tool == "pip" :
      for package0 in package_list :
       for condaenv in condaenv_list :
         if condaenv != 'root' :

          if os_name == 'lin' :   DIR1=  DIRANA +'/envs/'+ condaenv +'/bin/'
          elif os_name== "win" :  DIR1=  DIRANA +'/envs/'+ condaenv +'/Scripts/'

          cmd= DIR1 +'pip uninstall  --yes    '+package0
          print("condaenv: " + condaenv + " , "+package0)
          print("     " + cmd)
          sys.stdout.flush()
          #condaenv= 'sandbox'
          try :
            a= os.popen( cmd ).read()
            print("     "+a)
          except Exception as e:print(e)
          sys.stdout.flush()


    if install_tool == "conda" :
      for package0 in package_list :
       for condaenv in condaenv_list :
         if condaenv != 'root' :

          if   os_name == 'lin' :  DIR2=  DIRANA +'/bin/'
          elif os_name==  "win" :  DIR2=  DIRANA +'/Scripts/'

          cmd= DIR2+'conda uninstall --force --verbose  --yes -n '+condaenv + ' '  +package0

          print("condaenv: " + condaenv + " , "+package0)
          print("     " + cmd);   sys.stdout.flush()
          try :
            a= os.popen( cmd ).read()
            print("     "+a)
          except Exception as e:print(e)
          sys.stdout.flush()




def conda_path_get() :
   if   os.__file__.find("lib") > -1 :   DIRANA= os.__file__.split("lib")[0]   +"/"  # Anaconda from linux
   elif os.__file__.find("Lib") > -1 :  DIRANA= os.__file__.split("Lib")[0]   +"/"  # Anaconda from root

   os_name= sys.platform[:3]
   if   os_name == 'lin' :  DIR2=  DIRANA +'/bin/'
   elif os_name==  "win" :  DIR2=  DIRANA +'/Scripts/'
   return DIR2



def conda_env_export(condaenv_list=[], folder_export="/") :
    '''
    #  /anaconda/envs/venv_name/bin/pip install
    Export Config
    '''
    os_name= sys.platform[:3]
    prefix= os_name
    prefix= prefix+ "-"+os.path.expanduser('~').split("\\")[-1].split("/")[-1]
    prefix= prefix+ "-"+      str(platform.release().replace(".", "-"))
    date0= arrow.now().format('YYYYMMDD')

    CONDA_PATH= conda_path_get()

    for condaenv in condaenv_list :
          file0 = folder_export +"/condaenv_" + prefix + "_" + condaenv + "_" + date0 +".yml"
          cmd= CONDA_PATH +'/conda env export --json -n '+condaenv + '  -f '  +file0

          print("condaenv: " + condaenv )
          print("     " + cmd);   sys.stdout.flush()
          a= os.popen( cmd ).read()
          print("     "+a);      sys.stdout.flush()




def conda_env_readyaml(condaexport_file=".yaml") :
   import pandas as pd
   ##### Read Files
   # file1= r'D:\_devs\Python01\project27\__config\condaenv\condaenv_win-asus1-7_root_20170929.yml'
   with open(condaexport_file, mode='r') as f1 :
     ff= f1.read()
     ff2= ff.split("\n")

   ff3=[ x.replace("-","").strip() for x in ff2  ]
   df= pd.DataFrame(ff3, columns=["fullname"])
   df["name"]= df["fullname"].apply(lambda x: x.split("=")[0].strip())

   def version(x):
     try :    return x.split("=")[1]
     except : return ''
   df["version"]= df["fullname"].apply(lambda x: version(x ) )

   def pyversion(x):
     try :    return x.split("=")[-1]
     except : return ''
   df["pyversion"]= df["fullname"].apply(lambda x: pyversion(x ) )


   def condapip(x):
     if x["fullname"] != x["name"] :
       if x["version"] == "" :  return 'pip'
       else :                   return "conda"
     else : return ""

   df["condapip"]= df.apply(lambda x:  condapip(x ), axis=1 )
   return df









def install_on_allcondaenv(package='configmy'):
   condaenv_all= conda_env_list()
   conda_install(package_list= [package] , condaenv_list=condaenv_all, install_tool="pip")

   '''
   pip install  --upgrade --no-deps --force-reinstall configmy
   
   '''







###############################################################################################
global IIX; IIX=0
def pprint(a): global IIX; IIX+= 1; print("\n--" + str(IIX) + ": " + str(a), flush=True)


def  ztest():
   # print( configmy.get(PACKAGE_PATH+"ztest/test_config.py", output= ["_CFG", "DIRCWD",]) )

   # configmy.set(path+"/ztest/test_config.py")
   CFG, DIRCWD=  configmy.get(method0=["os", "username"], output=["_CFG", "DIRCWD"] )
   print("Test print", CFG, DIRCWD )
   # print(configmy.get())

   # try:    print(os.environ[CONFIGMY_ROOT_FILE])
   # except: print("Empty CONFIGMY_ROOT_FILE")

   '''
   print(
#Test
from configmy import configmy
configmy.get("configmy/ztest/test_config.py", output= ["_CFG", "DIRCWD",])
CFG, DIRCWD= configmy.get("configmy/ztest/test_config.py", output= ["_CFG", "DIRCWD",])
configmy.set( "D:/_devs/Python01/project27/github/configmy/configmy/ztest/test_config.py") 
configmy.set("D:/_devs/Python01/project27/__config/CONFIGMY_ROOT_FILE.py")

#Usage
import configmy; CFG, DIRCWD= configmy.get(output= ["_CFG", "DIRCWD"]);  CFG, DIRCWD

#Run test
python configmy.py --action test


    )
   psrint("All Methods")
   print(__all__)
 

   print("Sample of Config File :")
   with open(PACKAGE_PATH+"ztest/test_config.py", "r") as f1 :
      ff1= f1.read()
   print(ff1)
   '''


if __name__ == "__main__"  :
  import argparse
  ppa = argparse.ArgumentParser()
  ppa.add_argument('--do',  type=str, default= ''  ,  help=" unit_test")
  ppa.add_argument('--run', type=str, default= ''  ,  help=" unit_test")
  arg = ppa.parse_args()


if __name__ == "__main__" and  arg.run != ''   :
    print("Running Task")
    globals()[arg.run]()   #Execute command


if __name__ == "__main__" and  arg.do == 'test' :
    pprint('### Unit Tests')
    import configmy
    configmy.ztest()


























##################################################################################################################
'''

Open terminal window and change directory to /project/
  python setup.py sdist bdist_wheel --universal
  twine upload dist/*



5) Build and upload subsequent updates to pypi
Update the change log and edit the version number in setup.py and package/__init__.py.

Open terminal window and change directory to /project/

  python setup.py sdist bdist_wheel --universal
  twine upload dist/*



'''


'''
import os, sys
CFG   = {'plat': sys.platform[:3]+"-"+os.path.expanduser('~').split("\\")[-1].split("/")[-1], "ver": sys.version_info.major}
DIRCWD= {'win-asus1': 'D:/_devs/Python01/project27/', 'win-unerry': 'G:/_devs/project27/' , 'lin-noel': '/home/noel/project27/', 'lin-ubuntu': '/home/ubuntu/project27/' }
try :
  # DIRCWD= os.environ["DIRCWD"];
  from attrdict import AttrDict as dict2
  DIRCWD= DIRCWD[ CFG["plat"]]; # print(DIRCWD, flush=True)
  os.chdir(DIRCWD); sys.path.append(DIRCWD + '/aapackage')
  f= open(DIRCWD+'/__config/config.py'); CFG= dict2(dict(CFG,  **eval(f.read()))); f.close()  #Load Config
  # print(CFG.github_login, flush=True)
except :  print("Project Root Directory unknown")

execfile( DIRCWD + '/aapackage/coke_functions.py')


method_blocked= {'__builtins__':None}
method_allowed

eval('[1, cpu_count()]', method_blocked, {})

import ast
ast.literal_eval(node_or_string)



>>>from os import cpu_count
>>>exposed_methods = {'cpu_count': cpu_count}
>>>eval('cpu_count()', {'__builtins__':None}, exposed_methods)
8
>>>eval('abs(cpu_count())', {'__builtins__':None}, exposed_methods)
TypeError: 'NoneType' object is not subscriptable


'''




'''





https://gist.github.com/gboeing/dcfaf5e13fad16fc500717a3a324ec17


Set up pypi
Create a file in the home directory called ~/.pypirc with contents:
[distutils]
index-servers = pypi



[pypi]
repository = https://pypi.python.org/pypi
username = YourPyPiUsername
password = YourPyPiPassword
Build, register, and upload to pypi



Open terminal window and change directory to /project/
Then run setup.py with sdist to build a source distribution and bdist_wheel to build a wheel (with --universal flag if your package is Python 2/3 universal). Then use twine to register it and upload to pypi.
python setup.py sdist bdist_wheel --universal
twine register dist/project-x.y.z.tar.gz
twine register dist/mypkg-0.1-py2.py3-none-any.whl
twine upload dist/*
Build and upload subsequent updates to pypi







Update the change log and edit the version number in setup.py and package/__init__.py.
Open terminal window and change directory to /project/ then run setup.py with sdist to build a source distribution and bdist_wheel to build a wheel (with --universal flag if your package is Python 2/3 universal). Remove old versions from /project/dist/ and then use twine to upload to pypi.
python setup.py sdist bdist_wheel --universal
twine upload dist/*



Release your code on GitHub
To tag your current commit as a released version, run:

git tag -a v0.1 -m "annotation for this release"
git push origin --tags






'''























