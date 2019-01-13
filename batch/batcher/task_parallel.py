#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
1/ Look ito todo/
2/ Analyze task_*.py header (text parsing) --->  Calculate Nb of Input
                                                          Split  (numpy split).
3/ Split the input data  (dataframe, csv) --> into several files
                                                         ( use pickle to save )
4/ Launch in paralell task_0002.py with different input
                                     ( using  file_slit_XXXX) using sub_process
5/When script is finished ---> Move script into  with name:
                                                    task_0001_20160202HHMMSS.py
6/ Write in log_file
-------------------------------------
Recomended tab length for good alignement: 4
Recomended tab type for python: space

TODO:
    - rethink folder architecture inside of the code
    - try and improve time (for infinite time with 2 times that are euqal
    - add psutils for windows

Input :
  folder_root :   /myfolderoftasks/
  start_time  :   2017-12-03 15:45
  timeout_time :  2025-12-03 15:45

Folder Strucutre:
  /folder_root/  task1_finished/      #finished
                                task1.py
                                file2.pkl
                                otherfile
                                /tmpfolder/
                                /output/

                 task2_failed/    #failed to launch
                                task1.py
                                file2.pkl
                                /tmpfolder/
                                /output/

                 tasks3/    #Not yet launched
                        task1.py
                        inputfile1.pkl


                 logfile.txt

task_launcher.py
  import task_parallel as taskpa
  taskpa.execute("/myfolder/", starttime="2017-03-05 11:45")

python  /my_script_folder/task_launcher.py  /folder_task/"   "2017-03-05 11:45"



'''
from __future__ import print_function
import subprocess
import time
import pickle
import arrow
import os
import sys
from attrdict import AttrDict
if sys.platform.startswith('win'):
    from ospipen import pipe_no_wait

    def pid_exists(pid):
        import ctypes
        kernel32 = ctypes.windll.kernel32
        SYNCHRONIZE = 0x100000
        process = kernel32.OpenProcess(SYNCHRONIZE, 0, pid)
        if process != 0:
            kernel32.CloseHandle(process)
            return True
        else:
            return False
else:
    from fcntl import fcntl, F_GETFL, F_SETFL
    from psutil import pid_exists
# import numpy as np       # By(kda) not used in module
# import pandas as pd      # By(kda) not used in module

# ------------------------------ Folder ROOT  ----------------------------
DIRCWD =  'D:/_devs/Python01/project27/'  if (os.path.expanduser('~').find('asus1') >-1  and sys.platform.find('win')>-1)\
    else  'G:/_devs/project27/'  if sys.platform.find('win')> -1 else  '/home/noel/project27/'\
    if  os.path.expanduser('~').find('noel') >-1 and  sys.platform.find('linux')> -1\
    else 'lin, virtualbox'
DIRPACKAGE = DIRCWD + '/github/parallel_python/'
sys.path.append(DIRPACKAGE + '/aapackage/')

PY_PATH = sys.executable

# -----------------------------System package-----------------------------

# Local package  from  DIRPACKAGE/aapackage/
# import processify  #Laucnh function into sub-process


def zdoc():
    aa = """
       Write all docs + sample here
    """
    print(aa)


def ztest():
    """ Write test and unit code here """
    print("a")


# ------------------------------Util functions ------------------------------
# code to save and load the splited data
def z_key_splitinto_dir_name(keyname):
    lkey = keyname.split
    if len(lkey) == 1:
        dir1 = ""
    else:
        dir1 = '/'.join(lkey[:-1])
        keyname = lkey[-1]
    return dir1, keyname


def py_save_object(obj, folder='/folder1/keyname', isabsolutpath=0):
    if isabsolutpath == 0 and folder.find('.pkl') == -1:        # Local Path
        dir0, keyname = z_key_splitinto_dir_name(folder)
        os.makedirs(DIRCWD + '/aaserialize/' + dir0)
        dir1 = DIRCWD + '/aaserialize/' + dir0 + '/' + keyname + '.pkl'
    else:
        dir1 = folder
    with open(dir1, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        return dir1


def py_load_object(folder='/folder1/keyname',
                   isabsolutpath=0, encoding1='utf-8'):
    '''def load_obj(name, encoding1='utf-8' ):
        with open('D:/_devs/Python01/aaserialize/' + name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding=encoding1)'''
    if isabsolutpath == 0 and folder.find('.pkl') == -1:
        dir0, keyname = z_key_splitinto_dir_name(folder)
        os.makedirs(DIRCWD + '/aaserialize/' + dir0)
        dir1 = DIRCWD + '/aaserialize/' + dir0 + '/' + keyname + '.pkl'
    else:
        dir1 = folder
    with open(dir1, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
class manage():
    '''This is the main class of this script
        IT controls the basic flow of the program
        and defines all of the global variables'''

    def __init__(self, folder_root, timeout=0, verbose=False):
        ''' This function has all of the variables, you have to provide
            it the folder architecture, the timeout and the DIRCWD variable.'''
        # all of the string variables
        self.folder_root = folder_root
        # this is a dummy variable for the stdout of the running script
        self.stdout = ""
        # this is the same but for stderr
        self.stderr = ""

        # all of the file variables
        root_dir = os.path.abspath(os.path.dirname(folder_root))
        self.log_filename = "%s%slogfile.txt" % (root_dir, os.sep)

        # all of the dictionaries #####
        # dictionnary of variables found in the header
        self.header_vars = dict()

        # python envornement variable
        # ! this var is not used but has to be here
        self.env_vars = dict()

        self.task_list = dict()

        ''' use  attrdict for dictionnary.

        task_list = {
           "task01" :
               {"task_status": "finished/failed/wait/running/",
                "task_startime": "2017-05-01 11:25"
                "task_script_list":
                  [ { "script_name": "file1.py",
                      "script_pid": 2525,
                      "script_input_list": [],
                      "script_status":
                      "script_header":

                     },

                     { "script_name" : "file2.py",
                       "script_pid"  : 2525,
                       "input_list"  :25
                     },
                  ]
           },
s           "task01" :
               {"task_status": "finished/failed/wait/running/",
                "task_startime": "2017-05-01 11:25"
                "task_script_list":
                  [ { "script_name": "file1.py",
                      "script_pid": 2525,
                      "script_input_list": [],
                      "script_status":
                      "script_header":
                     },

                     { "script_name" : "file2.py",
                       "script_pid"  : 2525,
                       "input_list"  :25
                     },
                  ]
           }
        }  '''
        # task_list['task_name'] = [ status, [task_executables],
        #                            [input_files], start_time, pid ]
        # status: 0, 1, 2 | to_run, failed, finished
        #                              // declared in self.retrieve_folders

        # all of the time variables #####
        self.timeout = arrow.get(timeout)
        self.startime = arrow.now()

        # all of the other variables #####
        self.DIRCWD = DIRCWD
        self.sub_proc = None
        self.verbose = verbose

    # ---------------------misc fucntions------------------------------------
    def run_check_timeout(self):
        return arrow.now() > self.timeout

    def script_error_exit(self, msg, return_code=-1):
        ''' print_error function print to stderr the print_error and its line
        for easy correction '''
        sys.stderr.write(msg)
        sys.exit(return_code)

    def log(self, string, time=arrow.now().format('YYYY-MM-DD HH:mm:ss')):
        ''' function to log the given string to the logfile.  '''
        with open(self.log_filename, "a") as log_fp:
            log_fp.write('[%s] %s\n' % (time, string))
            log_fp.flush()
        if self.verbose:
            print(string)

    #  ---------------------header functions---------------------------------
    def script_header_isok(self, fname):
        ''' chek if header has the correct variables such as,
            NSPLIT | ARRAY '''
        try:
            head = self.header_vars[fname]
            return 'NSPLIT' in head and 'ARRAY' in head
        except:
            # if the index is out of range, it means that the scripts had
            # an EOF error so the header is not ok/ not existant
            return False

    def script_header_scan(self, fname):
        ''' Scan's throught the header, get all of the important variables
            inside of a dictionary, and then run the code that is inside of
            the header. Note, watch out for the EOF print_error inside of the
            header if python recognises an EOF inside of that header
            it will be broken. '''
        in_header = False
        source = ""

        self.log("Scanning header for %s" % fname)
        # open the file
        with open(fname, "r") as f:
            for line in f.readlines():
                if "START_HEADER_SCRIPT" in line:
                    # figure out if you are at the start of the
                    # header or not, set a variable true if it is
                    in_header = True
                else:
                    if in_header:
                        if "END_HEADER_SCRIPT" in line:
                            # if found this string you are no more in the
                            # header so you will leave it.
                            break
                        else:
                            # it has to retrieve the source code to execute
                            # adding source to a var
                            source += line
        self.header_vars[fname] = dict()
        try:
            # executing source, watch out for the EOF print_error,
            exec(source, self.env_vars, self.header_vars[fname])
        except EOFError:
            # if python find EOF in source it will crash
            self.log("EOF print_error in the script: {0}".format(fname))
            self.log("please check the data in the source header")

    def run_headers(self):
        ''' run the individual headers and get all of the data that
            are in the headers
        '''
        if not len(self.task_list.keys()):
            self.log("No scripts to work with...")
            return None
        for fd_name in self.task_list.keys():
            for script in self.task_list[fd_name].task_script_list:
                fn = "%s%s%s%s" % (self.folder_root, fd_name, os.sep, script)
                self.script_header_scan(fn)
                if not self.script_header_isok(fn):
                    self.log("ERROR: script %s does not have a good header so"
                             " is beeing ignored" % script)
                    del self.task_list[fd_name].task_script_list[script]
                    # I though t you had to remove timeout and pid but they
                    # are added in runcommand so the above line is good
                else:
                    self.log("script %s has been processed, and the header"
                             " has been executed" % script)

    # --------------------folder functions------------------------------------
    def folder_create_structure(self, name, task):
        ''' setup the structure of the given folder '''
        if not os.path.isdir(name):
            self.log("%s name is not a directory, please"
                     "put scripts inside of a directory")
            return
        self.log('created the folder structure for: %s' % name)
        for cdir in ("tmpfolder", "output", "input"):
            path = "%s%s%s%s" % (name, os.sep, cdir, os.sep)
            if not os.path.exists(path):
                self.log('creating %s folder' % cdir)
                os.makedirs(path)
        filenames = map(lambda x: x, os.listdir(name))
        scripts = filter(lambda x: x.endswith('.py'), filenames)
        inputs = filter(lambda x: x.endswith('.plk'), filenames)
        for script_name in scripts:
            self.log('found script: %s' % script_name)
            self.task_list[task]['task_script_list'][script_name] =  \
                AttrDict({"script_pid": None,
                          "script_status": None,
                          "script_input_list": list(inputs),
                          "script_header": None,
                          "script_proc": None,
                          "script_starttime": None})
            txts = ("%s%s%s%s%s_std%s.txt" % (name, os.sep, "output",
                                              os.sep, script_name, "out"),
                    "%s%s%s%s%s_std%s.txt" % (name, os.sep, "output",
                                              os.sep, script_name, "err"))
            for txt in txts:
                open(txt, 'w').close()
        for input_name in inputs:
            self.log('found input file: %s' % input_name)

    def folder_retrieve_task(self):
        ''' this function checks if the folders exists and add the file in
            the todo array, it also creates, the individual task folders for
            the inputs and outputs of each individual tasks. It also sets and
            id inside of a map so that when you provide the name of the file
            you get its id. '''
        if not os.path.exists(self.folder_root):
            self.script_error_exit('folder root does not exist')

        for folder in os.listdir(self.folder_root):
            if 'task' in folder and os.path.isdir(self.folder_root+folder):
                self.log('found task folder: %s' % folder)
                # see above for the detail of each element
                self.task_list[folder] = AttrDict({"task_status": None,
                                                   "task_startime": None,
                                                   "task_script_list": dict()})
                # normal folders waiting to be executed
                status = 0
                if 'failed' in folder:
                    # failed folders those will get reexecuted to see
                    status = 1
                elif 'finished' in folder:
                    # finished folders those will get ignored
                    status = 2
                self.task_list[folder].task_status = status
                self.folder_create_structure(self.folder_root + folder, folder)

        if not len(self.task_list.keys()):
            self.log("not find any task to work with...")

    def task_move_finished(self, task_name):
        ''' Function to rename the folders that finished being processed.
        '''
        folder_name = self.folder_root + task_name
        if task_name.rfind(os.sep) != -1:
            new_folder_name = "%s%s%s" % (self.folder_root,
                                          task_name[:task_name.rfind(os.sep)],
                                          '_finished')
        else:
            new_folder_name = self.folder_root +  task_name + '_finished%s' % os.sep

        if self.task_list[task_name].task_status == 2 and 'finished' not in task_name:
            self.log("renamed %s to %s" % (folder_name, new_folder_name))
            os.rename(folder_name, new_folder_name)

    def task_move_failed(self, task_name):
        ''' function to muve failed tasks '''
        folder_name = self.folder_root + task_name
        if task_name.rfind(os.sep) != -1:
            renamed_folder_name = "%s%s%s" % (self.folder_root,
                                              task_name[:task_name.rfind(os.sep)],
                                              '_failed')
        else:
            renamed_folder_name = self.folder_root + task_name + '_failed'

        if self.task_list[task_name].task_status == 1 and\
           'failed' not in task_name:
            self.log('%s %s %s' % (task_name,
                                   "has failed so is beeing renamed to",
                                   renamed_folder_name))
            os.rename(folder_name, renamed_folder_name)

    # -------------------running scripts--------------------------------------
    def run_command(self, command):
        ''' Function ot run a script, provide it a script formated like this
            "/script/path/name args" it will then cut this into an array for
            the Popen function, and it will also start the start_process_time
            for the individual process '''
        try:
            cmd = [PY_PATH, os.path.basename(command)]
            cwd = os.path.dirname(command)
            origWD = os.getcwd()
            os.chdir(cwd)
            num = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=False)
            os.chdir(origWD)
            if sys.platform.startswith('win'):
                pipe_no_wait(num.stdout.fileno())
                pipe_no_wait(num.stderr.fileno())
            else:
                flags1 = fcntl(num.stdout, F_GETFL)
                fcntl(num.stdout, F_SETFL, flags1 | os.O_NONBLOCK)
                flags2 = fcntl(num.stderr, F_GETFL)
                fcntl(num.stderr, F_SETFL, flags2 | os.O_NONBLOCK)
            self.log("[*] {0}'s pid is : {1}".format(command, num.pid))
            return num
        except Exception as e:
            self.log(str(e))
            self.log("[*] {0} failed".format(command))
            # return none so that the output does not get monitored
            return None

    def run_all_scripts(self):
        ''' This will run every script of every task asynchronously '''
        if not len(self.task_list):
            self.log("No task, with scripts to execute....")
        for task_name in self.task_list.keys():
            for script in self.task_list[task_name].task_script_list:
                path = self.folder_root + task_name + os.sep + script
                self.log("running script: %s" % (path))
                proc = self.run_command(path)
                time.sleep(0.05)
                print(proc)
                task = self.task_list[task_name]
                dscript = task.task_script_list[script]
                if proc is not None:
                    dscript.script_pid = proc.pid
                    dscript.script_proc = proc
                    dscript.script_status = 0
                    dscript.script_starttime = arrow.now()
                else:
                    dscript.script_proc = None
                    dscript.script_starttime = None

    def script_is_running(self, task, script):
        ''' function to check if a script is running or not '''
        scrdscr = self.task_list[task].task_script_list[script]
        if scrdscr.script_status == 2:
            return False
        script_pid = scrdscr.script_pid
        if pid_exists(script_pid):
            return True
        else:
            return False

    # ------------------process data files------------------------------------
    def data_split(self, NSPLIT, ar, folder_name):
        ''' splitting function to split the given arrays into a number nsplit
            of parts it gets the split size then saves it into sub arrays that
            then get save to the acording folders '''
        # Needs to be round number because files can not be
        splitsize = int(len(ar) / NSPLIT)
        # floats its 1 file not 1.5 file
        output_varname = 'ar' + '_'
        for ii in range(0, NSPLIT):
            # split the different arrays in sub arrays and save them
            ar_i = ar[ii*splitsize:((ii+1)*(splitsize))]
            py_save_object(ar_i, "%s%s%s%s%s%s" % (folder_name, os.sep,
                                                   'input', os.sep,
                                                   output_varname, str(ii)), 1)

    def data_split_all(self):
        ''' Split all of the data, it lists throught the files in TODO_FILE,
            scans there header executes it,
            Checks the header and split the data found in the header '''
        if not len(self.task_list):
            self.log("No data inside of headers...")
        for iname in self.task_list.keys():
            for sc in self.task_list[iname].task_script_list:
                self.log("spliting header data for %s" % sc)
                index = self.folder_root + iname + os.sep + sc
                arr = self.header_vars[index]['ARRAY']         # set the array
                nsplit = self.header_vars[index]['NSPLIT']     # set the nsplit
                self.data_split(nsplit, arr, self.folder_root + iname)

    # -----------------monitoring---------------------------------------------
    # TODO:dont forget to monitor timeout
    def task_isok(self, task):
        ''' check if the pid names and startime are all present if one is
            missing there is an error and the task will be removed '''
        return len(self.task_list[task].task_script_list)

    def task_isfinished(self, task):
        dtask = self.task_list[task]
        if dtask.task_status == 0:
            srcipts_states = map(
                lambda x: dtask.task_script_list[x].script_status,
                dtask.task_script_list)
            if 0 not in srcipts_states:
                self.log("task %s is finished" % task)
                dtask.task_status = 2
                self.task_move_finished(task)

    def task_all_running(self):
        states = map(lambda x: self.task_list[x].task_status,
                     self.task_list)
        if 0 in states:
            return True
        else:
            self.log("All task are finished!")
            return False

    def monitor_task(self, task):
        ''' function that monitors a task '''
        if not self.task_isok(task=task):
            del self.task_list[task]
            self.log("task: %s is not complete there is a problem in"
                     " the number of scripts / startime / pid" % task)
            return 1
        for script in self.task_list[task].task_script_list.keys():
            if self.script_is_running(task, script):
                pp = self.task_list[task].task_script_list[script].script_pid
                err = self.monitor_script(script=script, task=task, pid=pp)
                if err == -1:
                    self.log("script : %s finished" % script)
                    dtask = self.task_list[task]
                    dtask.task_script_list[script].script_status = 2
                    time.sleep(0.3)

    def monitor_script(self, script, task, pid):
        #print("MS")
        ''' function that monitors a single script '''
        proc = self.task_list[task].task_script_list[script].script_proc
        stdout_fname = self.folder_root+task + os.sep +\
            'output' + os.sep + '%s_stdout.txt' % script
        stderr_fname = self.folder_root+task + os.sep +\
            'output' + os.sep + '%s_stderr.txt' % script
        if sys.platform.startswith('win'):
            pipe_no_wait(proc.stdout.fileno())
            pipe_no_wait(proc.stderr.fileno())
        err_data = ''
        out_data = ''
        try:
            if sys.platform.startswith('win'):
                out_data = proc.stdout.read()
            else:
                out_data = os.read(proc.stdout.fileno(), 1024)
        except OSError:
            out_data = ''
        except IOError:
            out_data = ''
        try:
            if sys.platform.startswith('win'):
                err_data = proc.stderr.read()
            else:
                err_data = os.read(proc.stderr.fileno(), 1024)
        except OSError:
            err_data = ''
        except IOError:
            out_data = ''
        if out_data:
            self.log("got stdout data from %s" % script)
            if sys.version_info >= (3, 0):
                stdout_fp = open(stdout_fname, 'ab')
                stdout_fp.write(bytes(out_data))
            else:
                stdout_fp = open(stdout_fname, 'a')
                stdout_fp.write(out_data)
            stdout_fp.close()
        if err_data:
            self.log("got stderr data from %s" % script)
            if sys.version_info >= (3, 0):
                stderr_fp = open(stderr_fname, 'ab')
                stderr_fp.write(bytes(err_data))
            else:
                stderr_fp = open(stderr_fname, 'a')
                stderr_fp.write(err_data)
            stderr_fp.close()
        if proc.poll() is not None:
            return -1
        else:
            return 1

    def monitor_all_tasks(self):
        ''' function to monitor all tasks and check
            if they have failed or timedout '''
        if not len(self.task_list):
            self.log("No task to monitor...")
        while self.task_all_running() and not self.run_check_timeout():
            for task in self.task_list.keys():
                self.monitor_task(task=task)
                self.task_isfinished(task)
            time.sleep(0.05)
        if self.run_check_timeout():
            for task in self.task_list:
                if self.task_list[task].task_status != 2:
                    self.task_list[task].task_status = 1
                    self.task_move_failed(task)
            self.log("Script Exection Timed Out!")

    # -----------------main function-----------------------------------------
    def run_main(self):
        ''' Main function that check all of the folders splits the data,
            then goes through the script to execute then executes then with
            task_run_scripts, it then waits for them to finish. '''
        self.log("--Starting up--")
        # setup the folder structure and retrieve all of the tasks
        self.folder_retrieve_task()
        # run the scripts that where found and get the data from the headers
        # self.run_headers()
        # split the data founs in the headers and save them as input
        # self.split_data_all()
        # run each individual scripts and save usefull data about them
        self.run_all_scripts()
        self.log("--going into monitor loop--")
        self.monitor_all_tasks()


def execute(folder_root, startime="2010-03-01 11:25",
            timeout="2020-01-01 11:00"):
    ''' Launch a independant sub-process where the python code below
        is executed at time 2017 03 01 11:25 this is a function wrapper
        for execute_sub '''
    timeout = arrow.get(timeout)
    startime = arrow.get(startime)
    m = manage(timeout=timeout, folder_root=folder_root, verbose=True)
    while arrow.now() < startime:
        time.sleep(30)
    m.run_main()

    '''
    #@processify.processify.processify
    #def execute_sub(folder_root, starttime, timeout) :
       m = manage(timeout=timeout , folder_root=folder_root, verbose=True)
       t0= arrow.get(starttime)
       while arrow.now() < t0 :
         time.sleep(30)
       m.main()
    '''
    # Execute in separate sub-process
    # execute_sub(folder_root, starttime, timeout)

# folder_setup
if __name__ == "__main__":
    # execute('testing/', starttime=arrow.now().format('YYYY-MM-DD HH:mm:ss'),
    #         timeout='2000-01-01 00:00:00')
    # sm = manage(timeout="2020-01-01 11:00" ,
    #             folder_root='testing/', verbose=True)
    import task_parallel as tpar

    # todo_folder= "c:/My_todoFolder/"
    todo_folder = "testing" + os.sep
    tpar.execute(todo_folder, startime="2017-01-02-15:25")



