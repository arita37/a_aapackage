import sys, os



DIRCWD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(DIRCWD); sys.path.append(DIRCWD + '/aapackage'

from aapackage import util, util_aws





def  batch_run(
      file_script      = "folder/myscript_optim.py",
      hyperparam_file  = "hyperparam_list.csv",
      out_folder       = "/myresult/batch1/",
      schedule         = "11/02/2019 12:30",
      n_pid_max = 30,
  )  :  


      n_pid_max = ps_util.ncpu() - 1
      
      for ii, row of df_hyperparam_list.iterrows() , 
          pid = subprocess  row["file_script"]  ii  "hyperparam_list.csv"  out_folder   ###All subprocess are independdant !!!!
          server_info = "_".join( util.os_server_info())
          logs(arrow.timestamp, pid, server_info,  row["file_script"]  ii  hyperparam_file )
          wait(waitsecs)

  batch_monitor_pid(auto_close=True)  ### Auto close the EC2 instance when no PID is running.






def batch_run_ssh(from_folder, to_folder, bash_cmd, ipadress, login, passwd)
  """
    Start EC2 instance and get Ip adress and instnance number.
    Transfer by SSH the folder
    run the script

  """



def batch_monitor_pid() :
  """
    Details

     Use psutil to retrieve all PID with   python .
     and update the list with info on PID

     if n python process == 1 and batch_monitor  --> Close the 
       t0_flag = arrow.time

     if n python process == 1 and batch_monitor and t0 - to_flag > 10 mins  
        close EC2 instance


  """









