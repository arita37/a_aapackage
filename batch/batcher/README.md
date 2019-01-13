To do 


1) Daemon mode to task_parallell 

    1) Launch  on CLI like     python task_parallel.py  --folder  myFolder   --freq 60   
    2) task_parallel will monitor (every xx seconds) if there is a folder to execute.
    3)  If there is a folder to execute (!= finished), execute it 
        and monitor with task_parallel.
        
        Tasks are launched in no blocking subprocess
        
    4)  This is run as daemon and never stops.
    
        As soon as we add sub-folder task to myfolder, it keeps executing.
        
    
    
 ```
 https://blender.stackexchange.com/questions/45731/how-to-run-an-external-command-showing-its-progress-without-locking-blender-e
 
 https://stackoverflow.com/questions/16071866/non-blocking-subprocess-call
 
 
 https://stackoverflow.com/questions/375427/non-blocking-read-on-a-subprocess-pipe-in-python
 
 ```
 
 
    
        
2)  Write python code to remove all comments + multi=line comments in a folder, sub-folder.



```
### In file replacement
aa = util.os_file_listall(path1, pattern='*', dirlevel=100, onlyfolder=0)
for x in aa[2] :  
  if  x[x.rfind('.')+1:] in [ 'py', 'txt' , 'ini' ,'yaml'  ] :  
    with open(x, 'r') as file :
        filedata = file.read()

    filedata = regex.sub(wout, filedata)

    # Write the file out again
    with open(x, 'w') as file:
       file.write(filedata)
       
       
 
def os_file_listall(dir1, pattern="*.*", dirlevel=1, onlyfolder=0):
  import fnmatch; import os; import numpy as np;  matches = []
  dir1 = dir1.rstrip(os.path.sep)
  num_sep = dir1.count(os.path.sep)

  if onlyfolder :
   for root, dirs, files in os.walk(dir1):
    num_sep_this = root.count(os.path.sep)
    if num_sep + dirlevel <= num_sep_this: del dirs[:]
    matches.append([]); matches.append([]); matches.append([]);   
    for dirs in fnmatch.filter(dirs, pattern):
      matches[0].append(os.path.splitext(dirs)[0])
      matches[1].append(os.path.splitext(root)[0])
      matches[2].append(os.path.join(root, dirs))
   return np.array(matches)

  for root, dirs, files in os.walk(dir1):
    num_sep_this = root.count(os.path.sep)
    if num_sep + dirlevel <= num_sep_this: del dirs[:]
    matches.append([]); matches.append([]); matches.append([]);   Filename, DirName
    for files in fnmatch.filter(files, pattern):
      matches[0].append(os.path.splitext(files)[0])   
      matches[1].append(os.path.splitext(files)[1])  
      matches[2].append(os.path.join(root, files))   
  return np.array(matches)
```
      
       


