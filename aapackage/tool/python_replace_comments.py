A script to parse all sub-files 
and replace comments by nothing.



https://stackoverflow.com/questions/2319019/using-regex-to-remove-comments-from-source-files
  

  
  
def removeComments(string):
    string = re.sub(re.compile("/\*.*?\*/",re.DOTALL ) ,"" ,string) # remove all occurance streamed comments (/*COMMENT */) from string
    string = re.sub(re.compile("//.*?\n" ) ,"" ,string) # remove all occurance singleline comments (//COMMENT\n ) from string
    return string



def os_file_replace(source_file_path, pattern, substring):
    from tempfile import mkstemp
    from shutil import move
    from os import remove

    fh, target_file_path = mkstemp()
    with open(target_file_path, 'w') as target_file:
        with open(source_file_path, 'r') as source_file:
            for line in source_file:
                target_file.write(line.replace(pattern, substring))
    remove(source_file_path)
    move(target_file_path, source_file_path)


  
def os_file_listall(dir1, pattern="*.*", dirlevel=1, onlyfolder=0):
  '''
   # DIRCWD=r"D:\_devs\Python01\project"
   # aa= listallfile(DIRCWD, "*.*", 2)
   # aa[0][30];   aa[2][30]
  '''
  import fnmatch; import os; import numpy as np;  matches = []
  dir1 = dir1.rstrip(os.path.sep)
  num_sep = dir1.count(os.path.sep)

  if onlyfolder :
   for root, dirs, files in os.walk(dir1):
    num_sep_this = root.count(os.path.sep)
    if num_sep + dirlevel <= num_sep_this: del dirs[:]
    matches.append([]); matches.append([]); matches.append([]);   # Filename, DirName
    for dirs in fnmatch.filter(dirs, pattern):
      matches[0].append(os.path.splitext(dirs)[0])
      matches[1].append(os.path.splitext(root)[0])
      matches[2].append(os.path.join(root, dirs))
   return np.array(matches)

  for root, dirs, files in os.walk(dir1):
    num_sep_this = root.count(os.path.sep)
    if num_sep + dirlevel <= num_sep_this: del dirs[:]
    matches.append([]); matches.append([]); matches.append([]);   # Filename, DirName
    for files in fnmatch.filter(files, pattern):
      matches[0].append(os.path.splitext(files)[0])   
      matches[1].append(os.path.splitext(files)[1])  
      matches[2].append(os.path.join(root, files))   
  return np.array(matches)
