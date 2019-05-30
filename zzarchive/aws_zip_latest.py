import datetime
import os
import sys
import zipfile

import paramiko
import pysftp as sftp

##getting current time

now = datetime.datetime.now()

date_time=now.strftime("%Y-%m-%d_%H:%M")

## Setting Global variables...


global_file_zip_name= "filename_" + date_time
root_local_path=sys.argv[1]
dest_local_path=sys.argv[2]+global_file_zip_name+".zip"
temp_remote_location="/home/testuser/"+global_file_zip_name+".zip"
final_remote_location_unzip="/home/testuser/extracted_Data/"
remote_host="52.66.158.98"
remote_user="testuser"
remote_password="testpass"

print(root_local_path)
print(dest_local_path)
print(temp_remote_location)


##Zipping the Folder....

def zip(src, dst):
    zf = zipfile.ZipFile("%s" % (dst), "w", zipfile.ZIP_DEFLATED,allowZip64 = True)
    abs_src = os.path.abspath(src)
    for dirname, subdirs, files in os.walk(src):
        for filename in files:
            if filename.endswith('.py'):
               print("Skipping the Python Files.......")
            else:
                absname = os.path.abspath(os.path.join(dirname, filename))
                arcname = absname[len(abs_src) + 1:]
                print 'zipping %s as %s' % (os.path.join(dirname, filename),arcname)
                zf.write(absname, arcname)
    zf.close()


##uploading Folder on remote Server...


def upload_file(local_path,remote_path,host,user,password):
    try:
        s=sftp.Connection(host=host,username=user,password=password)
        print("connected to " + host + " with user : " + user)
        remotepath=remote_path
        localpath=local_path
        s.put(localpath,remotepath)
        print("Zip file uploaded on Location : " + remote_path)
        
    except Exception, e:
        print str(e)


### Unzip the folder on the remote side...


def unzip_folder_Remote(temp_remote_location,final_remote_location_unzip,hostname,username,password):
    nbytes = 4096
    port = 22
    command = "unzip -o {} -d {} ".format(temp_remote_location,final_remote_location_unzip)
    #print(command)
    client = paramiko.Transport((hostname, port))
    client.connect(username=username, password=password)

    stdout_data = []
    stderr_data = []
    session = client.open_channel(kind='session')
    session.exec_command(command)
    while True:
        if session.recv_ready():
            stdout_data.append(session.recv(nbytes))
        if session.recv_stderr_ready():
            stderr_data.append(session.recv_stderr(nbytes))
        if session.exit_status_ready():
           break
    
    

    print 'exit status: ', session.recv_exit_status()
    print ''.join(stdout_data)
    print ''.join(stderr_data)
    print("Unzip Completed on Location : " + final_remote_location_unzip)
    session.close()
    client.close()





zip(root_local_path,dest_local_path)

upload_file(dest_local_path,temp_remote_location,remote_host,remote_user,remote_password)
    
print("Unzipping the Folder from  " + temp_remote_location + " to " + final_remote_location_unzip)

unzip_folder_Remote(temp_remote_location,final_remote_location_unzip,remote_host,remote_user,remote_password)
