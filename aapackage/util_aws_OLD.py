# -*- coding: utf-8 -*-
# ---------AWS utilities--------------------------------------------------------
"""
os.ENVIRON["BOTO_CONFIG"] =  C:/Users/asus1\.aws1.credentials
# AWS_KEY_PEM= 'ec2_instance_test01.pem'
# AWS_SECRET,AWS_KEY= boto.config
exec (open("D:/_devs/keypair/aws_access.py").read(), globals())
"""
from __future__ import division
from __future__ import print_function
from future import standard_library

standard_library.install_aliases()

###############################################################################
import os, sys

import boto
from boto.ec2.blockdevicemapping import BlockDeviceMapping, EBSBlockDeviceType
import boto.ec2

from time import sleep
from attrdict import AttrDict as dict2
from pprint import pprint
import csv

# from aapackage import util


### Local  #####################################################################
#  DIRCWD  =  r"D:/_devs/Python01/aws/aapackage/"
# DIRCWD  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.chdir(DIRCWD); sys.path.append(DIRCWD + '/aapackage')
# __path__= DIRCWD +'/aapackage/'

AWS_ACCESS_LOCAL = "D:/_devs/keypair/aws_access.py"
AWS_KEY_PEM = "D:/_devs/keypair/oregon/aws_ec2_oregon.pem"


### Remote   ###################################################################
EC2CWD = "/home/ubuntu/notebook/"


################################################################################
#### Global ####################################################################
global EC2_CONN
EC2_CONN = None

EC2_FILTERS = ("id", "ip_address")
EC2_ATTRIBUTES = (
    "id",
    "instance_type",
    "state",
    "public_dns_name",
    "private_dns_name",
    "state_code",
    "previous_state",
    "previous_state_code",
    "key_name",
    "launch_time",
    "image_id",
    "placement",
    "placement_group",
    "placement_tenancy",
    "kernel",
    "ramdisk",
    "architecture",
    "hypervisor",
    "virtualization_type",
    "product_codes",
    "ami_launch_index",
    "monitored",
    "monitoring_state",
    "spot_instance_request_id",
    "subnet_id",
    "vpc_id",
    "private_ip_address",
    "ip_address",
    "platform",
    "root_device_name",
    "root_device_type",
    "state_reason",
    "interfaces",
    "ebs_optimized",
    "instance_profile",
)


if sys.platform.find("win") > -1:
    pass


################################################################################


###### Connection ###############################################################
def aws_accesskey_get(access="", key="", mode=""):
    """
    Return a tuple of AWS credentials (access key id and secret access key)
    try:
        #cfg = INIConfig(open(boto_config_path(account)))
        #return ( cfg.Credentials.aws_access_key_id, cfg.Credentials.aws_secret_access_key )
        return 0
    except Exception:
        raise
   """
    if access != "" and key != "":
        return access, key
    # access, key= config.get('IAM', 'access'), config.get('IAM', 'secret')
    # access, key= AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

    ### Boto Config
    access, key = (
        boto.config.get("Credentials", "aws_access_key_id"),
        boto.config.get("Credentials", "aws_secret_access_key"),
    )
    if access is not None and key is not None:
        print(access, key)
        return access, key

    ### Manual Config
    dd = {}
    exec(open(AWS_ACCESS_LOCAL).read(), dd)
    access, key = dd["AWS_ACCESS_KEY_ID"], dd["AWS_SECRET_ACCESS_KEY"]
    return access, key


def aws_conn_create_windows(AWS_REGION="us-west-2"):
    if sys.platform.find("win") > -1:
        # AWS_ACCESS_KEY_ID = "ACCES_KEY_ID"
        # AWS_SECRET_ACCESS_KEY = "SECRET_KEY"
        dd = {}
        exec(open(AWS_ACCESS_LOCAL).read(), dd)
        access, key = dd["AWS_ACCESS_KEY_ID"], dd["AWS_SECRET_ACCESS_KEY"]

        EC2_CONN2 = boto.ec2.connect_to_region(
            AWS_REGION, aws_access_key_id=access, aws_secret_access_key=key
        )
        print(EC2_CONN2)
        return EC2_CONN2


def aws_conn_create(region="ap-northeast-2", access="", key=""):
    from boto.ec2.connection import EC2Connection

    if access == "" and key == "":
        access, key = aws_accesskey_get()
    conn = EC2Connection(access, key)
    regions = aws_conn_getallregions(conn)
    for r in regions:
        if r.name == region:
            conn = EC2Connection(access, key, region=r)
            return conn
    print("Region not Find")
    return None


def aws_conn_getallregions(conn=None):
    return conn.get_all_regions()


def aws_conn_getinfo(conn):
    print(conn.region.name)


#### EC2 #######################################################################
def aws_ec2_ami_create(con, ip_address="", ami_name=""):
    """    Create AMI for the instance of ipadress """
    instance = con.get_all_instances(filters={"ip_address": ip_address})[0].instances[0]
    if instance:
        ami_id = instance.create_image(ami_name)
        print("AMI ID %s" % (ami_id))


def aws_ec2_get_instanceid(con, filters={"ip_address": ""}):
    instance = con.get_all_instances(filters=filters)[0].instances[0]
    if instance:
        return instance.id


def aws_ec2_allocate_elastic_ip(con, instance_id="", elastic_ip="", region="ap-northeast-2"):
    # con=  aws_conn_create(region=region)
    if elastic_ip == "":
        eip = con.allocate_address()
        con.associate_address(instance_id=instance_id, public_ip=eip.public_ip)
        print("Elastic assigned Public IP: " + eip.public_ip, ",Instance_ID:", instance_id)
        return eip.public_ip
    else:
        con.associate_address(instance_id=instance_id, public_ip=elastic_ip)
        print("Elastic assigned Public IP: " + elastic_ip, ",Instance_ID:", instance_id)


def aws_ec2_allocate_eip(
    instance_id,
    connection=None,
    eip_allocation_id=None,
    eip_public_ip=None,
    allow_reassociation=False,
):
    """Assign an Elastic IP to an instance. Works with either by specifying the
    connection: EC2Connection object
    instance_id: Desired EC2 instance's ID
    eip_allocation_id: ID of Elastic IP to assign (Required if no public_ip)
    eip_public_ip: Elastic IP to assign (Required if no allocation_id)
    allow_reassociation: Option to turn off reassociation (check caveats below)
    """
    if eip_public_ip:
        connection.associate_address(
            instance_id=instance_id,
            public_ip=eip_public_ip,
            allow_reassociation=allow_reassociation,
        )
        return

    if eip_allocation_id:
        connection.associate_address(
            instance_id=instance_id,
            allocation_id=eip_allocation_id,
            allow_reassociation=allow_reassociation,
        )
        return

    raise ValueError("eip_public_ip and eip_allocation_id cannot be both None!")


def aws_ec2_spot_start(
    con,
    region,
    key_name="ecsInstanceRole",
    inst_type="cx2.2",
    ami_id="",
    pricemax=0.15,
    elastic_ip="",
    pars={"security_group": [""], "disk_size": 25, "disk_type": "ssd", "volume_type": "gp2"},
):
    """
   :param con:   Connector to Boto
   :param region: AWS region (us-east-1,..) 
   :param key_name: AWS  SSH Key Name  (in EC2 webspage )
   :param security_group: AWS security group id
   :param inst_type:  AWS EC2 instance type (t1.micro, m1.small ...)
   :param ami_id:  AWS AMI ID
   :param pars: Disk Size, Volume type (General Purpose SSD - gp2, Magnetic etc)
   :param pricemax: minmum spot instance bid price
    """
    pars = dict2(pars)  # Dict to Attribut Dict
    print("starting EC2 Spot Instance")

    try:
        block_map = BlockDeviceMapping()
        device = EBSBlockDeviceType()
        device.size = int(pars.disk_size)  # size in GB
        device.volume_type = pars.volume_type  # [ standard | gp2 | io1 ]
        device.delete_on_termination = False
        block_map["/dev/xvda"] = device
        print("created a block device")

        req = con.request_spot_instances(
            price=pricemax,
            image_id=ami_id,
            instance_type=inst_type,
            key_name=key_name,
            security_groups=pars.security_group,
            block_device_map=block_map,
        )
        print("Spot instance request created. status  : %s" % (req[0].status))
        print("Waiting for spot instance provisioning")
        while True:
            current_req = con.get_all_spot_instance_requests([req[0].id])[0]
            if current_req.state == "active":
                print("Spot instance provisioning successful.")
                instance = con.get_all_instances([current_req.instance_id])[0].instances[0]

                aws_ec2_allocate_elastic_ip(con, current_req.instance_id, elastic_ip, region)

                # Print Instance details : ID, IP adress
                for x in ["ip_address", "id"]:
                    print(x, getattr(instance, x))
                return instance
            print(".", end="")
            sleep(15)
    except Exception as e:
        print("Error : %s " % (str(e)))
        # sys.exit(1)


def aws_ec2_spot_stop(con, ipadress="", instance_id=""):
    """
   :param con: connector 
   :param ipadress:   of the instance  to Identify the instance.
   :param instance_id:  OR use instance ID....
   :return: 
   """
    if instance_id == "":
        instance_id = aws_ec2_get_instanceid(con, ipadress)  # Get ID from IP Adress

    try:
        print("Terminating Spot Instance : %s" % (str(instance_id)))
        con.terminate_instances(instance_ids=[instance_id])
        print("Successful ")
    except Exception as e:
        print("Error : Failed to terminate. %s" % (str(e)))


################################################################################
def aws_ec2_res_start(
    con,
    region,
    key_name,
    ami_id,
    inst_type="cx2.2",
    min_count=1,
    max_count=1,
    pars={"security_group": [""], "disk_size": 25, "disk_type": "ssd", "volume_type": "gp2"},
):
    """  
        normal instance start
        :param con:   Connector to Boto
        :param region: AWS region (us-east-1,..) 
        :param key_name: AWS  SSH Key Name
        :param security_group: AWS security group id
        :param inst_type:  AWS EC2 instance type (t1.micro, m1.small ...)
        :param ami_id:  AWS AMI ID
        :param min_count: Minumum number of instances
        :param max_count : Maximum number of instances
        :param pars: Disk Size, Volume type (General Purpose SSD - gp2, Magnetic etc)
        :return 
    """
    pars = dict2(pars)  # Dict to Attribut Dict
    try:
        block_map = BlockDeviceMapping()
        device = EBSBlockDeviceType()
        device.size = int(pars.disk_size)  # size in GB
        device.volume_type = pars.volume_type  # [ standard | gp2 | io1 ]
        device.delete_on_termination = False
        block_map["/dev/xvda"] = device
        print("created a block device")

        req = con.run_instances(
            image_id=ami_id,
            min_count=min_count,
            max_count=max_count,
            instance_type=inst_type,
            key_name=key_name,
            security_groups=pars.security_group,
            block_device_map=block_map,
        )
        instance_id = req.instances[0].id
        print("EC2 instance has been created. Instance ID : %s" % (instance_id))
        print("Waiting for EC2 instance provisioning")
        while True:
            print(".", end="")
            current_req = con.get_all_instances(instance_id)
            if current_req[0].instances[0].state.lower() == "running":
                print("EC2 instance provisioning successful and the instance is running.")
                aws_ec2_printinfo(current_req[0].instances[0])
                return current_req[0].instances[0]
            print("."),
            sleep(30)
    except Exception as e:
        print("Error : %s " % (str(e)))
        # sys.exit(1)


def aws_ec2_res_stop(con, ipadress="", instance_id=""):
    """
   :param con: connector 
   :param ipadress:     Of the instance  to Identify the instance.
   :param instance_id:  OR use instance ID....
   """
    if instance_id == "":
        instance_id = aws_ec2_get_instanceid(con, ipadress)

    try:
        print("Stopping EC2 Instance : %s" % (str(instance_id)))
        req = con.stop_instances(instance_ids=[instance_id])
        print("EC2 instance has been stopped successfully. %s" % (req))
    except Exception as e:
        print("Error : Failed to stop EC2 instance. %s" % (str(e)))


def aws_ec2_printinfo(instance=None, ipadress="", instance_id=""):
    """   Idenfiy instnance of
   :param instance:
     ipadress
   :param instance_id:
   :return: return info on the instance : ip, ip_adress,
   """
    if ipadress != "":
        print("")

    if instance_id != "":
        pass


def aws_ec2_get_folder(ipadress, fromfolder1, tofolder1):
    pass
    # Copy Folder from remote to  EC2 folder
    # please use class aws_ec2_ssh(object):


#### S3 ########################################################################
"""
def aws_s3_file_putons3(fromfile, tobucket_path='bucket/folder1/folder2', AWS_KEY='', AWS_SECRET='' ) :
  from boto.s3.connection import S3Connection
  access, key= aws_accesskey_get()
  conn = S3Connection(access, key )

  tobucket, path1= aws_s3_url_split(tobucket_path)
  filename=        os_file_getname(fromfile)
  bucket = conn.get_bucket(tobucket)
  dest = bucket.new_key(path1+'/'+filename)
  dest.set_contents_from_file(fromfile)
"""


"""
def aws_s3_file_getfroms3(s3file='bucket/folder/perl_poetry.pdf', tofilename='/home/larry/myfile.pdf', AWS_KEY='', AWS_SECRET='' ):
  from boto.s3.connection import S3Connection
  #if access== '' and key=='' : access, key= aws_accesskey_get()
  access, key= aws_accesskey_get()
  conn = S3Connection(access, key )
  bucket, path1= aws_s3_url_split(s3file)
  bucket = conn.get_bucket(bucket)
  key = bucket.get_key(path1)
  key.get_contents_to_filename(tofilename)
"""


def aws_s3_url_split(url):
    """Split into Bucket, url """
    url1 = url.split("/")
    return url1[0], "/".join(url1[1:])


def aws_s3_getbucketconn(s3dir):
    import boto.s3

    bucket_name, todir = aws_s3_url_split(s3dir)
    ACCESS, SECRET = aws_accesskey_get()
    conn = boto.connect_s3(ACCESS, SECRET)
    bucket = conn.get_bucket(bucket_name)  # , location=boto.s3.connection.Location.DEFAULT)
    return bucket


def aws_s3_put(fromdir_file="dir/file.zip", todir="bucket/folder1/folder2"):
    """ Copy File or Folder to S3 
         "aws s3 cp s3://s3-bucket/scripts/HelloWorld.sh /home/ec2-user/HelloWorld.sh",
        "chmod 700 /home/ec2-user/HelloWorld.sh",
        "/home/ec2-user/HelloWorld.sh"
 
 
 """
    import boto.s3

    bucket = aws_s3_getbucketconn(todir)
    bucket_name, todir = aws_s3_url_split(todir)

    MAX_SIZE = 20 * 1000 * 1000
    PART_SIZE = 6 * 1000 * 1000

    if fromdir_file.find(".") > -1:  # Case of Single File
        filename = util.os_file_getname(fromdir_file)
        fromdir_file = util.os_file_getpath(fromdir_file) + "/"
        uploadFileNames = [filename]
    else:
        uploadFileNames = []
        for (fromdir_file, dirname, filename) in os.walk(fromdir_file):
            uploadFileNames.extend(filename)
            break

    def percent_cb(complete, total):
        sys.stdout.write(".")
        sys.stdout.flush()

    for filename in uploadFileNames:
        sourcepath = os.path.join(fromdir_file + filename)
        destpath = os.path.join(todir, filename)
        print("Uploading %s to Amazon S3 bucket %s" % (sourcepath, bucket_name))

        filesize = os.path.getsize(sourcepath)
        if filesize > MAX_SIZE:
            print("multipart upload")
            mp = bucket.initiate_multipart_upload(destpath)
            fp = open(sourcepath, "rb")
            fp_num = 0
            while fp.tell() < filesize:
                fp_num += 1
                print("uploading part %i" % fp_num)
                mp.upload_part_from_file(fp, fp_num, cb=percent_cb, num_cb=10, size=PART_SIZE)
            mp.complete_upload()
        else:
            print("singlepart upload: " + fromdir_file + " TO " + todir)
            k = boto.s3.key.Key(bucket)
            k.key = destpath
            k.set_contents_from_filename(sourcepath, cb=percent_cb, num_cb=10)


def aws_s3_get(froms3dir="task01/", todir="", bucket_name="zdisk"):
    """ Get from S3 file/folder  """
    bucket_name, dirs3 = aws_s3_url_split(froms3dir)
    bucket = aws_s3_getbucketconn(froms3dir)
    bucket_list = bucket.list(prefix=dirs3)  #  /DIRCWD/dir2/dir3

    for l in bucket_list:
        key1 = str(l.key)
        file1, path2 = util.os_file_getname(key1), util.os_file_getpath(key1)
        path1 = os.path.relpath(path2, dirs3).replace(".", "")  # Remove prefix path of S3 to mach
        d = todir + "/" + path1
        # print d, path2
        # sys.exit(0)
        if not os.path.exists(d):
            os.makedirs(d)
        try:
            l.get_contents_to_filename(d + "/" + file1)
        except OSError:
            pass


def aws_s3_folder_printtall(bucket_name="zdisk"):
    ACCESS, SECRET = aws_accesskey_get()
    conn = boto.connect_s3(ACCESS, SECRET)
    bucket = conn.create_bucket(bucket_name, location=boto.s3.connection.Location.DEFAULT)
    folders = bucket.list("", "/")
    for folder in folders:
        print(folder.name)
    return folders


def aws_s3_file_read(bucket1, filepath, isbinary=1):
    """ s3_client = boto3.client('s3')
    #Download private key file from secure S3 bucket
  s3_client.download_file('s3-key-bucket','keys/keyname.pem', '/tmp/keyname.pem')
  """
    from boto.s3.connection import S3Connection

    conn = S3Connection(aws_accesskey_get())
    response = conn.get_object(Bucket=bucket1, Key=filepath)
    file1 = response["Body"]
    return file1


####### SSH ####################################################################
class aws_ec2_ssh(object):
    """
    ssh= aws_ec2_ssh(host)
    print ssh.command('ls ')
    ssh.put_all( DIRCWD +'linux/batch/task/elvis_prod_20161220', EC2CWD + '/linux/batch/task' )
    ssh.get_all(  EC2CWD + '/linux/batch/task',  DIRCWD +'/zdisks3/fromec2' )

    # Detects DSA or RSA from key_file, either as a string filename or a file object.  Password auth is possible, but I will judge you for
    # ssh=SSHSession('targetserver.com','root',key_file=open('mykey.pem','r'))
    # ssh=SSHSession('targetserver.com','root',key_file='/home/me/mykey.pem')
    # ssh=SSHSession('targetserver.com','root','mypassword')
    # ssh.put('filename','/remote/file/destination/path')
    # ssh.put_all('/path/to/local/source/dir','/path/to/remote/destination')
    # ssh.get_all('/path/to/remote/source/dir','/path/to/local/destination')
    # ssh.command('echo "Command to execute"')
    """

    def __init__(self, hostname, username="ubuntu", key_file=None, password=None):
        import paramiko, socket

        #  Accepts a file-like object (anything with a readlines() function)
        #  in either dss_key or rsa_key with a private key.  Since I don't
        #  ever intend to leave a server open to a password auth.
        self.host = hostname
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((hostname, 22))
        self.t = paramiko.Transport(self.sock)
        self.t.start_client()
        # keys = paramiko.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
        # key = self.t.get_remote_server_key()
        # supposed to check for key in keys, but I don't much care right now to find the right notation

        key_file = AWS_KEY_PEM if key_file is None else key_file
        pkey = paramiko.RSAKey.from_private_key_file(key_file)

        """
        key_file = AWS_KEY_PEM  if key_file is None else key_file
        
        if key_file is not None:
            if isinstance(key_file,str):
                key_file=open(key_file, mode='r')
            key_head=key_file.readline()
            key_file.seek(0)
            if 'DSA' in key_head:
                keytype=paramiko.DSSKey
            elif 'RSA' in key_head:
                keytype=paramiko.RSAKey
            else:
                raise Exception("Can't identify key type")
            pkey=keytype.from_private_key(key_file)
            self.t.auth_publickey(username, pkey)
        else:
            if password is not None:
                self.t.auth_password(username,password,fallback=False)
            else: raise Exception('Must supply either key_file or password')
        """

        self.t.auth_publickey(username, pkey)
        self.sftp = paramiko.SFTPClient.from_transport(self.t)
        print(self.command("ls "))

    def command(self, cmd):
        #  Breaks the command by lines, sends and receives
        #  each line and its output separately
        #  Returns the server response text as a string

        chan = self.t.open_session()
        chan.get_pty()
        chan.invoke_shell()
        chan.settimeout(20.0)
        ret = ""
        try:
            ret += chan.recv(1024)
        except:
            chan.send("\n")
            ret += chan.recv(1024)
        for line in cmd.split("\n"):
            chan.send(line.strip() + "\n")
            ret += chan.recv(1024)
        return ret

    def cmd(self, cmdss):
        ss = self.command(cmdss)
        print(ss)

    def put(self, localfile, remotefile):
        #  Copy localfile to remotefile, overwriting or creating as needed.
        self.sftp.put(localfile, remotefile)

    def put_all(self, localpath, remotepath):
        #  recursively upload a full directory
        # localpath= localpath[:-1] if localpath[-1]=='/' else localpath
        # remotepath= remotepath[:-1] if remotepath[-1]=='/' else remotepath

        os.chdir(os.path.split(localpath)[0])
        parent = os.path.split(localpath)[1]
        print(parent)
        for walker in os.walk(parent):
            try:
                self.sftp.mkdir(os.path.join(remotepath, walker[0]).replace("\\", "/"))
            except:
                pass
            for file in walker[2]:
                print(
                    os.path.join(walker[0], file).replace("\\", "/").replace("\\", "/"),
                    os.path.join(remotepath, walker[0], file).replace("\\", "/"),
                )
                self.put(
                    os.path.join(walker[0], file).replace("\\", "/"),
                    os.path.join(remotepath, walker[0], file).replace("\\", "/"),
                )

    def get(self, remotefile, localfile):
        #  Copy remotefile to localfile, overwriting or creating as needed.
        self.sftp.get(remotefile, localfile)

    def sftp_walk(self, remotepath):
        from stat import S_ISDIR

        # Kindof a stripped down  version of os.walk, implemented for
        # sftp.  Tried running it flat without the yields, but it really
        # chokes on big directories.
        path = remotepath
        files = []
        folders = []
        for f in self.sftp.listdir_attr(remotepath):
            if S_ISDIR(f.st_mode):
                folders.append(f.filename)
            else:
                files.append(f.filename)
        print(path, folders, files)
        yield path, folders, files
        for folder in folders:
            new_path = os.path.join(remotepath, folder).replace("\\", "/")
            for x in self.sftp_walk(new_path):
                yield x

    def get_all(self, remotepath, localpath):
        #  recursively download a full directory
        #  Harder than it sounded at first, since paramiko won't walk
        # For the record, something like this would gennerally be faster:
        # ssh user@host 'tar -cz /source/folder' | tar -xz
        localpath = localpath[:-1] if localpath[-1] == "/" else localpath
        remotepath = remotepath[:-1] if remotepath[-1] == "/" else remotepath

        self.sftp.chdir(os.path.split(remotepath)[0])
        parent = os.path.split(remotepath)[1]
        try:
            os.mkdir(localpath)
        except:
            pass
        for walker in self.sftp_walk(parent):
            try:
                os.mkdir(os.path.join(localpath, walker[0]).replace("\\", "/"))
            except:
                pass
            for file in walker[2]:
                print(
                    os.path.join(walker[0], file).replace("\\", "/"),
                    os.path.join(localpath, walker[0], file).replace("\\", "/"),
                )
                self.get(
                    os.path.join(walker[0], file).replace("\\", "/"),
                    os.path.join(localpath, walker[0], file).replace("\\", "/"),
                )

    def write_command(self, text, remotefile):
        #  Writes text to remotefile, and makes remotefile executable.
        #  This is perhaps a bit niche, but I was thinking I needed it.
        #  For the record, I was incorrect.
        self.sftp.open(remotefile, "w").write(text)
        self.sftp.chmod(remotefile, 755)

    def python_script(
        self, ipython_path="/home/ubuntu/anaconda3/bin/ipython ", script_path="", args1=""
    ):

        cmd1 = ipython_path + " " + script_path + " " + '"' + args1 + '"'
        self.cmd2(cmd1)
        # self.command(cmd1)

    def command_list(self, cmdlist):
        for command in cmdlist:
            print("Executing {}".format(command))
            ret = self.command(command)
            print(ret)
        print("End of SSH Command")

    def listdir(self, remotedir):
        return self.sftp.listdir(remotedir)

    def jupyter_kill(self):
        pid_jupyter = aws_ec2_ssh_cmd(cmdlist=["fuser 8888/tcp"], host=self.host, doreturn=1)[0][
            0
        ].strip()
        print(self.command("kill -9 " + pid_jupyter))

    def jupyter_start(self):
        pass

    def cmd2(self, cmd1):
        return aws_ec2_ssh_cmd(cmdlist=[cmd1], host=self.host, doreturn=1)

    def _help_ssh(self):
        s = """
         fuser 8888/tcp     Check if Jupyter is running
           ps -ef | grep python     :List of  PID Python process
          kill -9 PID_number     (i.e. the pid returned)
        top     : CPU usage
       """
        print(s)

    def put_all_zip(
        self,
        suffixfolder="",
        fromfolder="",
        tofolder="",
        use_relativepath=True,
        usezip=True,
        filefilter="*.*",
        directorylevel=1,
        verbose=0,
    ):
        """
    fromfolder:  c:/folder0/folder1/folder2/ *
    suffixfolder:         /folders1/folder2/
    tofolder:    /home/ubuntu/myproject1/
    
      """
        fromfolder = fromfolder if fromfolder[-1] != "/" else fromfolder[:-1]
        tmpfolder = "/".join(fromfolder.split("/")[:-1]) + "/"
        zipname = fromfolder.split("/")[-1] + ".zip"
        print(tmpfolder, zipname)
        zipath = tmpfolder + zipname

        print("zipping")
        filezip = util.os_zipfolder(
            dir_tozip=fromfolder, zipname=zipath, dir_prefix=True, iscompress=True
        )
        print("ssh on remote")
        remote_zipath = os.path.join(tofolder, zipname).replace("\\", "/")
        self.put(filezip, remote_zipath)

        if usezip:
            print("Unzip on remote " + self.host)
            cmd = "/usr/bin/unzip " + remote_zipath + " -d " + tofolder
            ss = self.command(cmd)
            if verbose:
                print(ss)


def aws_ec2_ssh_create_con(
    contype="sftp/ssh",
    host="ip",
    port=22,
    username="ubuntu",
    keyfilepath="",
    password="",
    keyfiletype="RSA",
    isprint=1,
):
    """ Transfert File  host = '52.79.79.1'
        keyfilepath = 'D:/_devs/aws/keypairs/ec2_instanc'

# List files in the default directory on the remote computer.
dirlist = sftp.listdir('.')
sftp.get('remote_file.txt', 'downloaded_file.txt')
sftp.put('testfile.txt', 'remote_testfile.txt')

http://docs.paramiko.org/en/2.1/api/sftp.html
    """
    import paramiko

    sftp, ssh, transport = None, None, None
    try:
        if keyfilepath == "":
            keyfilepath = AWS_KEY_PEM
        if keyfiletype == "DSA":
            key = paramiko.DSSKey.from_private_key_file(keyfilepath)
        else:
            key = paramiko.RSAKey.from_private_key_file(keyfilepath)

        if contype == "sftp":
            # Create Transport object using supplied method of authentication.
            transport = paramiko.Transport((host, port))
            transport.add_server_key(key)
            transport.connect(None, username, pkey=key)
            sftp = paramiko.SFTPClient.from_transport(transport)
            if isprint:
                print("Root Directory :\n ", sftp.listdir())
            return sftp

        if contype == "ssh":
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname=host, username=username, pkey=key)

            # Test
            if isprint:
                stdin, stdout, stderr = ssh.exec_command("uptime;ls -l")
                stdin.flush()  # Execute
                data = stdout.read().splitlines()  # Get data
                print("Test Print Directory ls :")
                for line in data:
                    print(line)
            return ssh

    except Exception as e:
        print("An error occurred creating client: %s: %s" % (e.__class__, e))
        if sftp is not None:
            sftp.close()
        if transport is not None:
            transport.close()
        if ssh is not None:
            ssh.close()


def aws_ec2_ssh_cmd(
    cmdlist=["ls "], host="ip", doreturn=0, ssh=None, username="ubuntu", keyfilepath=""
):
    """ SSH Linux terminal Command

      ### PEM File is needed
      aws_ec2_ssh_cmd(cmdlist=  ["ls " ],  host='52.26.181.200', doreturn=1, username='ubuntu', keyfilepath='') 

    fuser 8888/tcp     Check if Jupyter is running
    ps -ef | grep python     :List of  PID Python process
    kill -9 PID_number     (i.e. the pid returned)
    top     : CPU usage
    
      Run nohup python bgservice.py & to get the script to ignore the hangup signal and keep running.
      Output will be put in nohup.out.
    https://aws.amazon.com/blogs/compute/scheduling-ssh-jobs-using-aws-lambda/

   """
    if ssh is None and len(host) > 5:
        ssh = aws_ec2_ssh_create_con(
            contype="ssh", host=host, port=22, username=username, keyfilepath=""
        )
        print("EC2 connected")

    c = cmdlist
    if isinstance(c, str):  # Only Command to be launched
        if c == "python":
            cmdlist = ["ps -ef | grep python "]
        elif c == "jupyter":
            cmdlist = ["fuser 8888/tcp  "]

    readall = []
    for command in cmdlist:
        print("Executing {}".format(command))
        stdin, stdout, stderr = ssh.exec_command(command)
        outread, erread = stdout.read(), stderr.read()
        readall.append((outread, erread))
        print(outread)
        print(erread)
    print("End of SSH Command")
    ssh.close()
    if doreturn:
        return readall


def aws_ec2_ssh_python_script(
    python_path="/home/ubuntu/anaconda2/bin/ipython", script_path="", args1="", host=""
):
    #!!! No space after ipython
    cmd1 = python_path + " " + script_path + " " + '"' + args1 + '"'
    aws_ec2_ssh_cmd(cmdlist=[cmd1], ssh=None, host=host, username="ubuntu")


###############################################################################
def aws_ec2_get_instances(con=None, attributes=None, filters=None, csv_filename=".csv"):
    """Fetch all EC2 instances and write selected attributes into a csv file.
    Parameters:
      connection: EC2Connection object
      attributes: Tuple of attributes to retrieve. Default: id, ip_address
      filters={"ip_address":""}
    """
    if attributes is None:
        attributes = EC2_ATTRIBUTES
    instances = con.get_only_instances(filters=filters)
    instance_list = []
    for instance in instances:
        x = {attribute: getattr(instance, attribute) for attribute in attributes}
        instance_list.append(x)

    try:
        if len(csv_filename) > 4:
            with open(csv_filename, "a") as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=",")
                csvwriter.writerow(list(attributes))
                for xi in instance_list:
                    csvwriter.writerow([xi[t] for t in attributes])
                print("EC2 instance", csv_filename)
    except Exception as e:
        print(e)

    return instance_list


def aws_ec2_getfrom_ec2(fromfolder, tofolder, host):
    sftp = aws_ec2_ssh_create_con(contype="sftp", host=host)

    if fromfolder.find(".") > -1:  # file
        folder1, file1 = util.z_key_splitinto_dir_name(
            fromfolder[:-1] if fromfolder[-1] == "/" else fromfolder
        )
        tofolder2 = tofolder if tofolder.find(".") > -1 else tofolder + "/" + file1
        sftp.get(fromfolder, tofolder2)

    else:  # Pass the Folder in Loop
        pass


def aws_ec2_putfolder(fromfolder="D:/_d20161220/", tofolder="/linux/batch", host=""):
    # fromfolder= DIRCWD +'/linux/batch/task/elvis_prod_20161220/'
    # tofolder=   '/linux/batch/'
    # If you don't care whether the file already exists and you always want to overwrite the files as they are extracted without prompting use the -o switch as follows:
    # https://www.lifewire.com/examples-linux-unzip-command-2201157
    # unzip -o filename.zip

    folder1, file1 = util.z_key_splitinto_dir_name(
        fromfolder[:-1] if fromfolder[-1] == "/" else fromfolder
    )
    tofolderfull = EC2CWD + "/" + tofolder if tofolder.find(EC2CWD) == -1 else tofolder

    # Zip folder before sending it
    file2 = folder1 + "/" + file1 + ".zip"
    util.os_zipfolder(fromfolder, file2)
    res = aws_ec2_putfile(file2, tofolder=tofolderfull, host=host)
    print(res)

    ###### Need install sudo apt-get install zip unzip
    cmd1 = "/usr/bin/unzip " + tofolderfull + "/" + file1 + ".zip " + " -d " + tofolderfull + "/"
    aws_ec2_ssh_cmd(cmdlist=[cmd1], host=host)


def aws_ec2_put(
    fromfolder="d:/file1.zip", tofolder="/home/notebook/aapackage/", host="", typecopy="code"
):
    """
   Copy python code, copy specific file, copy all folder content
  :param fromfolder: 1 file or 1 folder
  :param tofolder:
  :param host:
  """
    if fromfolder.find(".") > -1:  # Copy 1 file
        aws_ec2_putfile(fromfolder=fromfolder, tofolder=tofolder, host=host)

    else:  # Copy Folder to
        sftp = aws_ec2_ssh_create_con("sftp", host, isprint=1)
        if typecopy == "code" and fromfolder.find(".") == -1:  # Local folder and code folder
            # foldername = fromfolder
            # fromfolder = DIRCWD+ '/' + foldername
            tempfolder = DIRCWD + "/ztemp/" + foldername

            util.os_folder_delete(tempfolder)
            util.os_folder_copy(fromfolder, tempfolder, pattern1="*.py")
            sftp.put(tempfolder, tofolder)
            return 1

        if typecopy == "all":
            if fromfolder.find(":") == -1:
                print("Please put absolute path")
                return 0
            if fromfolder.find(".") > -1:  # 1 file
                fromfolder, file1 = util.os_split_dir_file(fromfolder)
                tofull = tofolder + "/" + file1 if tofolder.find(".") == -1 else tofolder
                tofolder, file2 = util.os_split_dir_file(tofull)

                sftp.put(fromfolder + "/" + file1, tofull)

                try:
                    sftp.stat(tofull)
                    isexist = True
                except:
                    isexist = False
                print(isexist, tofull)


def aws_ec2_putfile(fromfolder="d:/file1.zip", tofolder="/home/notebook/aapackage/", host=""):
    """
   Copy python code, copy specific file, copy all folder content
  :param fromfolder: 1 file or 1 folder
  :param tofolder:
  :param host:
  """
    sftp = aws_ec2_ssh_create_con("sftp", host, isprint=1)

    if fromfolder.find(".") > -1:  # Copy 1 file
        if fromfolder.find(":") == -1:
            print("Please put absolute path")
            return 0

        fromfolder2, file1 = util.z_key_splitinto_dir_name(fromfolder)
        tofull = tofolder + "/" + file1 if tofolder.find(".") == -1 else tofolder

        print("from:", fromfolder, "to:", tofull)
        isexist = False
        try:
            sftp.put(fromfolder, tofull)
            ss = sftp.stat(tofull)
            isexist = True
        except Exception as e:
            print(e)
        sftp.close()
        return (isexist, fromfolder, tofull, ss)


##############################################################################################
def sleep2(wsec):
    from time import sleep
    from tqdm import tqdm

    for i in tqdm(range(wsec)):
        sleep(1)


def sftp_isdir(path):
    from stat import S_ISDIR

    try:
        return S_ISDIR(sftp.stat(path).st_mode)
    except IOError:
        # Path does not exist, so by definition not a directory
        return False


def aws_ec2_getfolder():
    import paramiko, os

    paramiko.util.log_to_file("/tmp/paramiko.log")
    from stat import S_ISDIR

    def sftp_walk(remotepath):
        path = remotepath
        files = []
        folders = []
        for f in sftp.listdir_attr(remotepath):
            if S_ISDIR(f.st_mode):
                folders.append(f.filename)
            else:
                files.append(f.filename)
        if files:
            yield path, files
        for folder in folders:
            new_path = os.path.join(remotepath, folder)
            for x in sftp_walk(new_path):
                yield x

    for path, files in sftp_walk("." or "/remotepath/"):
        for file1 in files:
            # sftp.get(remote, local) line for dowloading.
            sftp.get(os.path.join(os.path.join(path, file1)), "/local/path/")


##############################################################################################
############################ CLI #############################################################
if __name__ == "__main__":
    print("Start")
    import argparse

    ppa = argparse.ArgumentParser()  # Command Line input
    ppa.add_argument("--do", type=str, default="action", help="start_spot")
    ppa.add_argument("--price", type=float, default=0.5, help="spot price")
    ppa.add_argument("--spec_file", type=str, default="", help="spec file")

    ppa.add_argument("--spec_file2", type=str, default="", help="spec file")

    arg0 = ppa.parse_args()


if __name__ == "__main__" and arg0.do == "start_spot_windows":
    # :\_devs\Python01\aws\aapackage\
    # D:\_devs\Python01\ana27\python D:\_devs\Python01\aws\aapackage\util_aws.py --do start_spot
    print(arg0.do)
    """ aws_ec2_spot_start(EC2_CONN, "west-2", key_name="ecsInstanceRole", inst_type="cx2.2",  ami_id="", pricemax=0.15,  elastic_ip='',
                     pars={"security_group": [""], "disk_size": 25, "disk_type": "ssd", "volume_type": "gp2"})
  """
    ss = 'aws ec2 request-spot-instances   --region us-west-2  --spot-price "0.55" --instance-count 1 '
    ss += ' --type "one-time" --launch-specification "file://D:\_devs\Python01\\awsdoc\\ec_config2.json" '
    print(ss)
    os.system(ss)
    sleep2(65)

    EC2_CONN = aws_conn_create_windows()
    ec2_list = aws_ec2_get_instances(EC2_CONN, csv_filename="zz_ec2_instance.csv")
    print(ec2_list)

    for x in ec2_list:
        if x["state"] == "running":
            sleep(5)
            ss = ' start "Chrome" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" '
            ss += ' "http://' + x["ip_address"] + ':8888/tree#notebooks"  '
            os.system(ss)


if __name__ == "__main__" and arg0.do == "put_file":
    aws_ec2_putfile(fromfolder=arg0.fromfolder, tofolder=arg0.tofolder, host=arg0.host)


"""
import util

aws_ec2_put( fromfolder='D:/_devs/Python01/aws/aapackage/util.py', 
             tofolder='batch/', host='52.26.181.200', typecopy='code') 



#Copy Folder to Remote
util.aws_ec2_putfolder(fromfolder='D:/_devs/Python01/project27//linux/batch/task/elvis_prod_20161220/',
                  tofolder='/linux/batch/task/', host=host)


#Copy Files
util.aws_ec2_put(DIRCWD + '/linux/batch/task/elvis_prod_20161220.zip',
                 tofolder=EC2CWD + '/linux/batch/task/',
                 host=host, typecopy='all')



http://52.26.181.200:8888/tree#notebooks


"""


"""
Only Spot Instance
ec2_list = aws_ec2_get_instances( con=EC2_CONN,  attributes=EC2_ATTRIBUTES, 
filters={"InstanceState":"running"}, csv_filename=".csv")
print(ec2_list)


con=  cloud_conn_create( "gcloud",
                         {'project': "myproject",  'vmname':"my_vm",  "ip": '25.25.25',
                          'user': 'my_user',       'pass': '25' } )





def aws_ec2_jupyter(host):
   Launch Jupyter server with in Backgroun Process
   At the terminal execute which jupyter. Observe the path. In both local terminal and connecting via ssh, execute echo $PATH.
Ensure that the path to jupyter is included in the $PATH in your environment. – user4556274 10 hours ago

The simple stuff
PATH=$PATH:~/opt/bin
PATH=~/opt/bin:$PATH
depending on whether you want to add ~/opt/bin at the end (to be searched after all other directories, in case
there is a program by the same name in multiple directories) or at the beginning (to be searched before all other directories).

   aws_ec2_cmd_ssh(['nohup /home/ubuntu/anaconda2/bin/jupyter notebook'], host=host, username='ubuntu')


http://boto.cloudhackers.com/en/latest/s3_tut.html

Storing Large Data¶

At times the data you may want to store will be hundreds of megabytes or more in size. S3 allows you to split such files into smaller components. You upload each component in turn and then S3 combines them into the final object. While this is fairly straightforward, it requires a few extra steps to be taken. The example below makes use of the FileChunkIO module, so pip install FileChunkIO if it isn’t already installed.

import math, os
import boto
from filechunkio import FileChunkIO

# Connect to S3
c = boto.connect_s3()
b = c.get_bucket('mybucket')

# Get file info
source_path = 'path/to/your/file.ext'
source_size = os.stat(source_path).st_size

# Create a multipart upload request
mp = b.initiate_multipart_upload(os.path.basename(source_path))

# Use a chunk size of 50 MiB (feel free to change this)
chunk_size = 52428800
chunk_count = int(math.ceil(source_size / float(chunk_size)))

# Send the file parts, using FileChunkIO to create a file-like object
# that points to a certain byte range within the original file. We
# set bytes to never exceed the original file size.
for i in range(chunk_count):
    offset = chunk_size * i
    bytes = min(chunk_size, source_size - offset)
    with FileChunkIO(source_path, 'r', offset=offset,
                         bytes=bytes) as fp:
        mp.upload_part_from_file(fp, part_num=i + 1)

# Finish the upload
mp.complete_upload()




##################################################################################################
################  Amazon testing  in SSH #########################################################
host=    '52.79.79.1'
EC2CWD=      '/home/ubuntu/notebook/'
EC2_ipython= '/home/ubuntu/anaconda2/bin/'

##################################################################################################

ssh= util.aws_ec2_ssh(host)
print ssh.command('ls ')

#### Copy Folder to EC2
ssh.put_all( DIRCWD +'linux/batch/task/elvis_prod_20161220', EC2CWD + '/linux/batch/task' )


#### Retrieve Folder from EC2
ssh.get_all(  EC2CWD + '/linux/batch/task',  DIRCWD +'/zdisks3/fromec2' )



#Execute Script  Single Batch  GOOD !
batch_folder= EC2CWD + '/linux/batch/task/elvis_prod_20161220/'
ssh.python_script(batch_folder+'pygmo_batch_02.py', 'args1')



#### Retrieve Results from EC2
ssh.listdir(EC2CWD + '/linux/batch/output/20161228')

ssh.get_all(  EC2CWD + '/linux/batch/output/20161228',  DIRCWD +'/zdisks3/fromec2/new' )




########################  Full Batch    #######################################################
##  ipython /home/ubuntu/notebook//linux/batch/task/elvis_prod_20161220/batch_launcher_01.py
batch_folder= EC2CWD + '/linux/batch/task/elvis_prod_20161220/'
ssh.python_script(batch_folder+'batch_launcher_01.py', 'args1')


c4.4xlarge 	16 CPU  $0.1293  /  2h
Split into 16 batch !!!

Generate list of parameters :
Split into 15 parts

input_params_all()
 for
   for
      pi=
      input_params_all.append(pi)

#save in folder

0.10


for ii in xrange(i0, i1) :
  pi= input_params_all[ii,:]
  var1= pi[ii,:

  execfile()

4**6: 4096
256

48000 : combinaison of params
6 loops with 5 elelmets



print ssh.command('ps -ef | grep python ')


a= ssh.command('ps --sort=-pcpu | head -n 6')
print a


print ssh.command('top -b | head -n 8')


util.aws_ec2_cmd_ssh(cmdlist=  ['ps -ef | grep python '], host=host ,doreturn=1)




####################### Execute Script
batch_folder= EC2CWD + '/linux/batch/task/elvis_prod_20161220/'

#Execute Script  Single Batch  GOOD !
util.aws_ec2_python_script(batch_folder+'pygmo_batch_02.py', 'args1', host)




#Start Jupyter in nohup mode
util.aws_ec2_jupyter(host)



#Copy Folder to Remote
util.aws_ec2_putfolder(fromfolder='D:/_devs/Python01/project27//linux/batch/task/elvis_prod_20161220/',
                  tofolder='/linux/batch/task/', host=host)


#Copy Files
util.aws_ec2_put(DIRCWD + '/linux/batch/task/elvis_prod_20161220.zip',
                 tofolder=EC2CWD + '/linux/batch/task/',
                 host=host, typecopy='all')

#UnZip
#   Need install sudo apt-get install zip unzip
foldertarget= '/linux/batch/task/elvis_prod_20161220/*'
file1=EC2CWD + 'linux/batch/task/elvis_prod_20161220.zip'
cmd= '/usr/bin/unzip '+ file1 + ' -d ' + EC2CWD + 'linux/batch/task/'
util.aws_ec2_cmd_ssh(cmdlist=  [cmd], host=host)



##################################################################################################
####################### Execute Script
batch_folder= EC2CWD + '/linux/batch/task/elvis_prod_20161220/'

#Execute Script  Single Batch  GOOD !
util.aws_ec2_python_script(batch_folder+'pygmo_batch_02.py', 'args1', host)


#Full Scripts
util.aws_ec2_python_script(batch_folder+'batch_launcher_01.py', 'args1', host)




lfile= [ (EC2CWD+'/linux/batch/output/20161228/batch_20161228_090648553369/', 'output_result.txt'),
 (EC2CWD+'/linux/batch/output/20161228/batch_20161228_090648553369/', 'aafolio_storage_20161228.pkl'),
]

sftp= util.aws_ec2_create_con('sftp', host)
for x in lfile :
  sftp.get( x[0] + '/' +x[1], DIRCWD+'/zdisks3/fromec2/'+ x[1])




###############################################################################
#### SFTp Command ################################################
sftp= util.aws_ec2_create_con('sftp', host)

sftp.listdir(EC2CWD + '/linux/batch/task/')
sftp.mkdir(EC2CWD + '/newdir')
sftp.put(DIRCWD +'linux/batch/task/elvis_prod_20161220.zip', EC2CWD + 'linux/batch/task/elvis_prod_20161220.zip')

sftp.listdir()






###### SSH Command   ##########################################################
ssh= util.aws_ec2_create_con('ssh', host)

#Delete folder
rm *

#Check CPU Time
util.aws_ec2_cmd_ssh(cmdlist=  ['top '], host=host)


#Monitor the process
# top   : Monitor usage
# apt-get install sysstat
# https://www.cyberciti.biz/tips/how-do-i-find-out-linux-cpu-utilization.html




###############################################################################
accepted
please check the below code from the link
 https://gist.github.com/johnfink8/2190472. 
I have used Put_all method in the below snippet



import
util.aws_ec2_jupyter(host)

print(a)

###############################################################################


CHUNKSIZE = 10240  # how much to read at a time
txt= proc.stdout.read(CHUNKSIZE)
print txt


################ Watcher on the processs
from subprocess import Popen, PIPE
from threading import Thread
from Queue import Queue, Empty

io_q = Queue()

def stream_watcher(identifier, stream):
    for line in stream:
        io_q.put((identifier, line))
    if not stream.closed:        stream.close()

proc = Popen('svn co svn+ssh://myrepo', stdout=PIPE, stderr=PIPE)


Thread(target=stream_watcher, name='stdout-watcher',   args=('STDOUT', proc.stdout)).start()
Thread(target=stream_watcher, name='stderr-watcher',   args=('STDERR', proc.stderr)).start()


def printer():
    while True:
        try:
            # Block for 1 second.
            item = io_q.get(True, 1)
        except Empty:
            # No output in either streams for a second. Are we done?
            if proc.poll() is not None:                break
        else:
            identifier, line = item
            print identifier + ':', line

Thread(target=printer, name='printer').start()




#############################################################################



#block the main
subprocess.call(["ipython", filescript ])

#No block the main
subprocess.Popen(["ipython", filescript] )


import util

util.os_process_run(cmd_list=["ipython", filescript], shell=False)



    import subprocess
    #cmd_list= os_path_norm(cmd_list)
    PIPE=subprocess.PIPE
    STDOUT=subprocess.STDOUT
    proc = subprocess.Popen(cmd_list, stdout=PIPE, stderr=STDOUT, shell=shell)
    stdout, stderr = proc.communicate()
    err_code = proc.returncode
    print("Console Msg: \n")
    print(str(stdout)) #,"utf-8"))
    print("\nConsole Error: \n"+ str(stderr) )
    #    return stdout, stderr, int(err_code)






proc.returncode

buf = StringIO()




while True:
    # do whatever other time-consuming work you want here, including monitoring
    # other processes...


    # this keeps the pipe from filling up
    buf.write(proc.stdout.read(CHUNKSIZE))

    proc.poll()
    if proc.returncode is not None:
        # process has finished running
        buf.write(proc.stdout.read())
        print "return code is", proc.returncode
        print "output is", buf.getvalue()

        break





https://peteris.rocks/blog/script-to-launch-amazon-ec2-spot-instances/



#!/usr/bin/python2.7 -u

# pip install boto paramiko

import argparse
import boto, boto.ec2, boto.ec2.blockdevicemapping, boto.manage
import paramiko
import os, sys, time

#boto.set_stream_logger('boto')

def launch_spot_instance(id, profile, spot_wait_sleep=5, instance_wait_sleep=3):
  ec2 = boto.ec2.connect_to_region(profile['region'])

  if not 'key_pair' in profile:
    profile['key_pair'] = ('KP-' + id, 'KP-' + id + '.pem')
    try:
      print >> sys.stderr, 'Creating key pair...',
      keypair = ec2.create_key_pair('KP-' + id)
      keypair.save('.')
      print >> sys.stderr, 'created'
    except boto.exception.EC2ResponseError as e:
      if e.code == 'InvalidKeyPair.Duplicate':
        print >> sys.stderr, 'already exists'
      else:
        raise e

  if not 'security_group' in profile:
    try:
      print >> sys.stderr, 'Creating security group...',
      sc = ec2.create_security_group('SG-' + id, 'Security Group for ' + id)
      for proto, fromport, toport, ip in profile['firewall']:
        sc.authorize(proto, fromport, toport, ip)
      profile['security_group'] = (sc.id, sc.name)
      print >> sys.stderr, 'created'
    except boto.exception.EC2ResponseError as e:
      if e.code == 'InvalidGroup.Duplicate':
        print >> sys.stderr, 'already exists'
        sc = ec2.get_all_security_groups(groupnames=['SG-' + id])[0]
        profile['security_group'] = (sc.id, sc.name)
      else:
        raise e

  existing_requests = ec2.get_all_spot_instance_requests(filters={'launch.group-id': profile['security_group'][0], 'state': ['open', 'active']})
  if existing_requests:
    if len(existing_requests) > 1:
      raise Exception('Too many existing spot requests')
    print >> sys.stderr, 'Reusing existing spot request'
    spot_req_id = existing_requests[0].id
  else:
    bdm = boto.ec2.blockdevicemapping.BlockDeviceMapping()
    bdm['/dev/sda1'] = boto.ec2.blockdevicemapping.BlockDeviceType(volume_type='gp2', size=profile['disk_size'], delete_on_termination=profile['disk_delete_on_termination'])
    bdm['/dev/sdb'] = boto.ec2.blockdevicemapping.BlockDeviceType(ephemeral_name='ephemeral0')
    print >> sys.stderr, 'Requesting spot instance'
    spot_reqs = ec2.request_spot_instances(
      price=profile['price'], image_id=profile['image_id'], instance_type=profile['type'], placement=profile['region'] + profile['availability_zone'],
      security_groups=[profile['security_group'][1]], key_name=profile['key_pair'][0], block_device_map=bdm)
    spot_req_id = spot_reqs[0].id

  print >> sys.stderr, 'Waiting for launch',
  instance_id = None
  spot_tag_added = False
  while not instance_id:
    spot_req = ec2.get_all_spot_instance_requests(request_ids=[spot_req_id])[0]
    if not spot_tag_added:
      spot_req.add_tag('Name', id)
      spot_tag_added = True
    if spot_req.state == 'failed':
      raise Exception('Spot request failed')
    instance_id = spot_req.instance_id
    if not instance_id:
      print >> sys.stderr, '.',
      time.sleep(spot_wait_sleep)
  print >> sys.stderr

  print >> sys.stderr, 'Retrieving instance by id'
  reservations = ec2.get_all_instances(instance_ids=[instance_id])
  instance = reservations[0].instances[0]
  instance.add_tag('Name', id)
  print >> sys.stderr, 'Got instance: ' + str(instance.id) +  ' [' + instance.state + ']'
  print >> sys.stderr, 'Waiting for instance to boot',
  while not instance.state in ['running', 'terminated', 'shutting-down']:
    print >> sys.stderr, '.',
    time.sleep(instance_wait_sleep)
    instance.update()
  print >> sys.stderr
  if instance.state != 'running':
    raise Exception('Instance was terminated')
  return instance

def connect_to_instance(ip, username, key_filename, timeout=10):
  print >> sys.stderr, 'Connecting to SSH [' + ip + '] ',
  client = paramiko.SSHClient()
  client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
  retries = 0
  while retries < 30:
    try:
      print >> sys.stderr, '.',
      client.connect(ip, username=username, key_filename=key_filename, timeout=timeout)
      break
    except:
      retries += 1
  print >> sys.stderr
  return client

def setup_instance(id, instance, file, user_name, key_name):
  script = open(file, 'r').read().replace('\r', '')

  client = connect_to_instance(instance.ip_address, user_name, key_name)
  session = client.get_transport().open_session()
  session.set_combine_stderr(True)

  print >> sys.stderr, 'Running script: ' + os.path.relpath(file, os.getcwd())
  session.exec_command(script)
  stdout = session.makefile()
  try:
    for line in stdout:
      print line.rstrip()
  except (KeyboardInterrupt, SystemExit):
    print >> sys.stderr, 'Ctrl-C, stopping'
  client.close()
  exit_code = session.recv_exit_status()
  print >> sys.stderr, 'Exit code: ' + str(exit_code)
  return exit_code == 0

if __name__ == '__main__':

  profiles = {
    '15G': {
      'region': 'eu-west-1',
      'availability_zone': 'a',
      'price': '0.05',
      'type': 'r3.large',
      'image_id': 'ami-ed82e39e',
      'username': 'ubuntu',
      #'key_pair': ('AWS-EU', 'eu-key.pem'),
      'disk_size': 20,
      'disk_delete_on_termination': True,
      'scripts': [],
      'firewall': [ ('tcp', 22, 22, '0.0.0.0/0') ]
    }
  }

  parser = argparse.ArgumentParser(description='Launch spot instance')
  parser.add_argument('-n', '--name', help='Name', required=True)
  parser.add_argument('-p', '--profile', help='Profile', default=profiles.keys()[0], choices=profiles.keys())
  parser.add_argument('-s', '--script', help='Script path', action='append', default=[])
  parser.add_argument('-i', '--interactive', help='Connect to SSH', action='store_true')
  args = parser.parse_args()

  profile = profiles[args.profile]

  try:
    instance = launch_spot_instance(args.name, profile)
  except boto.exception.NoAuthHandlerFound:
    print >> sys.stderr, 'Error: No credentials found, try setting the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables'
    sys.exit(1)

  for script in profile['scripts'] + args.script:
    if not setup_instance(id=args.name, instance=instance, file=script, user_name=profile['username'], key_name=profile['key_pair'][1]):
      break

  if args.interactive:
    print 'ssh ' + profile['username'] + '@' + instance.ip_address + ' -i ' + profile['key_pair'][1] + ' -oStrictHostKeyCheck




  """
