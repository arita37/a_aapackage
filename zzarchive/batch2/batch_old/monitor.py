# -*- coding: utf-8 -*-
"""
Listener:
   Listen --> Event : Launch EC + update database
   Config file
# Keep central db, Separate process
# Generate Tasks in DB : by separate process

Current File Name
import inspect
print inspect.stack()[0][1]

print inspect.getfile(inspect.currentframe())

import sys, os
print "script: sys.argv[0] is", repr(sys.argv[0])
print "script: __file__ is", repr(__file__)
"""

import sys, os

DIR1 = os.getcwd()
DIR_package = os.getcwd() + "/aapackage/aws/"
from boto.ec2.connection import EC2Connection
from time import sleep
import subprocess, configparser, socket, boto, pandas as pd
import util

######################  Initialization    #######################################################
INSTANCE_TYPE = [
    "t1.micro",
    "m1.small",
    "m1.medium",
    "m1.large",
    "m1.xlarge",
    "m3.medium",
    "m3.large",
    "m3.xlarge",
    "m3.2xlarge",
    "c1.medium",
    "c1.xlarge",
    "m2.xlarge",
    "m2.2xlarge",
    "m2.4xlarge",
    "cr1.8xlarge",
    "hi1.4xlarge",
    "hs1.8xlarge",
    "cc1.4xlarge",
    "cg1.4xlarge",
    "cc2.8xlarge",
    "g2.2xlarge",
    "c3.large",
    "c3.xlarge",
    "c3.2xlarge",
    "c3.4xlarge",
    "c3.8xlarge",
    "c4.large",
    "c4.xlarge",
    "c4.2xlarge",
    "c4.4xlarge",
    "c4.8xlarge",
    "i2.xlarge",
    "i2.2xlarge",
    "i2.4xlarge",
    "i2.8xlarge",
    "t2.micro",
    "t2.small",
    "t2.medium",
]
REGIONS = [
    ("ap-northeast-2", "Asia Pacific (Seoul)"),
    ("ap-northeast-1", "Asia Pacific (Tokyo)"),
    ("ap-southeast-1", "Asia Pacific (Singapore)"),
    ("ap-southeast-2", "Asia Pacific (Sydney)"),
    ("eu-central-1", "EU (Frankfurt)"),
    ("eu-west-1", "EU (Ireland)"),
    ("sa-east-1", "South America (Sao Paulo)"),
    ("us-east-1", "US East (N. Virginia)"),
    ("us-west-1", "US West (N. California)"),
    ("us-west-2", "US West (Oregon)"),
]
DBNAME = DIR_package + "task_scheduler.db"

config = configparser.ConfigParser()
config.read(DIR_package + "config.cfg")


############################################################################################################
def aws_accesskey_get():
    access, key = config.get("IAM", "access"), config.get("IAM", "secret")
    # access, key=(boto.config.get('Credentials', 'aws_access_key_id'), boto.config.get('Credentials', 'aws_secret_access_key'))
    return access, key


def aws_conn_getallregions(conn=None):
    return conn.get_all_regions()


def aws_conn_create(region="ap-northeast-2", access="", key=""):
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


def aws_conn_getinfo(conn):
    print(conn.region.name)


def aws_region_get_keypair(ec2, outype=0):
    from random import randint

    rs = ec2.get_all_key_pairs()
    if len(rs) < 1:
        print("No Key-Pair, Creating new one")
        name1 = "ec2_newkeypair_" + str(randint(1000, 9999))
        keypair = ec2.create_key_pair(name1)
        keypair.save(".")
        return name1, keypair
    else:
        for x in rs:
            print(x.name)
        if outype == 0:
            return x.name
        if outype == 1:
            return x.name, x


def aws_region_get_securitygroup_id(conn, isprint=0, kid=0):
    from random import randint

    rs = conn.get_all_security_groups()
    if len(rs) < 1:
        print("No Security Group, Creating new one")
        return None
    else:
        for x in rs:
            print(x.id, x.name)
        return rs[kid].id


############################################################################################################


############################################################################################################
def instance_get_allactive():
    return instance_spot_get_allactive(
        accounts=None, region_names=["ap-northeast-2"], filters={}, isprint=0
    )


def instance_spot_request_cancelall():
    insts = instance_spot_request_getall()
    # istlist=   insts[0][2]; regioname= insts[0][1];  aws_conn_getinfo(conn)
    for account, regioname, istlist in insts:
        conn = aws_conn_create(regioname)
        insts_id = [ist.id for ist in istlist]
        try:
            print("Terminating All bid requests...", end=" ")
            rs = conn.cancel_spot_instance_requests(request_ids=insts_id)
            print("done.")
        except:
            print("Failed to terminate:", sys.exc_info()[0])


def instance_spot_request_getall(
    accounts=None, region_names=["ap-northeast-2"], filters={}, isprint=0
):
    """Will find all requests across all EC2 regions for all accounts
      # filters={'instance-state-name': 'running'}
   """
    if not accounts:
        creds = aws_accesskey_get()
        accounts = {"main": creds}

    if len(region_names) > 0:
        regions = [boto.ec2.get_region(x) for x in region_names]
    else:
        regions = boto.ec2.regions()

    requests = {}
    rsv_by_account_region = []
    for account_name in accounts:
        requests[account_name] = {}
        # ak, sk= accounts[account_name]
        for region in regions:
            conn = aws_conn_create(region.name)
            if conn is not None:

                # Code Specific ------------------------------------------------
                rsvlist = conn.get_all_spot_instance_requests(filters=filters)
                rsv_by_account_region.append((account_name, region.name, rsvlist))
                requests[account_name][region.name] = rsvlist

                if isprint:
                    print((account_name, region.name))
                    for x in rsvlist:
                        print("_____", x.id, x.state, x.price, x.instance_id)

    return rsv_by_account_region

    """
   rs= instance_spot_request_getall(accounts=None,region_names=[])

   rs= instance_spot_request_getall(accounts=None, isprint=1)
   rs
   for x in rs: print    x[2].id
   """


def instance_spot_get_allactive(
    accounts=None, region_names=["ap-northeast-2"], filters={}, isprint=0
):
    """Will find all running instances across all EC2 regions for all of theaccounts supplied.
   Spot Fleet usage ONLY BOTO 3
   http://boto3.readthedocs.io/en/latest/reference/services/ec2.html?highlight=spot%20fleet#EC2.Client.request_spot_fleet
   """
    if not accounts:
        creds = aws_accesskey_get()
        accounts = {"main": creds}

    if len(region_names) > 0:
        regions = [boto.ec2.get_region(x) for x in region_names]
    else:
        regions = boto.ec2.regions()

    running_instances = {}
    rsv_by_account_region = []
    for account_name in accounts:
        running_instances[account_name] = {}
        # ak, sk=accounts[account_name]
        for region in regions:
            conn = aws_conn_create(region.name, "", "")
            if conn is not None:

                filters = {"instance-state-name": "running"}
                rsvlist = conn.get_all_instances(filters=filters)
                rsv_by_account_region.append((account_name, region.name, rsvlist))
                running_instances[account_name][region.name] = rsvlist

                if isprint:
                    print((account_name, region.name))
                    for x in rsvlist:
                        print("_____", x.id, x.state)

    return rsv_by_account_region
    """
   rs= instance_spot_get_allactive(accounts=None, isprint=1)
   rs
   for x in rs: print    x[2].id
   """


def instance_spot_get_pricelast(conn=None, inst_type="C3.Large", region=""):
    price_history = conn.get_spot_price_history(
        instance_type=inst_type, product_description="Linux/UNIX"
    )
    return price_history[0].price


def instance_spot_stopall(conn):
    insts = instance_get_allactive(conn)
    insts_id = [ist.id for ist in insts]
    try:
        print("Terminating All instances...", end=" ")
        conn.terminate_instances(instance_ids=insts_id)
        print("done.")
    except:
        print("Failed to terminate:", sys.exc_info()[0])


def instance_spot_stop(conn, inst):
    try:
        print("Terminating", str(inst.id), "...", end=" ")
        conn.terminate_instances(instance_ids=[inst.id])
        print("done.")
        inst.remove_tag("Name", config.get("EC2", "tag"))
    except:
        print("Failed to terminate:", sys.exc_info()[0])


def instance_spot_request_new(region="ap-northeast-2", pars=None, refreshfreq=5):
    """ New Spot instance request
   :param region:  'ap-northeast-2'
   :param pars: price= pars['max_bid'],pars['ami'],  instance_type=pars['type'], user_data=pars['user_data']
   :param refreshfreq:
   :return: None
   pars={'max_bid': 0.009, 'ami': 'ami-983ce8f6', 'type': 'c4.large', 'user_data': ''  }
   instance_spot_request_new(region='ap-northeast-2', pars=pars)
   """
    conn = aws_conn_create(region)
    keypair = aws_region_get_keypair(conn)
    security_id = aws_region_get_securitygroup_id(conn)

    req = conn.request_spot_instances(
        price=pars["max_bid"],
        image_id=pars["ami"],
        instance_type=pars["type"],
        key_name=keypair,
        user_data=pars["user_data"],
        security_group_ids=[security_id],
    )[0]

    print("Spot request created, status: " + req.state)
    print("Waiting for aws provisioning", end=" ")
    while True:
        current_req = conn.get_all_spot_instance_requests([req.id])[0]

        if current_req.state == "open":
            print("/n Request Open, Pending filling.")
            return None
        sleep(refreshfreq)

        # if current_req.state=='active':
        #   print 'done.'
        #   instance=conn.get_all_instances([current_req.instance_id])[0].instances[0]
        #   # instance.add_tag('Name', config.get('EC2', 'tag'))
        #   return instance
        # print '.',


############################################################################################################
""" # Mode 3: Launch on local machine
1) Schedule task at some precise time ----> Execute on local machine

Use Excel to create Schema of database

Database task      :
    task_id  script_id  max_bid   min_bid  time_run_start  time_run_limit time_run_duration
    repeat_mode   repeat_date
    script_filename script_location script_location_type   data_filename  data_location  data_location_type
    group1  group2 group3  priority1  priority2  priority3  type1  type2   run_type1   runtype2
    group_dependency  group_dependency_order
    task_status(wait/cancel/finish/draft),  time_run_finish,    output_data_location
    summary  date_create  date_update

"""


def task_get_maxbid(tasklist):
    return [0.005, 0.007]


def task_get_pending(dbcon=dbcon):
    """ Task not assigned from tasks table """
    ss = "select * from tasks where status= 'pending' "
    df = util.sql_query(ss, dbcon=dbcon)
    return df


def task_get_assigned(dbcon=dbcon):
    """ Task assigned to a worker  """
    ss = "select * from tasks where status= 'assigned' "
    df = util.sql_query(ss, dbcon=dbcon)
    return df


def task_get_active(dbcon=dbcon):
    """ Task being active to a worker  """
    ss = "select * from tasks where status= 'active' "
    df = util.sql_query(ss, dbcon=dbcon)
    return df


def task_put(task_folder, task_script_main, pars={"name": "new_task"}, dbcon=dbcon):
    """ Put a task in database as pending """
    """
   scol= '( task_folder, task_script_main'
   sval= '(' + task_folder + ','+ task_script_main + ','
   for key,x  in pars :
      sval= str(x) + ','
      scol= str(key) +','
   """
    colname, val = util.np_dict_tolist(pars)
    colname.append(["task_folder", "task_script_main"])
    val.append([task_folder, task_script_main])
    df = util.pd_array_todataframe(val, colname)
    util.sql_insert_df(df, dbcon)


def task_put_batch(df, table1="tasks", dbcon=dbcon):
    """ Put a task dataframe in database as pending """
    df = pd.DataFrame()
    df.to_sql(table1, dbcon, if_exists="append")


############################################################################################################


def monitor_task_sendto_instance(task_list):
    """ Send Task to Celery, Rabbit MQ
     1) Copy Task Folder to S3
     2) Send to Celery Command Line task + Params
     3) Register Task as active vs check with Celery Database

     http://docs.celeryproject.org/en/latest/userguide/workers.html#remote-control
  """
    from . import tasklist  # File containing all the tasks_folder + Celery Launch

    for tt in task_list:
        s3folder = "zdisk/tasks/" + tt.folder_name
        aws_s3_folder_copytos3(tt.folder_location, s3folder)

        # Launch Celery task, Maybe in Sub-Process1  ???--------------------------
        # celery_send_task(tt.id, tt.params, tt.script_location)
        tasklist.t_run_filescript.delay(tt.id, s3folder, tt.params, 5)


def monitor_add_instance_toworker():
    """ When a new Instance is ready, Register to the Available Workers in Celery """
    inst_active = instance_get_allactive()
    worker_active_ipadress = celery_worker_getactive("ipadress")
    for tt in inst_active:
        if util.find(tt.ip_adress, worker_active_ipadress) < 0:
            celery_worker_add(tt.ipaddrss, tt.cloudtype)
            # No need because Worker will look for AM Broker Ip Adress and Look for Tasks.


def _map_computetime_toinstance(tt):
    """ c4.large	2	3.75	EBS-Only	500
   c4.xlarge	4	7.5	EBS-Only	750
   c4.2xlarge	8	15	EBS-Only	1,000
   c4.4xlarge	16	30	EBS-Only	2,000
   c4.8xlarge	36	60
   As documented here, each vCPU is a hyperthread from a Xeon core. So a c4.xlarge has two cores, and with HyperThreads that provides four logical processors (vCPUs).
   The CPU topology with regard to physical threads and cores is provided to the OS running in your C4 EC2 instance through the CPUID instruction. You can use the Intel documentation to determine which vCPUs are paired together on a single core.
      Time: Time_per_task * nbTask

   Time_req=  NbCPU * 4 core * nbHours

   Nb_Of_Instance=   Time_need / Time_available
   Mapping (Task ---> Machine )

   """
    inst_list = []
    if tt < 2 * 60.0:
        m1 = {"max_bid": x.max_bid, "ami": x.ami, "type": "c4.large", "user_data": x.user_data}
        return inst_list.append(m1)

    elif tt < 4 * 60.0:
        m1 = {"max_bid": x.max_bid, "ami": x.ami, "type": "c4.xlarge", "user_data": x.user_data}
        return inst_list.append(m1)

    elif tt < 8 * 60.0:
        m1 = {"max_bid": x.max_bid, "ami": x.ami, "type": "c4.2xlarge", "user_data": x.user_data}
        return inst_list.append(m1)

    elif tt < 16 * 60.0:
        m1 = {"max_bid": x.max_bid, "ami": x.ami, "type": "c4.4xlarge", "user_data": x.user_data}
        return inst_list.append(m1)
    else:
        m1 = {"max_bid": x.max_bid, "ami": x.ami, "type": "c4.4xlarge", "user_data": x.user_data}
        return inst_list.append(m1)


def monitor_instance_start_fromtask(task_pending, pricelist):
    """ Launch Instance from Task List
   :param task_pending:
   :param pricelist:
   """
    sum_dur = 0
    for tt in task_pending:
        sum_dur += tt.time_duration

    inst_list = _map_computetime_toinstance(sum_dur)

    # Launch Bid request
    for x in inst_list:
        pars = {"max_bid": x.max_bid, "ami": x.ami, "type": x.type, "user_data": x.user_data}
        instance_spot_request_new(region=x.region, pars=pars)


############################################################################################################
if __name__ == "__main__":
    # Running Background, Loop on Tasks:  ------------------------------------------------------
    MAX_BID = 0.005
    REFRESH_TIME = 60 * 5

    from . import tasklist  # Start the Celery Scheduler

    while True:
        task_pending = task_get_pending()
        monitor_add_instance_toworker()

        if len(task_pending) > 0:
            inst_active = instance_get_allactive()
            if len(inst_active) == 0:
                monitor_launch_fromtask(task_pending)
            else:
                monitor_sendto_instance(inst_active)

        task_pending = task_get_pending()
        task_active = task_get_active()

        # if len(task_pending) == 0   : instance_spot_request_cancel_all()

        if len(task_active) == 0:
            instance_spot_stop_all()
            instance_spot_request_cancel_all()

        sleep(REFRESH_TIME)


############################################################################################################
"""
class boto.ec2.spotinstancerequest.SpotInstanceRequest(connection=None)

id – The ID of the Spot Instance Request.
price – The maximum hourly price for any Spot Instance launched to fulfill the request.
type – The Spot Instance request type.
state – The state of the Spot Instance request.
fault – The fault codes for the Spot Instance request, if any.
valid_from – The start date of the request. If this is a one-time request, the request becomes active at this date and time and remains active until all instances launch, the request expires, or the request is canceled. If the request is persistent, the request becomes active at this date and time and remains active until it expires or is canceled.
valid_until – The end date of the request. If this is a one-time request, the request remains active until all instances launch, the request is canceled, or this date is reached. If the request is persistent, it remains active until it is canceled or this date is reached.
launch_group – The instance launch group. Launch groups are Spot Instances that launch together and terminate together.
launched_availability_zone – foo
product_description – The Availability Zone in which the bid is launched.
availability_zone_group – The Availability Zone group. If you specify the same Availability Zone group for all Spot Instance requests, all Spot Instances are launched in the same Availability Zone.
create_time – The time stamp when the Spot Instance request was created.
launch_specification – Additional information for launching instances.
instance_id – The instance ID, if an instance has been launched to fulfill the Spot Instance request.
status – The status code and status message describing the Spot Instance request.




for i in instances:
    pprint(i.__dict__)
    break # remove this to list all instances

{'_in_monitoring_element': False,
 'ami_launch_index': u'0',
 'architecture': u'x86_64',
 'block_device_mapping': {},
 'connection': EC2Connection:ec2.amazonaws.com,
 'dns_name': u'ec2-xxx-xxx-xxx-xxx.compute-1.amazonaws.com',
 'id': u'i-xxxxxxxx',
 'image_id': u'ami-xxxxxxxx',
 'instanceState': u'\n                    ',
 'instance_class': None,
 'instance_type': u'm1.large',
 'ip_address': u'xxx.xxx.xxx.xxx',
 'item': u'\n                ',
 'kernel': None,
 'key_name': u'FARM-xxxx',
 'launch_time': u'2009-10-27T17:10:22.000Z',
 'monitored': False,
 'monitoring': u'\n                    ',
 'persistent': False,
 'placement': u'us-east-1d',
 'previous_state': None,
 'private_dns_name': u'ip-10-xxx-xxx-xxx.ec2.internal',
 'private_ip_address': u'10.xxx.xxx.xxx',
 'product_codes': [],
 'public_dns_name': u'ec2-xxx-xxx-xxx-xxx.compute-1.amazonaws.com',
 'ramdisk': None,
 'reason': '',
 'region': RegionInfo:us-east-1,
 'requester_id': None,
 'rootDeviceType': u'instance-store',
 'root_device_name': None,
 'shutdown_state': None,
 'spot_instance_request_id': None,
 'state': u'running',
 'state_code': 16,
 'subnet_id': None,
 'vpc_id': None}


"""
