# -*- coding: utf-8 -*-

class AWSGLOBALS:
    """All the globals for AWS utility functionalities."""
    AWS_ACCESS_LOCAL = 'D:/_devs/keypair/aws_access.py'
    AWS_KEY_PEM = "D:/_devs/keypair/oregon/aws_ec2_oregon.pem"
    AWS_REGION = "us-west-2"
    APNORTHEAST2 = 'ap-northeast-2'
    EC2CWD = '/home/ubuntu/notebook/'
    EC2_CONN = None
    EC2_FILTERS = ('id', 'ip_address')
    EC2_ATTRIBUTES = (
        "id", "instance_type", "state", "public_dns_name", "private_dns_name",
        "state_code", "previous_state", "previous_state_code", "key_name",
        "launch_time", "image_id", "placement", "placement_group",
        "placement_tenancy", "kernel", "ramdisk", "architecture", "hypervisor",
        "virtualization_type", "product_codes", "ami_launch_index", "monitored",
        "monitoring_state", "spot_instance_request_id", "subnet_id", "vpc_id",
        "private_ip_address", "ip_address", "platform", "root_device_name",
        "root_device_type", "state_reason", "interfaces", "ebs_optimized",
        "instance_profile"
    )

    def __init__(self, ):
        """Nothing to be constructed """
        pass

    @classmethod
    def get_keypair(cls):
        """Get the current keypair used"""
        return None, None

    @classmethod
    def set_keypair(cls, keypairname, keypairlocation):
        """Set the keypair to be used."""
        pass

    @classmethod
    def set_attribute(cls, key, value):
        """Add or update attribute to the class, maybe protect with a lock."""
        setattr(cls, key, value)

    @classmethod
    def get_ec2_conn(cls):
        """Return the current EC2 connection."""
        return


""" Global variables
global01.varname import global01 as global01
"""

global ONE_SQRT_2PI, PI, PI2, TWOPI, ONE_2PI
ONE_SQRT_2PI =  0.3989422804014326779399460599343818684758586311649346,
ONE_2PI =  0.159154943091895335768883763372514362034459645740456448747,
PI= 3.1415926535897932384626433832795028841971693993751058,
TWOPI= 6.2831853071795864769252867665590057683943387987502115,
PI2= 6.2831853071795864769252867665590057683943387987502115


global DAYS_PER_YEAR
DAYS_PER_YEAR= 252


#-----------Compiler--VS Studio----Path---------------------------------------
global VS_DEVENV, VS_COMPILDIR, VS_MSBUILD, NVIDIA_DRIVER

#Need to put r to have Python compliant for string
VS_DEVENV= r"D:\_app\visualstudio13\Common7\IDE\devenv.exe"
VS_BIN= r"D:\_app\visualstudio13\VC\bin"
VS_MSBUILD= r"D:\_app\visualstudio13\VC\bin"
NVIDIA_DRIVER= r"D:\_app\nvidiacuda\gpu\gdk_win7_amd64_release\nvml"





global lock
global res_shared
global allv



'''
# os.environ['MyVar'] = 'Hello World!' #Variable
# os.getenv('MyVar',-1)   # get your variable


 Winpython   D:\_devs\Python01\WinPython-64\settings

Package        D:\_devs\Python01\WinPython-64\python-3.4.3.amd64\libs
compiler_bindir=   D:\_app\visualstudio13\VC\bin


[cuda]
# Set to where the cuda drivers are installed.
#  change this where your cuda driver/what version is installed.
root=   D:\_app\nvidiacuda\gpu\gdk_win7_amd64_release\nvml

#C:\Program Files\Microsoft Visual Studio 8\Common7\IDE\





'''


    