import io
import os
import subprocess
import sys
from setuptools import setup, find_packages


root = os.path.abspath(os.path.dirname(__file__))


# Check if GPU is available.
p = subprocess.Popen(['command -v nvidia-smi'], stdout=subprocess.PIPE, shell=True)
out = p.communicate()[0].decode('utf8')
gpu_available = len(out) > 0


version = "0.1.0"


with open("README.txt", "r") as fh:
    long_description = fh.read()


packages = ["aapackage"] + ["aapackage." + p for p in find_packages("aapackage")]


# CLI scripts
scripts = [
    "aapackage/batch/batch_daemon_launch_cli.py",
    "aapackage/batch/batch_daemon_monitor_cli.py",
    "aapackage/batch/batch_daemon_autoscale_cli.py",
    "aapackage/cli_module_autoinstall.py",  #
    "aapackage/cli_module_analysis.py",  #
    "aapackage/cli_convert_ipny.py",  #  ipny to py scrips
]


setup(
    name="aapackage",
    version=version,
    description="Tools for Python",
    uthor="KN",
    author_email="brookm291 gmail.com",
    url="https://github.com/arita37/a_aapackage",
    install_requires=["numpy"],
    packages=packages,
    scripts=scripts,
)
