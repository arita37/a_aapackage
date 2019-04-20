from setuptools import setup, find_packages


with open("README.txt", "r") as fh:
    long_description = fh.read()


### Packages
packages = ['aapackage'] + ['aapackage.' + p for p in find_packages('aapackage')]


### CLI Scripts
scripts  = [ "aapackage/batch/batch_daemon_launch_cli.py", 
             "aapackage/batch/batch_daemon_monitor_cli.py",
             "aapackage/batch/batch_daemon_autoscale_cli.py",
             
             "aapackage/cli_module_autoinstall.py",  #
             "aapackage/cli_module_analysis.py",     #
             "aapackage/cli_convert_ipny.py"         #  ipny to py scrips
            ]


version = '0.1.0'


setup(name='aapackage',
      version=version,
      description='Tools for Python',
      author='KN',
      author_email='brookm291@gmail.com',
      url='https://github.com/arita37/a_aapackage',
      install_requires=['numpy'],
      packages=packages,
      scripts=scripts
      )







################################################################################
################################################################################
"""


https://packaging.python.org/tutorials/packaging-projects/


import os
from io import open

from setuptools import find_packages, setup

packages = ['elfi'] + ['elfi.' + p for p in find_packages('elfi')]

# include C++ examples
package_data = {'elfi.examples': ['cpp/Makefile', 'cpp/*.txt', 'cpp/*.cpp']}

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

optionals = {'doc': ['Sphinx'], 'graphviz': ['graphviz>=0.7.1']}

# read version number
__version__ = open('elfi/__init__.py').readlines()[-1].split(' ')[-1].strip().strip("'\"")

setup(
    name='elfi',
    keywords='abc likelihood-free statistics',
    packages=packages,
    package_data=package_data,
    version=__version__,
    author='ELFI authors',
    author_email='elfi-support@hiit.fi',
    url='http://elfi.readthedocs.io',
    install_requires=requirements,
    extras_require=optionals,
    description='ELFI - Engine for Likelihood-free Inference',
    long_description=(open('docs/description.rst').read()),
    license='BSD',
    classifiers=[
        'Programming Language :: Python :: 3.5', 'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Mathematics', 'Operating System :: OS Independent',
        'Development Status :: 4 - Beta', 'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License'
    ],
    zip_safe=False)



"""



