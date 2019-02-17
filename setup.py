from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


version = '0.1.0'

setup(name='aapackage',
      version=version,
      description='Tools for Python',
      author='Kevin Noel',
      author_email='\@gmail.com',
      url='https://github.com/arita37/a_aapackage',
      install_requires=['numpy'],
      packages=find_packages(),
      )




"""


https://packaging.python.org/tutorials/packaging-projects/





"""



