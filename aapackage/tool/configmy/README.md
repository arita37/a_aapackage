configmy
Easiest config loader in Python

Goal is to load config dynamically in 1 liner code (for linux, windows, yython 2 or python3).

import configmy; CFG= configmy.get() #CFG is python dictionnary containing the current config, directory and details.

Then we can acces with :


CFG.github_login
CFG.ROOTFOLDER
CFG.aws_login
CFG.selenium_folder
CFG.JAVA_HOME



Config is define with a dict embed in python file, and dynamically selection
based on OS_NAME, USER_NAME, Python_version















