
conda env list
source activate root



####### Core Install  #############
conda create -n py36t --yes  python=3.6.5
source activate py36t


pip install arrow==0.10.0 attrdict==2.0.0 backports.shutil-get-terminal-size==1.0.0 configmy==0.14.87 github3.py==1.2.0 jwcrypto==0.6.0 kmodes==0.9 rope-py3k==0.9.4.post1 tables==3.3.0 tabulate==0.8.2 uritemplate==3.0.0
pip install pytest==4.3.0
pip install toml


conda upgrade psutil



python -c "import tensorflow as tf; print(tf)"
python -c "import numpy; print('numpy %s' % numpy.__version__)"
    

##### Check  #####################
python --version
conda list
conda env list





