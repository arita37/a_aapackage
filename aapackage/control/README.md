```

################### Install
1) install mini conda

2)
conda create -n py36 python=3.6.7

conda install tensorflow-mkl -c anaconda

conda install -c anaconda pandas scipy scikit-learn seaborn matplotlib 

conda install pytorch=0.4.1 -c pytorch


### TF Wheels
https://github.com/davidenunes/tensorflow-wheels/releases/download/r1.14.cp37.gpu.xla/tensorflow-1.14.0-cp37-cp37m-linux_x86_64.whl




################## Files
main.py : File to run the training.


config.py :  configuation file

solver.py :
   file where the TF graph, train
   

submodels.py : where LSTM, FFoward and Attn LSTM are defined , related to solver.py


equation.py 
    where the time series generation, Neural network definition.



utils.py :  Where the time series generator is written.
    mostly a geoemetric brownina motion :
        Xt =X0 . exp(   ut. t +  volt . Wt  )
        dWt being a gaussian process.




################## How it works.
Sample generator   bdse.sample generates time series sample of X :  4D tensor stacked in 3D format.
   X[ nsample,
      ndim_x   :   3 for 3 assets,
      0...T : time steps
     ]
     
   We feed X into a sub-network.
   For each time step, sub-network predicts some value Z ( a tensor of dimension  (nsample, nim_x).
       
   We use z to calculate other values :  w (weights ) and p ( linear weight * X values).    
       
   Final Loss is    the total variance over time of P  + Regualization.
   

2) We export all the paths, weights, into numpy files, 
   we can analyze convergence in Jupyter, ...
  



     
################## Things for improvement :

1) Implement Sequence to Sequence LSTM and test it




2) Regularization  with Graph in TF 2.0


3) Change the Loss



























#### AVX optimized       ###########################################
Wheel link: https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.13.1-py37-cpu-ivybridge/tensorflow-1.13.1-cp37-cp37m-linux_x86_64.whl

Install via:
pip install --no-cache-dir https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.13.1-py37-cpu-ivybridge/tensorflow-1.13.1-cp37-cp37m-linux_x86_64.while


pip install --ignore-installed --upgrade "Download URL" --user



https://github.com/inoryy/tensorflow-optimized-wheels


gitpod /workspace/control $ which gcc
/usr/bin/gcc
gitpod /workspace/control $ gcc --version
gcc (Ubuntu 8.3.0-6ubuntu1~18.10) 8.3.0



```




```
