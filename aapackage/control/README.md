# [Deep BSDE Solver](https://arxiv.org/abs/1707.02568) 


## Training

```
python3 main.py --problem=SquareGradient


#### Working Install     ###########################################
pip3 install tensorflow==1.13.1
pip3 install scipy





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
Copyright (C) 2018 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Package              Version
-------------------- -------
absl-py              0.7.1
astor                0.7.1
gast                 0.2.2
grpcio               1.20.1
h5py                 2.9.0
Keras-Applications   1.0.7
Keras-Preprocessing  1.0.9
Markdown             3.1
mock                 3.0.5
numpy                1.16.3
protobuf             3.7.1
scipy                1.2.1
tensorboard          1.13.1
tensorflow           1.13.1
tensorflow-estimator 1.13.0
termcolor            1.1.0
Werkzeug             0.15.4
wheel                0.33.4


```

Command-line flags:

* `problem_name`: Name of partial differential equation (PDE) to solve.
There are seven PDEs implemented so far. See [Problems](#problems) section below.
* `num_run`: Number of experiments to repeatedly run for the same problem.
* `log_dir`: Directory to write event logs and output array.


## Problems

`equation.py` and `config.py` now support the following problems:

* `AllenCahn`: Allen-Cahn equation with a cubic nonlinearity.
* `HJB`: Hamilton-Jacobi-Bellman (HJB) equation.
* `PricingOption`: Nonlinear Black-Scholes equation for the pricing of European financial derivatives
with different interest rates for borrowing and lending.
* `PricingDefaultRisk`: Nonlinear Black-Scholes equation with default risk in consideration.
* `BurgesType`: Multidimensional Burgers-type PDEs with explicit solution.
* `QuadraticGradients`: An example PDE with quadratically growing derivatives and an explicit solution.
* `ReactionDiffusion`: Time-dependent reaction-diffusion-type example PDE with oscillating explicit solutions.



