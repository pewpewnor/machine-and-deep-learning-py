# Machine and Deep Learning with Python

A collection of hands-on machine-learning examples and small utilities. The repository covers perceptrons and Adaline, sigmoid neural networks, a composite autoencoder example, a PyTorch classifier for Fashion-MNIST, Lasso regression, feature scaling, and a loader for the original MNIST IDX files.

The examples are independent scripts rather than a single application or packaged library. Some use included sample datasets; the PyTorch examples can download Fashion-MNIST through torchvision.

## Stack

Python, NumPy, pandas, Matplotlib, scikit-learn, PyTorch, and torchvision.

## Set up

Use Python 3 and create a virtual environment:

~~~sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn torch torchvision
~~~

On Windows, activate the environment with `.venv\\Scripts\\activate`. Choose the PyTorch install command that matches your CPU or CUDA setup from the official PyTorch selector.

## Run an example

Run commands from the repository root so the scripts can find their local data and modules:

~~~sh
python -m deep_learning.perceptron.main_iris
python -m deep_learning.pytorch.main_cnn
python -m machine_learning.regularlization.lasso_r1
~~~

The Fashion-MNIST script downloads its dataset on first run. The Iris and MNIST examples expect the corresponding files under data/. Some scripts open Matplotlib figures or train models directly when run.
