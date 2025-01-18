## Setup python venv
(if working on jupyterhub environment where home directory is erased with every new instance)

https://si-apc.pages.in2p3.fr/computer-science-crash-course/pyenv/pyenv-intro.html 
```
# create venv
python -m venv venv

# activate venv
source venv/bin/activate
```

## make the pyblur directory inside this repository the default one
```
cd pyblur
pip install -e .
cd ..
```

## install other packages
first do: `pip install cmake`  
Then repeat `python src/tools/generate_synthetic_data.py` and install the missing packages  

results of this lengthy proces should now be saved in venv. On next use just activate venv and start working instantly



## Current working environment:

```
pip list
Package                Version     Editable project location
---------------------- ----------- --------------------------------------------------------------------
cmake                  3.31.4
colorama               0.4.6
contourpy              1.3.1
cycler                 0.12.1
dill                   0.3.9
fonttools              4.55.3
fpie                   0.2.4
imageio                2.36.1
kiwisolver             1.4.8
lazy_loader            0.4
llvmlite               0.43.0
markdown-it-py         3.0.0
matplotlib             3.10.0
mdurl                  0.1.2
networkx               3.4.2
numba                  0.60.0
numpy                  2.0.2
opencv-python          4.11.0.86
opencv-python-headless 4.11.0.86
packaging              24.2
pillow                 11.1.0
pip                    24.0
pyblur                 0.2.5       /project_ghent/luversmi/attempt2/synthetic-dataset-generation/pyblur
Pygments               2.19.1
pyparsing              3.2.1
python-dateutil        2.9.0.post0
rich                   13.9.4
scikit-image           0.25.0
scipy                  1.15.1
setuptools             65.5.0
six                    1.17.0
taichi                 1.7.3
tifffile               2025.1.10
tqdm                   4.67.1
```