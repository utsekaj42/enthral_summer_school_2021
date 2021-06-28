# Enthral Summer School Days 1 (1D arterial models) and 3 (UQSA) 
_NOTE_ This is not the final set of files, so remember to check for updates closer to the course.


The course content is organized into Jupyter Notebooks which require a Jupyter Server to run the code and provide an interface accessible through the web browser.

The Jupyter Server environment must provide a Python kernel (3.7 or newer) and the required modules: Required modules: numpy, scipy, sympy, numba, pandas, statsmodels, matplotlib, chaospy (4.3.2), ipywidgets, networkx, openpyxl, h5py, lxml, graphviz.

We will make a preconfigured Jupyter server available for the course, so participants should be able to use these notbooks through any up-to-date webbrowser.

To run these notebooks on your own device, you will need to create a Python environment with the required dependencies. The most reliable way to accomplish this is by using the Anaconda Python distribtuion and the `environment.yml` file provided that specifies the required modules. (More detailed instructions about this are provided below)

# Setup and usage
First, you need to have a Python environment with the required dependencies. These are specified in the `environment.yml` file, which can be automatically processed by the `conda` package manager provided by Anaconda (See <https://www.anaconda.com/download/>). The steps outlined below assume one is using a terminal where `conda` is on the path, but the steps should be achievable in the Anaconda Navigator as well.
<https://docs.anaconda.com/anaconda/user-guide/getting-started/>

# Installation
Assuming `conda` is installed then

```
conda env create --file=environment.yml 
```
should install dependencies in the environment `uqsa_tut` which is the name specified in `environment.yml`. If you prefer to name the environment something else you can specify it with the `-n` option:
```
conda env create --file=environment.yml  -n NAME # Creates an environment named NAME
``` 

# Starting Jupyter-Notebook in this envrionment
You can now start a `jupyter-notebook` server by:
1. opening a terminal
2. change the current directory to where you have downloaded these files 
3. activating your environment 
4. and then running the command `jupyter-notebook` 
```
cd enthral_summer_school_2021
conda activate uqsa_tut
jupyter-notebook
```
Now keep the terminal open as closing it will stop the server process. It should automatically open your web browser at `localhost:8888`, but if not you should see the address with an access token printed out in the terminal.

The home page should show a list of files in the directory where the `jupyter-notebook` server is running. _Note_ if you have started the server in another directory, you will not be able to access directories above the initial directory. Select the notebook `index.ipynb` in the same directory as this `README.md`.


An alternative is to use the graphical menu of the Anaconda Navigator <https://docs.anaconda.com/anaconda/user-guide/getting-started/>, and select the `uqsa_tut` environment then open the Jupyter-Notebook app (Screen shots to follow).




# Registering the kernel
If you have an existing installation of `Jupyter` or `conda` you may find it easiest to run the Jupyter server in your root environment and just register your new environment as a kernel. <https://github.com/Anaconda-Platform/nb_conda_kernels>

