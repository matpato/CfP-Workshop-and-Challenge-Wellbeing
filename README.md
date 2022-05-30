# [CfP-Workshop-and-Challenge-Wellbeing 2022](https://github.com/matpato/CfP-Workshop-and-Challenge-Wellbeing)


<p align="center">
<img src="https://lasigebiotm.github.io/RecSys.Scifi/assets/img/recommender-bg.jpg" width="50%" height="50%">
</p>


Competition at the [EMBC 2022 Challenge on Detection of Stress and Mental Health Using Wearable Sensors](https://compwell.rice.edu/workshops/embc2022)
May and June, 2022


## Table of Contents
  * [0. Prerequirements](#0-prerequirements)
  * [1. Download the repository](#1-download-the-repository)
  * [2. Preparation](#2-preparation)
  * [3. Open the challenge](#3-open-the-challenge)
  * [4. Close and Shutdown Jupyter Notebook](#4-close-and-shutdown-jupyter-notebook)
  * [5. Remove the created virtual environment](#5-remove-the-created-virtual-environment)
   

## 0. Prerequirements

- OS: Ubuntu 18.04 LTS or higher, OS X

- Python 3.8

- Browser (Firefox, Safari or Google Chrome)


## 1. Download the repository

a) With git clone:

```
git clone git@github.com:matpato/CfP-Workshop-and-Challenge-Wellbeing.git

cd CfP/
```

b) By retrieving directly the repository zip file 

```
wget https://github.com/matpato/CfP-Workshop-and-Challenge-Wellbeing

cd CfP-Workshop-and-Challenge-Wellbeing/
```



## 2. Preparation

a) **If [Anaconda](https://docs.conda.io/en/latest/miniconda.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed**:

- Create virtual environment

```
conda create --name cfp_wellbeing_isel python=3.8
```

- Activate the created virtual environment:

```
conda activate cfp_wellbeing_isel
```

- Install dependencies:

```
python3.8 -m pip install -r requirements.txt
```

- Add the virtual environment to Jupyter:

```
ipython kernel install --user --name=cfp_wellbeing_isel
```


b) **With [venv](https://docs.python.org/3/library/venv.html) module**:

- Install venv package and create virtual environment:

```
sudo apt install python3.8-venv

python3.8 -m venv cfp_wellbeing_isel
```

- Activate the created virtual environment:

```
source cfp_wellbeing_isel/bin/activate 
```

- Install dependencies:

```
python3.8 -m pip install -r requirements.txt
```

- Add the virtual environment to Jupyter:

```
ipython kernel install --user --name=cfp_wellbeing_isel
```


## 3. Open the challenge

- To start the Notebook Server run the following command:

```
jupyter notebook
```

Which will open http://localhost:8888/notebooks/ in the default browser.

- Open the file 'challenge.ipynb'.

- Click on 'Kernel' > 'Change kernel' > 'cfp_wellbeing_isel' to ensure the challenge is running on the created virtual environment.


## 4. Close and Shutdown Jupyter Notebook

**a) Shutdown Jupyter Notebook Files from the dashboard**

- In the Kernel Sessions tab, click on 'SHUTDOWN' for the appropriate notebook to terminate the session for that notebook.

or

- In the File Browser tab and selecting 'Kernel' > 'Shutdown Kernel'.


**b) Shutdown the Jupyter Notebook Local Server**

You can also close your terminal by typing the command ```exit``` and hitting Enter.

## 5. Remove the created virtual environment

**a) Anaconda or Miniconda**


- Deactivate the created virtual environment:

```
conda deactivate
```

- Remove the virtual environment and all packages:

```
conda env remove --name cfp_wellbeing_isel
```


**b) venv**


- Deactivate the created virtual environment:

```
deactivate
```

- Remove the virtual environment and all packages:

```
rm -r cfp_wellbeing_isel

jupyter kernelspec uninstall cfp_wellbeing_isel
```
