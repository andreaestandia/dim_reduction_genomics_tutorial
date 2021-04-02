# Principal component analysis of genomic data in Python
## Andrea Estandia - University of Oxford
### Create a Python environment
In order to avoid potential conflicts with other packages it is strongly recommended to use a conda environment. Here you can learn how to create and activate one and how to install modules on it.
Open the terminal and paste the line below.
The `-n` flag allows you to choose a name for your environment. You can also choose the python version that you want.
```
conda create -n name-env python=3.6
```
### Activate your environment 
Activating your environment is essential to making the software in the environments work well. In the terminal, paste the line below to activate your environment. 
```
conda activate name-env
```

You can learn more about environments here (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

### Install modules in your environment
Once your environment has been activated you can install modules like this:
```
conda install scikit-learn
```
This is just an example. You should replace `scikit-learn` with the modules you need

### 

Let's crack on! 
What you're going to need for this to work: 
1. `Variant Call Format (VCF)` file. If you're searching for this, it is very likely that you already have one. If you don't, please refer to https://samtools.github.io/hts-specs/VCFv4.2.pdf
2. A csv file with the names of the samples present in the VCF file and the populations where they belong to.
