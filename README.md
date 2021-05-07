# Dimensionality reduction of genomic data in Python
## Andrea Estandia - University of Oxford

In this tutorial we will learn how to do a **Principal Component Analysis (PCA)** of genomic data. We will also learn how to use **UMAP** (Uniform Manifold Approximation and Projection for Dimension Reduction) using a VCF file.

**This tutorial is not finished and some things might not work. Please report any problem you have directly to *andrea.estandia@zoo.ox.ac.uk* ** 

### Set up
## Installation

1. Clone the repository:

```
 git clone https://github.com/andreaestandia/dim_reduction_genomics_tutorial.git
```

2. Navigate to main project folder, then run:

```
  conda env create --file umap.yml && conda activate umap
```

3. Install source code: `pip install .` (install) 

### Principal Component Analysis (PCA) using Scikit-allel

Let's crack on! 
What you're going to need for this to work: 

1. `Variant Call Format (VCF)` file. If you're searching for tutorial, it is very likely that you already have one. If you don't know how to get to this stage, please refer to https://samtools.github.io/hts-specs/VCFv4.2.pdf
2. A csv file with the names of the samples present in the VCF file and the populations where they belong to. The header should be called `sample` and `population`

To create the csv file you can extract the name of the samples by running in the terminal:

```
vcf-query -l myfile.vcf > list_names.txt
```
`vcf-query` is a module within VCFtools (https://vcftools.github.io/)

Now you have the names of your samples in `list_names.txt`. You can add the names of your populations in another column and add headers.

Now open a new file in a code editor (VScode is great), a Jupyter notebook or just a text file and save it as a .py file. For example, my file will be called `pca.py`. From now on you can forget about using the terminal, we will mainly be working on this file.
In your new .py file, paste the code below. These modules should have been installed beforehand. If you cloned the repo as indicated above, and then ran `pip install .` then you should already have all these modules.

#### Import your modules 
```
import bz2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pysam
import requests
import seaborn as sns
import vcf
import h5py
import allel
import umap.umap_ as umap
import umap.plot
import plotly.express as px
import datashader
import skimage
import colorcet
import holoviews
from mpl_toolkits.mplot3d import Axes3D 
from plotly.offline import init_notebook_mode, iplot
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
from matplotlib.lines import Line2D
import pandas_plink
warnings.filterwarnings('ignore')
```
Custom functions for plotting that will be useful afterwards
```
def selection_palette(labs, colors=['#e60000', '#0088ff'], rest_col='#636363', select_labels=['Bean Wood', 'Great Wood']):
    rest_col = sns.color_palette([rest_col])
    pal = sns.color_palette(colors)
    lab_dict = {lab: pal[i] for i, lab in enumerate(np.unique(select_labels))}
    for lab in np.unique(labs):
        if lab not in select_labels:
            lab_dict[lab] = rest_col[0]
    colors = np.array([lab_dict[i] for i in labs])
    return colors, lab_dict

def draw_umap(data, n_neighbors, min_dist, n_components, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data);
    fig = plt.figure()
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=colours, s=100)
    plt.title(title, fontsize=18)
```
#### VCF to HDF5
The first step is to extract data from a VCF file and save it as a HDF5 file
```
callset = allel.read_vcf('path/to/vcf_file.vcf')
allel.vcf_to_hdf5('path/to/vcf_file.vcf',
                  'path/to/hfd5_file.h5', fields='*', overwrite=True)
callset = h5py.File('path/to/hfd5_file.h5', mode='r')

```

```
calldata = callset['calldata']
list(calldata.keys())
genotypes = allel.GenotypeChunkedArray(calldata['GT'])
genotypes
```
#### Read csv file with info about samples and population
```
panel_path = 'path/to/csv_file.csv'
panel = pd.read_csv(panel_path, sep=',')
```

```
samples = callset['samples'][:]
np.all(samples == panel['sample'].values)
samples_list = list(samples)
samples_callset_index = [samples_list.index(s) for s in panel['sample']]
panel['callset_index'] = samples_callset_index
panel.head()
```

```
subpops = {population: panel[panel.population == population].index.tolist(
) for population in np.unique(panel.population.values.tolist())}

subpops['all'] = list(range(len(panel)))

ac_subpops = genotypes.count_alleles_subpops(subpops, max_allele=3) 

is_seg = ac_subpops['all'].is_segregating()[:]

genotypes_seg = genotypes.compress(is_seg, axis=0)

ac_seg = ac_subpops.compress(is_seg)

ac = ac_seg['all'][:]
ac
```
#### PCA
```
pca_selection = (ac.max_allele() == 1) & (ac[:, :2].min(axis=1) > 2)
np.count_nonzero(pca_selection)
indices = np.nonzero(pca_selection)[0]
indices_ds = np.random.choice(indices, size=7000, replace=False) #size can be replaced with any number of SNPs that you want to add in your PCA
indices_ds.sort()
genotypes_pca = genotypes_seg.take(indices_ds, axis=0)
gn = genotypes_pca.to_n_alt()[:]

coords, model = allel.pca(gn)
```
#### Plot PCA results 
```
pc1 = 0
pc2 = 1
x = coords[:, pc1]
y = coords[:, pc2]
labs = panel.population.values.tolist()
colours, lab_dict = make_cat_palette(labs, 'tab10')
fig, ax = plt.subplots(figsize=(10, 10))
plt.scatter(x,y,c=colours, s=10, label=labs)

legend_elements = [
    Line2D([0], [0], marker="o", linestyle="None", color=value, label=key)
    for key, value in lab_dict.items()
]
leg = ax.legend(handles=legend_elements, markerscale=1.3, labelspacing=1,
                facecolor='grey', edgecolor=None, framealpha=0.3,
                loc='upper left', fontsize=12
                )
leg.get_frame().set_linewidth(0.0)
for text in leg.get_texts():
    text.set_color('black')
```
#### Plot PCA results in 3D 

~~~
pc1 = 0
pc2 = 1
pc3 = 2
x = coords[:, pc1]
y = coords[:, pc2]
z = coords[:, pc3]

fig = px.scatter_3d(
    coords, x=0, y=1, z=2, color=panel.population, labels={'color': 'population'},
    color_discrete_sequence=px.colors.qualitative.Prism
)
fig.show()
~~~

#### Save PCA results in a csv file

```
PCA_df = pd.DataFrame([x, y, labs, samples_list]).T
PCA_df.columns = ['x', 'y', 'pop', 'sample']

PCA_df.to_csv('path/to/wherever/you/want/to/save/it/pca_results.csv', index=False)
```
### UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction

1. For using UMAP first we need to convert our VCF files to PED (it stands for pedigree format and it’s a format used by PLINK). 

```
vcftools --vcf my_data.vcf --out my_data --plink
```

2. Once you get the PED file you can convert it to binary PLINK format (BED).

~~~
plink --file my_data --make-bed
~~~

3. Read PLINK file, compute the genotypes, transpose the array and remove NAs

~~~
#Read PLINK file
snp_info,sample_info,genotypes  = pandas_plink.read_plink('/path/to/plink/file')
#Compute genotypes
genotype_mat = genotypes.compute()
#Transpose array
np_trans= np.transpose(genotype_mat)
#Remove NAs
genotype_mat_edited = np.nan_to_num(np_trans)
~~~

4. Run UMAP with 3 components. You can vary the number of neighbors and the minimum distance between them. This gives more weight to local or global structure depending on the combination you choose

~~~
umap_3d = umap.UMAP(n_components=3, n_neighbors=100, min_dist=0.2)

proj_3d = umap_3d.fit_transform(genotype_mat_edited)

fig = px.scatter_3d(
    proj_3d, x=0, y=1, z=2,
    color=panel.population, labels={'color': 'population'},
    color_discrete_sequence=px.colors.qualitative.Prism
)
fig.update_traces(marker_size=8)
fig.show()
~~~

