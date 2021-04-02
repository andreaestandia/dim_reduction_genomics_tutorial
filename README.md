# Principal component analysis of genomic data using Scikit-allel
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
This is just an example. You should replace `scikit-learn` with anything you 

# %%
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
#import umap
from plotly.offline import init_notebook_mode, iplot
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')

# %%
def make_cat_palette(labs, color_palette):
    pal = sns.color_palette(color_palette, n_colors=len(np.unique(labs)))
    lab_dict = {lab: pal[i] for i, lab in enumerate(np.unique(labs))}
    colors = np.array([lab_dict[i] for i in labs])
    return colors, lab_dict


def selection_palette(labs, colors=['#e60000', '#0088ff'], rest_col='#636363', select_labels=['Bean Wood', 'Great Wood']):
    rest_col = sns.color_palette([rest_col])
    pal = sns.color_palette(colors)
    lab_dict = {lab: pal[i] for i, lab in enumerate(np.unique(select_labels))}
    for lab in np.unique(labs):
        if lab not in select_labels:
            lab_dict[lab] = rest_col[0]
    colors = np.array([lab_dict[i] for i in labs])
    return colors, lab_dict
# %%
# %%
# The first step is to extract data from a VCF file and save it as a HDF5 file
callset = allel.read_vcf(
    '/home/zoo/sjoh4959/sjoh4959/projects/0.0_phylo_silvereye/data/vcf/mel.vcf')
allel.vcf_to_hdf5('/home/zoo/sjoh4959/sjoh4959/projects/0.0_phylo_silvereye/data/vcf/mel.vcf',
                  'mel.h5', fields='*', overwrite=True)
callset = h5py.File('mel.h5', mode='r')

# %%
calldata = callset['calldata']
list(calldata.keys())
genotypes = allel.GenotypeChunkedArray(calldata['GT'])
genotypes

# %%
panel_path = '/home/zoo/sjoh4959/sjoh4959/projects/0.0_phylo_silvereye/data/mel.csv'
panel = pd.read_csv(panel_path, sep=',')
# %%

samples = callset['samples'][:]
np.all(samples == panel['sample'].values)
samples_list = list(samples)
samples_callset_index = [samples_list.index(s) for s in panel['sample']]
panel['callset_index'] = samples_callset_index
panel.head()
# %%

subpops = {population: panel[panel.population == population].index.tolist(
) for population in np.unique(panel.population.values.tolist())}

subpops['all'] = list(range(len(panel)))

ac_subpops = genotypes.count_alleles_subpops(subpops, max_allele=3) 

is_seg = ac_subpops['all'].is_segregating()[:]

genotypes_seg = genotypes.compress(is_seg, axis=0)

ac_seg = ac_subpops.compress(is_seg)

###
###Example for each pop. Unmute if useful and substitute 'NSW' with pop name
###
#is_biallelic_01 = ac_seg['all'].is_biallelic_01()[:]
#ac_nsw = ac_seg['NSW'].compress(is_biallelic_01, axis=0)[:, :2]

#All pops
ac = ac_seg['all'][:]
ac

# %%


pca_selection = (ac.max_allele() == 1) & (ac[:, :2].min(axis=1) > 2)
np.count_nonzero(pca_selection)
indices = np.nonzero(pca_selection)[0]
indices_ds = np.random.choice(indices, size=7000, replace=False)
indices_ds.sort()
genotypes_pca = genotypes_seg.take(indices_ds, axis=0)
gn = genotypes_pca.to_n_alt()[:]

coords, model = allel.pca(gn)

# %%

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

plt.savefig('pca_plot_mel.pdf')  

#%%

PCA_df = pd.DataFrame([x, y, labs, samples_list]).T
PCA_df.columns = ['x', 'y', 'pop', 'sample']

PCA_df.to_csv('/home/zoo/sjoh4959/sjoh4959/projects/0.0_phylo_silvereye/mel.csv', index=False)

# %%

jsfs = allel.joint_sfs(ac_seg['Tanna'][:, 1], ac_seg['Gaua'][:, 1])
fig, ax = plt.subplots(figsize=(6, 6))
allel.plot_joint_sfs(jsfs, ax=ax)
ax.set_ylabel('Alternate allele count, Tanna')
ax.set_xlabel('Alternate allele count, Gaua')
