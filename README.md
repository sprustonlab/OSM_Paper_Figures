# Code for the paper 'Learning produces a hippocampal cognitive map in the form of an orthogonalized state machine'

# Quick start

## Browsing the data
> 📘 **Additional tutorials available**
>
> Extended notebook tutorials on how to read and manipulate the data is available in the vr2p module [here](https://github.com/sprustonlab/vr2p-public/tree/main/notebooks/tutorials).

See below on how to get the relevant data files to run the code.

### Creating the vr2p object
This object is the main access point to the data of a single animal and is used extensively throughout the code.
```python
data = vr2p.ExperimentData('../Set-A-E/Tyche-A2')
```

### Accessing fluorescence data
Access the fluorescence information of a single cell.
```python
# registered fluorescence data
F = data.signals.multi_session.F
# single cell data
cell = 100
session = 3
single_cell_f = F[session][cell,:]
```

### Field-of-view images
See the FOV of a single session before and after registration.
```python
# fov of season 0 before registration
session=0
original_fov = data.images.original[session]['mean_img']
# and after registration.
registered_fov = data.images.registered[session]['mean_img']
```

### Cell mask information
Plot the mask roi of a single cell (overlaid on top of the FOV image)
```python
import numpy as np
%matplotlib inline
fig, ax = plt.subplots(1,1,figsize=(10,10))

# get FOV image.
fov_img = data.images.registered[0]['mean_img']
ax.imshow(fov_img,interpolation='none',cmap='gray',aspect=1.5) # correct non uniform aspect ratio.

# create masks overlay image.
cell_mask = data.cells.multi_session.registered[0]
mask_img = np.zeros(fov_img.shape)
mask_img = np.ma.masked_where(mask_img ==0 , mask_img) # mask zero values.
mask_img[cell_mask['ypix'],cell_mask['xpix']]=1
ax.imshow(mask_img,interpolation='none',cmap='jet',aspect=1.5,vmin=0,vmax=2,alpha=0.75)
```


# Data availability
All data needed to run the notebooks in this repository is available on figshare: https://doi.org/10.25378/janelia.27273552

Please note that the data is available as compressed archives and need to be unpacked before they can be used. Files with a number as a extension (e.g. 001) are part of a split archive and need to be unpacked together.

# Dependencies
## Main dependency
Needed to run the notebooks in this repo.
* [vr2p](https://github.com/sprustonlab/vr2p-public) module. The main way to access and manipulate the joined VR and 2-Photon calcium imaging data. 
## Auxillary dependencies.
Modules and code used to generate the data.
* [Gimbl](https://github.com/winnubstj/Gimbl).  VR package for Unity to perform VR animal experiments.
* [gimbl-2acdc](https://github.com/sprustonlab/gimbl-2acdc). Gimbl related files to run the 2ACDC behavior task.
* [2ACDC_parse](https://github.com/sprustonlab/2ACDC_parse). Adds additional meta data relevant to the 2ACDC task.
* [multiday-suite2p](https://github.com/sprustonlab/multiday-suite2p-public). Performs the multi-day/session alignment on [suite2p](https://github.com/MouseLand/suite2p) exported files. Some utility function from this repository are used in a small number of notebooks.
