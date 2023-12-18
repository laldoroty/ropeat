# ropeat
Basic package for Roman photometry, associated with Aldoroty et al. 2024 in prep. 

This package is modular. Modules are compatible with each other, and they interface
easily, but you do not need to use all the modules. It contains:
```
sourcedetection.py
objectidentification.py
photometry.py
```

You'll need:
- Your science image
- The truth file corresponding to that science image
- WCS information 

Usage example:
```
# Standard imports
import numpy as np
import pandas as pd

# Astro imports
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

# This package
from ropeat.sourcedetection import detect_sources, plot_sources, catalog_matching
from ropeat.photometry import ap_phot

imgpath = '/path/to/science/image.fits.gz'
truthpath = '/path/to/truth/file.txt'

hdu = fits.open(imgpath)
wcs = WCS(hdu[1].header)
img = hdu[1].data

objects = detect_sources(img)
# plot_sources(img, objects)

# Note: column names in your truth object table must be distinct from the names of the output columns,
# and the suffix must be '_truth'. e.g. don't put 'x' in your truth object table, put 'x_truth'.
truth_colnames = ['object_id', 'ra_truth', 'dec_truth', 'x_truth', 'y_truth', 'realized_flux', 'flux_truth', 'mag_truth']
truthread = pd.read_csv(truthpath, skipinitialspace=True, names=truth_colnames, sep=' ', comment='#')
cat = Table(Table.from_pandas(truthread), masked=True)

matched_catalog = catalog_matching(objects, cat, wcs)
detected_coords = matched_catalog[matched_catalog['detection'] == 1]

apphot = ap_phot(img, detected_coords)

```