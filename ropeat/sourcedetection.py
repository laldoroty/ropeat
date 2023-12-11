# IMPORTS Standard:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# IMPORTS Astro:
from astropy.table import Table, hstack, join
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from astropy.visualization import ZScaleInterval
import sep

# IMPORTS Internal:
from .utils import get_obj_type_from_ID

###########################################################################

"""
This module works on one image at a time. You will input:
detect_sources: Numpy array for your science image, not background-subtracted
catalog_matching: The astropy table from detect_sources and an astropy table
                    with your catalog coordinates. 
"""

def detect_sources(scienceimage, byteswap=True):
    """_summary_

    Args:
        scienceimage (array-like): Array containing science image.
        byteswap (bool, optional): Toggle if required by sep. Defaults to True.

    Returns:
        astropy.table.Table: Table containing results from source detection. 
    """    
    # Byteswap because sep demands it. 
    if byteswap:
        img = scienceimage.byteswap(inplace=True).newbyteorder()
    
    # Background subtraction.
    bkg = sep.Background(img)
    img_sub = img - bkg

    # Object detection.
    # NOTE: In the future, mask the diffraction spikes. For now, this is
    # addressed with an ellipticity cut for ellipticity < 0.7.
    objects = sep.extract(img_sub, thresh=1.5, err=bkg.globalrms)
    objtab = Table(objects)
    objtab['ellipticity'] = 1 - objtab['b']/objtab['a']
    objtab = objtab[objtab['ellipticity'] < 0.7]

    return objtab

def plot_sources(scienceimage, objects):
    """Plot the ellipses describing the sources on top of the science image. 

    Args:
        scienceimage (array-like): Array containing science image.
        objects (astropy.table.Table): The table returned from detect_sources.
    """    
    zscale=ZScaleInterval()
    z1,z2 = zscale.get_limits(scienceimage)
    fig, ax = plt.subplots()
    plt.imshow(scienceimage,vmin=z1,vmax=z2,cmap='Greys',origin='lower')
    for i in range(len(objects)):
        e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                width=6*objects['a'][i],
                height=6*objects['b'][i],
                angle=objects['theta'][i] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)
    plt.xlabel('x [px]')
    plt.ylabel('y [px]')
    plt.colorbar()
    plt.show()

def catalog_matching(objtab, catalog, wcs):
    """Match the detected sources to the truth catalog. 

    Args:
        objtab (astropy.table.Table): Table with detected objects. Can be directly from output of 
                                        detect_sources(). Coordinates must be in pixel coordinates,
                                        and WCS must be supplied (see argument wcs). 
        catalog (astropy.table.Table): Table with catalog objects. Coordinates must be ra, dec in degrees. 
                                        There can be any number of columns, but it must contain:
                                        ['ra_truth', 'dec_truth']. If there are x and y coordinates, those
                                        should be 'x_truth' and 'y_truth'. Similarly, flux should be 'flux_truth'.
        wcs (astropy.wcs.WCS): Astropy WCS object for converting the pixel coordinates in objtab to ra, dec.

    Returns:
        astropy.table.Table: Merged objtab and catalog. 
    """    
    detected_coords = wcs.pixel_to_world(objtab['x'], objtab['y'])
    catalog_coords = SkyCoord(ra=catalog['ra_truth'], dec=catalog['dec_truth'], unit=(u.deg,u.deg))
    idx, sep2d, _ = match_coordinates_sky(detected_coords,catalog_coords)
    catalog['detection'] = 0
    catalog['detection'][idx] = 1
    detected_objects = hstack([catalog[idx], objtab])
    merged_tables = join(catalog,detected_objects,join_type='outer')
    merged_tables['objtype'] = list(map(get_obj_type_from_ID,merged_tables['object_id']))

    return merged_tables