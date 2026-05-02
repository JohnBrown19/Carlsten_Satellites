#Imports and data Loading
from astropy.io import ascii
import numpy as np
import pandas as pd
import ezpadova #isochrones
import matplotlib.pyplot as plt
from scipy import stats
import time
import astropy
from astropy.table import Table
from astropy.coordinates import SkyCoord #match_coordinates_sky, Angle 
import astropy.units as u 
import astropy.table as tb
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import interp1d
from itertools import product
import multiprocess as mp

#import from other file
from .Analysis import cull_data, extinction_correction

class Dwarf(object):
    #Variables assigned here are global for all Dwarf objects
    ast_cols = ['ext','chip','X_in','Y_in','RA_in','Dec_in','X_out','Y_out','RA_out', 
                'Dec_out', 'chi_fit','snr_det','shp_det','rnd_det', 'dir', 'crow_det', 
                'type','pass','cts_606','sky_1', 'rate_606','rateerr_606','m606_in', 
                'm606_out','NOUSE_1','err_1', 'chi_1','snr_1','shp_1','rnd_1','crow_1','ef_1',
                 'cts_814', 'sky','rate_814', 'rateerr_814','m814_in','m814_out','NOUSE', 
                'err', 'chi','snr', 'shp','rnd', 'crow', 'ef'
               ]
    
    #variables assigned here are for instances (attributes), and may differ with each Dwarf. 
    #you access attributes using object.attribute
    def __init__(self, name, dmod, logmass, Host = None):
        self.name = str.upper(name)
        self.dmod = float(dmod)
        self.logmass = float(logmass)
        self.Host = Host
        self.asts = None
        self.data = None
        self.data_masked = None
        self.mask=None
        self.data_clean = None
        self.ast_clean = None
        self._data_mask = None #(optional, for debugging)
        self._ast_mask = None
        
    #below here are methods
    #you access attributes using object.attribute()
    def load_asts(self):
        self.asts = ascii.read("./17797/merged-results/17797_" + self.name 
                                + "_fakestars.dat", names=self.ast_cols)
        self.sats_type = type(self.asts)
        print(f"AST data for {self.name} loaded sucessfully")

    def load_data(self):
        self.data = pd.read_hdf("./17797/fake-results2/17797_" + self.name 
                                + "/proc_default_deepCR/17797_" + self.name 
                                + ".phot_full.hdf5", 
                                key='data')
        print(f"Data for {self.name} loaded sucessfully")

    # def load_data_masked(self): #defunct function? 
    #     self.data_masked = pd.read_hdf("./17797/fake-results2/17797_" + self.name 
    #                             + "/proc_default_deepCR/17797_" + self.name 
    #                             + ".phot_full_masked.hdf5", 
    #                             key='data')
    #     print(f"Masked data for {self.name} loaded sucessfully")

    def load_mask(self): #defunct function? 
        mask_path = "./17797/fake-results2/17797_" + self.name + "/proc_default_deepCR/17797_" + self.name + "_mask.fits"
        with fits.open(mask_path) as hdul:
            # choose the HDU that actually holds the mask array; 0 or 1 are common
            # If you’re unsure, print([i.data.shape for i in hdul]) and pick the one with 2D data
            mask_hdu = hdul[0] if hdul[0].data is not None else hdul[1]
            #mask_data = np.asarray(mask_hdu.data)
            self.wcs_mask = WCS(mask_hdu.header)
            self.mask = np.asarray(mask_hdu.data)
        print(f"Masks for {self.name} loaded sucessfully")

    def clean_data(self):
        #1. Apply the photometric culls, extinction correction
        self.data = cull_data(self.data) #apply photometric cuts as part of stap 2
        self.data = extinction_correction(self.data) #wasn't in the file, but I need to do it anyway. 
        print(f"Applied the photometric culls, extinction correction to {self.name}")

    def apply_mask(self):
        #1. Apply the photometric culls, extinction correction
        #self.data = cull_data(self.data) #apply photometric cuts as part of stap 2
        #self.data = extinction_correction(self.data) #wasn;t in the file, but I need to do it anyway. 
        
        #2. Now to apply filters --> First to the culled data
        ra = np.asarray(self.data['ra'])
        dec = np.asarray(self.data['dec'])
        pix = self.wcs_mask.all_world2pix(np.column_stack([ra, dec]), 0)   # shape (N,2)
        xpix = pix[:,0]
        ypix = pix[:,1]
        print('Recovered X, Y pixels from the Photometry')

        #3. Keep only sources that land inside the image bounds
        ny, nx = self.mask.shape
        in_bounds = (xpix >= 0) & (xpix < nx) & (ypix >= 0) & (ypix < ny)
        print('Found all valid pixel locations')

        # 4) Sample the mask at integer pixel indices (choose rounding mode)
        #floor is common; you can also use np.rint for nearest-neighbor
        xi = np.floor(xpix[in_bounds]).astype(int)
        yi = np.floor(ypix[in_bounds]).astype(int)
        print('Apply pixel limits from .fits file!')

        # 5) Define what "masked" means. Here: non-zero == BAD
        bad = np.zeros_like(in_bounds, dtype=bool) #create an array of True Values
        bad[in_bounds] = self.mask[yi, xi] == 0   # flip to ==0 if your mask encodes “good==1”
        print('Apply `Masked = 1` to the dataset')

        # 6) Final keep mask = not bad (and in bounds). If out-of-bounds, you can choose to drop or keep.
        keep = in_bounds & (~bad)
        #self.data = self.data[keep]
        self.data_masked = self.data.loc[keep].copy()
        print('Applied masks to dataset')
        
    def data_type(self):
        self.data_type = type(self.data)
        return print(self.data_type)

    def asts_type(self):
        self.asts_type = type(self.asts)
        return print(self.asts_type)

    def pd_to_apt(self):
        #turn a pandas table to an Astropy Table
        self.data = Table.from_pandas(self.data)
        #return print(self.data_type)
        
    def apt_to_pd(self):
        self.data = self.data.to_pandas()