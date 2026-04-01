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
from dustmaps.sfd import SFDQuery
from dustmaps.planck import PlanckGNILCQuery
from matplotlib.path import Path
from astropy.table import Table
from scipy.interpolate import interp1d
from itertools import product
import multiprocess as mp

#if using a new system, run this one time, but afterwards, you won't need 
#to continually import it to use dustmaps.sfd
#import dustmaps.sfd 
#dustmaps.sfd.fetch()



ast_cols = 
['ext','chip','X_in','Y_in','RA_in','Dec_in','X_out','Y_out','RA_out', 
'Dec_out',
             'chi_fit','snr_det','shp_det','rnd_det', 'dir', 'crow_det', 
'type','pass','cts_606','sky_1', 'rate_606', 
             
'rateerr_606','m606_in','m606_out','NOUSE_1','err_1','chi_1','snr_1','shp_1','rnd_1','crow_1', 
'ef_1',
             'cts_814', 'sky','rate_814', 'rateerr_814','m814_in', 
'm814_out','NOUSE', 'err', 'chi','snr', 'shp',
             'rnd', 'crow', 'ef']        

import warnings
warnings.filterwarnings("ignore", message=".*Signature.*longdouble.*")
