#imports
import pandas as pd
from astropy.io import ascii

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

    def plot_ast(self):
        pass

    def plot_data(self):
        pass

    def extinction_correction(self):
        from dustmaps.sfd import SFDQuery
        from dustmaps.planck import PlanckGNILCQuery

        ###---- Correct for Extinction and Reddening ----###
        stars_coords = SkyCoord(self.data['ra'], self.data['dec'], unit=(u.deg, u.deg), frame='icrs')

        ### Use Schlegel, Finkbeiner & Davis (1998) maps and Schlafly & Finkbeiner (2011) calibration
        sfd = SFDQuery()
        ebv_stars = sfd(stars_coords)

        ### Ab/E(B − V)_SFD coefficients from https://iopscience.iop.org/article/10.1088/0004-637X/737/2/103#apj398709app1
        ## Av = [2.488, 1.536] WFC3 F606W, F814W 
        Av = [2.471, 1.526] ## ACS/WFC F606W, F814W 

        #F606 correction
        self.data['acs_f606w_vega'] = self.data['acs_f606w_vega'] - Av[0] * ebv_stars 
        #F814 correction
        self.data['acs_f814w_vega'] = self.data['acs_f814w_vega'] - Av[1] * ebv_stars