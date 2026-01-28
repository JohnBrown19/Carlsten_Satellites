class Dwarf(object):
    #Variables assigned here are global for all Dwarf objects
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
    
    #variables assigned here are for instances (attributes), and may 
differ with each Dwarf. 
    #you access attributes using object.attribute
    def __init__(self, name, dist_mod, logmass, Host = None):
        self.name = str.upper(name)
        self.dist_mod = float(dist_mod)
        self.logmass = float(logmass)
        self.Host = Host
        self.asts = None
        self.data = None
        

    #below here are methods
    #you access attributes using object.attribute()
    def load_asts(self):
        self.asts = ascii.read("./17797/merged-results/17797_" + self.name 
+ "_fakestars.dat", names=self.ast_cols)
        self.sats_type = type(self.asts)
        print(f"AST data for {self.name} loaded sucessfully")

    def load_data(self):
        self.data = pd.read_hdf("./17797/fake-results2/17797_" + self.name 
+ "/proc_default_deepCR/17797_" + self.name + ".phot_full.hdf5", 
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

    
