#This file is meant to house my analysis functions so that I can call them ina  file and not have to repeat lines of code each time for only minor changes

#### imports
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
from scipy.interpolate import interp1d
from itertools import product
import multiprocess as mp

def extinction_correction(data_apt):
    '''
    This function takes ina  dat array and returns the extinction corrected version of the same array. Extinction correction here utilizes the `dustmaps.sfd` package in orde rot use the Schlegel, Finkbeiner & Davis (1998) maps and the Schlafly & Finkbeiner (2011) calibration. At present, the Av values are hard-coded for HST F606W and F814W.

    Parameters
    ----------

    data_apt : ndarray
    This is the input data file that we get from our photometry loaded in here. 

    Returns
    -------
    
    data_pd_clean : ndarray
    This is the same photometry arrya as before, but now the F606W adn f814W filters have been extinction corrected. 
    
    '''

    data_pd_clean = data_apt.copy()
    #import dustmaps.sfd #one-time usage, not needed after the first time to run this. 
    #dustmaps.sfd.fetch()
    from dustmaps.sfd import SFDQuery
    from dustmaps.planck import PlanckGNILCQuery

    ###---- Correct for Extinction and Reddening ----###
    stars_coords = SkyCoord(data_pd_clean['ra'], data_pd_clean['dec'], unit=(u.deg, u.deg), frame='icrs')

    ### Use Schlegel, Finkbeiner & Davis (1998) maps and Schlafly & Finkbeiner (2011) calibration
    sfd = SFDQuery()
    ebv_stars = sfd(stars_coords)

    ### Ab/E(B − V)_SFD coefficients from https://iopscience.iop.org/article/10.1088/0004-637X/737/2/103#apj398709app1
    ## Av = [2.488, 1.536] WFC3 F606W, F814W 
    Av = [2.471, 1.526] ## ACS/WFC F606W, F814W 

    #F606 correction
    data_pd_clean['acs_f606w_vega'] = data_pd_clean['acs_f606w_vega'] - Av[0] * ebv_stars 
    #F814 correction
    data_pd_clean['acs_f814w_vega'] = data_pd_clean['acs_f814w_vega'] - Av[1] * ebv_stars
    return data_pd_clean

#outputs = ['F606Wmag', 'F814Wmag'] # a a variable number of columns that reside in calling Parsec via ezpadova

### Start with mock_star_simple_pops
#from Markito in order to create a Mock Population --> mock_stars_simple_pops

def mock_stars_simple_pops(isoch, logage=10.00, met=-2.19174, mass=5e5,  minmass=0.55, 
                           outputs=['F606Wmag', 'F814Wmag', 'Mass', 'MH'], seed=None):
    """
    Generate a mock stellar population from a single isochrone or isochrone file. 
    This works by selecting the specific age, metallicty from our grid of isochrones, adjusting the 
    Integrated Initial Mass Function `Int_IMF` to the selected stellar mass, and then drawing the number of stars
    from this via `(Inverse Transform Sampling)`.
    
    This is meant for an astorpy table approach for the isochrone grid; `ezpadova` natively imports using pandas
    
    Parameters
    ----------
    
    isoch : astropy.table 
        Isochrone table from ezpadova or CDM website 
    logage : Float
        LogAge for a selected isochrone
    met : Float
        metallicity `[MH]` for a given isochrone
        since I am looking for the bluest possible stars, my defualt value is the PARSEC lowest of `-2.19174 [-2.2]`
    mass: float
        Set the total stellar mass that you want to sample your Int_IMF from 
        is what we are using to sample and interpolate the Int_IMF by. 
    minmass : float
        set's the minimum mass that our stars can have (acts as a filter for stars that won't reach the RGB)
    outputs: array_like, optional
            Sets the return columns that you want for the isochrones
            
    Returns
    -------
    
    outtab : astropy.table
        This is the newly sampled set of stars that come from the Int_IMF with the desired age, metallicity
    
    """

    np.random.seed(seed=seed) #set for reproducability

    #age
    ula = tb.unique(isoch, keys='logAge')['logAge'] #unique logage
    best_age_idx = np.argmin(abs(ula-logage))
    
    #metallicity
    umet = tb.unique(isoch, keys='MH')['MH'] ## unique metallicities
    ### should in principle interpolate between metallicities, because these are pretty sparse
    
    best_met_idx = np.argmin(abs(umet - met))
    to_use, = np.where( (isoch['logAge']==ula[best_age_idx]) 
                       #&(isoch['Zini']==umet[best_met_idx])
                        &(isoch['MH'] == umet[best_met_idx])
                       &(isoch['Mini'] > minmass) 
                       &(isoch['label'] < 9)
                        #&(isoch['label']==3)
                      )  
    #set this to our initial mass for our satellite 
    mwt = mass * isoch['int_IMF'][to_use]

    #for the number of stars    
    nstmin, nstmax = np.amin(mwt), np.amax(mwt)
    
    if nstmax - nstmin > 1.0:  #if n < 1, we get no stars
        nst = (nstmax - nstmin) * np.random.random(size=int(nstmax - nstmin)) + nstmin
        outtab = tb.Table()
        if len(outputs)>1:
            for i in outputs:
                tmp = tb.Column(np.interp(nst, mwt, isoch[i][to_use]), name=i)
                outtab.add_column(tmp)
    #print(ula)
    #print(best_age_idx)
    #print(umet)
    #print(best_met_idx)
    return outtab
####


###### Pandas version of teh same function ######

def mock_stars_simple_pops_pd(isoch, logage=10.00, met=-1.0, mass=5e5,  minmass=0.1, 
                           outputs=['F606Wmag', 'F814Wmag', 'Mass', 'MH'], seed=None):

    np.random.seed(seed=seed)

    #age (pandas)
    #ula = np.sort(isoch['logAge'].unique())
    ula = np.unique(isoch['logAge'])
    best_age_idx = np.argmin(abs(ula - logage))

    #metallicity (pandas)
    #umet = np.sort(isoch['MH'].unique())
    umet = np.unique(isoch['MH'])
    best_met_idx = np.argmin(abs(umet - met))

    # selection (pandas mask instead of np.where)
    mask = (
        (isoch['logAge'] == ula[best_age_idx]) 
        & (isoch['MH'] == umet[best_met_idx])
        & (isoch['Mini'] > minmass)
        & (isoch['label'] < 9)
    )
    to_use = isoch[mask] 

    # If nothing selected, return empty DF with requested columns
    #if to_use.empty:
    #    return pd.DataFrame(columns=outputs)

    #set this to our initial mass for our satellite 
    mwt = mass * to_use['int_IMF'] #.to_numpy()
    print(type(mwt))

    #for the number of stars    
    nstmin, nstmax = np.amin(mwt), np.amax(mwt)

    # always define outtab as a DataFrame
    outtab = pd.DataFrame(columns=outputs)

    if nstmax - nstmin > 1.0:
        nst = (nstmax - nstmin) * np.random.random(size=int(nstmax - nstmin)) + nstmin

    # sort by mwt for np.interp
    order = np.argsort(mwt)
    
    mwt_sorted = mwt.iloc[order].to_numpy()   # or: np.asarray(mwt)[order]

    for i in outputs:
        y_sorted = to_use[i].iloc[order].to_numpy()
        outtab[i] = np.interp(nst, mwt_sorted, y_sorted)
        
    return outtab

#### apply Distance Modulus ######

def apply_dist_mod(mock_df, dist_mod):
    """
    Adding distance modulus to the F606 & F814 columns of the array containing the sampled stars from the Int_IMF. 

    Parameters
    ----------

    mock_df : ndarray
    This is the array that should be the output of the `mock_stars_simple_pops` function

    dist_mod : float
    This is the calcualted distance modulus we get from work done during the TRGB luminosity fitting (not genreated from analysis in this pakage). 

    Returns
    -------

    new_df : ndarray
    This is the resulting ndarray after applying the distance modulus to our sampled stars. 
    
    """
    new_df = mock_df.copy()
    new_df['F606Wmag'] = new_df['F606Wmag'] + dist_mod #
    new_df['F814Wmag'] = new_df['F814Wmag'] + dist_mod #
    return new_df

def color_wrapper(F606W, F814W):
    '''
    This is a wrapper function to create the F606W and F814W color

    Parameters
    ----------

    F606W : ndarray
    This is the F606W array

    F814W : ndarray
    This is the F814W array

    Returns
    -------

    Color : ndarray
    returns the element-wise subtraction of each array to get the color. 
    
    '''
    return F606W.dropna() - F814W.dropna()

#### remove AST bad pixels ######

def clean_asts(ast_apt):
    '''
    This is a function used to remove the bad pixels from the DOLPHOT output files for the Artificial Star Tests
    
    Parameters
    ----------

    ast_apt : ndarray
    This is the input ndarray coming from the artifical star tests

    Returns
    -------
    
    ast_apt : ndarray
    This is the resulting ndarray after removing bad bad pixels

    '''
    
    ast_apt = ast_apt[np.where((ast_apt['m606_in'] < 99.0) & (ast_apt['m814_in'] < 99.0)
                           & (ast_apt['m606_out'] < 99.0) & (ast_apt['m814_out'] < 99.0))
                     ]
    return ast_apt
    
def median_pop(dataframe, notes = "", return_N = False):
    """
    Calculates and prints the median and number of stars in a data frame

    Parameters
    ----------

    dataframe : ndarray
    This holds the data that we want to compute the number of stars and its median. 

    notes : str
    This is to add in if this is the mock population, data color, or something else (added to the print statement)

    Returns
    -------

    len(dataframe): int, optional

    Optionally returns the length of the array (the number of stars). This will be utilized in generating the number statistics for the population to see if the observed number is in agreement with another population.
    """
    print(f"The number of stars in " + notes + f" is {len(dataframe)}.")
    print(f"The median of " + notes + f" is : {np.median(dataframe):.4f}")
    if return_N is not False:
        return len(dataframe)
    
    #print(f"number of stars in mock population: {len(mock_pop_color)}")
    #print(f"Mock median: {np.median(mock_pop_color):.4f}")
    

### AI rewrite ###
def cull_data(data_pd,
              snr_min_606=4.0,
              snr_min_814=4.0,
              err_max_606=3.0,
              err_max_814=3.0,
              sharp2_max_606=0.2,
              sharp2_max_814=0.2,
              crowd_max= 0.4,
              objtype_max=2,
              mag_max_606=99.0,
              mag_max_814=99.0, 
             Data=True):
    """
    Apply quality cuts to a photometry dataframe.

    Parameters
    ----------
    
    data_pd : pandas.DataFrame
        Input photometry table with ACS columns.
    snr_min_606, snr_min_814 : float
        Minimum S/N in F606W and F814W.
    err_max_606, err_max_814 : float
        Maximum magnitude error in F606W and F814W.
    sharp2_max_606, sharp2_max_814 : float
        Maximum sharp^2 in F606W and F814W.
    crowd_max_606, crowd_max_814 : float
        Maximum crowding in F606W and F814W.
    objtype_max : int
        Maximum allowed objtype_gl.
    mag_max_606, mag_max_814 : float
        Maximum allowed magnitudes (to reject 99.0 sentinel values).
    Data = True: Boolean
        Use the data culls if True
        Use the AST culls (same values, but the columns have different names) if False

    Returns
    -------
    
    data_apt_clean : pandas.DataFrame
        Masked/cleaned dataframe.
    """
    if Data: 
        data_mask = (
        (data_pd['acs_f606w_snr'] >= snr_min_606) &
        (data_pd['acs_f814w_snr'] >= snr_min_814) &
        (data_pd['acs_f606w_err'] <= err_max_606) &
        (data_pd['acs_f814w_err'] <= err_max_814) &
        (data_pd['acs_f606w_sharp']**2 < sharp2_max_606) &
        (data_pd['acs_f814w_sharp']**2 < sharp2_max_814) &
#        (data_pd['acs_f814w_crowd'] < crowd_max_814) &
#        (data_pd['acs_f606w_crowd'] < crowd_max_606) &
        (data_pd['acs_f606w_crowd'] + data_pd['acs_f814w_crowd'] <  crowd_max) &
        (data_pd['objtype_gl'] <= objtype_max) &
        (data_pd['acs_f606w_vega'] < mag_max_606) &
        (data_pd['acs_f814w_vega'] < mag_max_814)
        )

    else: 
        data_mask = (
            (data_pd['snr_1'] >= 4.0) & (data_pd['snr'] >= 4.0) &
            (data_pd['err_1'] <= 3.0) & (data_pd['err'] <= 3.0) &
            (data_pd['shp_1']**2 < 0.2) & (data_pd['shp']**2 < 0.2) &
            ((data_pd['crow'] + data_pd['crow_1']) < 0.4) &
            (data_pd['type'] <= 2)
            )
        
    clean_data = data_pd[data_mask]
    return clean_data

#### AST_completneess table + interpolating functions for magnitude and completion #######

#set 
def compute_ast_completeness_table(ast_apt, bin_width=0.1, mag_min = 22.0, mag_max=29.0, return_interp=True):
    """
    Computes AST completeness for both F606W and F814W bands.

    Parameters
    ----------
    
    ast_df : astropy.table.
        Artificial star test table with columns for magnitudes, SNR, errors, etc.
    bin_width : float
        Width of the magnitude bins.
    mag_min : float
    mag_max : float
        Upper magnitude bound for binning.
    return_interp : bool
        If True, returns inverse interpolation functions:
        interp_func(completeness_level) -> magnitude

    Returns
    -------
    
    completeness_table : astropy.table
        Table with magnitude bin centers, completeness, and median error for both F606W and F814W.
    interp_funcs : dict (optional)
        Dictionary of inverse interpolation functions for F606W and F814W.
    interp_mag_to_comp: dict (optional)
    """

    # Define shared bins and centers
    bins = np.arange(mag_min, mag_max + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width / 2

    bands = {
        'F606W': {'mag': 'm606_in'},
        'F814W': {'mag': 'm814_in'}
    }

    results = {
        'mag_bin_center': bin_centers,
        'bias_606': [], #use from TRGB
        'scatter_606': [], #use
        'bias_814': [], #use from TRGB
        'scatter_814': [] #use       
    }
    ###add bias, scatter

    #interpolation functions as dictionaries
    interp_funcs = {}
    
    interp_mag_to_comp = {}

    for band_name, band_info in bands.items(): #for F606W, for F814W
        mag_col = band_info['mag']
        completeness = []
        bias = []
        scatter = []

        for i in range(len(bins) - 1):
            #in_bin = (ast_df[mag_col] >= bins[i]) & (ast_df[mag_col] < bins[i+1])
            in_bin = (ast_apt[mag_col] >= bins[i]) & (ast_apt[mag_col] < bins[i+1])
            ast_inbin = ast_apt[in_bin]
            N_injected = len(ast_inbin)

            # recovered = (
            #     (ast_inbin['snr_1'] >= 4.0) & (ast_inbin['snr'] >= 4.0) & #originally > 4.0
            #     (ast_inbin['err_1'] <= 3.0) & (ast_inbin['err'] <= 3.0) &
            #     (ast_inbin['shp_1']**2 < 0.2) & (ast_inbin['shp']**2 < 0.2) & #originally 0.1
            #     ((ast_inbin['crow'] + ast_inbin['crow_1']) < 0.4) & #originally 0.3
            #     (ast_inbin['type'] <= 2)
            # )
            # N_recovered = np.sum(recovered)

            recovered = cull_data(ast_inbin, Data=False)
            N_recovered = len(recovered)
            
            completeness.append(N_recovered / N_injected if N_injected > 0. else 0.0)

            # Compute bias and scatter for magnitude residuals
            if N_recovered > 0:
                if band_name == 'F606W':
                #     delta_mag = ast_inbin['m606_out'][recovered] - ast_inbin['m606_in'][recovered]
                # else:
                #     delta_mag = ast_inbin['m814_out'][recovered] - ast_inbin['m814_in'][recovered]
                    delta_mag = recovered['m606_out'] - recovered['m606_in']
                else:
                    delta_mag = recovered['m814_out'] - recovered['m814_in']
            
                bias.append(np.median(delta_mag)) #follows MTRGB Notebook
                scatter.append(1.4826*np.median(np.abs(delta_mag - np.median(delta_mag)))) #follows MTRGB Notebook
            else:
                scatter.append(1e-8)
                bias.append(1e-8)

        #convert ot arrays for later use
        completeness = np.array(completeness)
        bias = np.array(bias)
        scatter = np.array(scatter)

        # Remove NaN and ensure monotonic decreasing completeness
        valid = ~np.isnan(completeness)

        #use for magnitude interpolation in --> Completeness out
        mag_sorted = bin_centers[valid]
        comp_sorted = completeness[valid]

        #mag_sort2 = bin_centers[valid] #added to check for monotonicness
        #comp_sort2 = completeness[valid] #added to check for monotonicness

        # Sort by completeness (descending)
        #use for completeness in --> magnitude out
        sort_idx = np.argsort(comp_sorted)[::-1] #commented out to see fi this helps
        comp_sorted2 = comp_sorted[sort_idx]
        mag_sorted2 = mag_sorted[sort_idx]

        if band_name == 'F606W':
            results['completeness_606'] = completeness
            results['bias_606'] = bias
            results['scatter_606'] = scatter
        else:
            results['completeness_814'] = completeness
            results['bias_814'] = bias
            results['scatter_814'] = scatter

        if return_interp:
            
            # Remove duplicate completeness values for inverse mapping #added for deduping
            comp_u, idx_u = np.unique(comp_sorted2, return_index=True)
            mag_u = mag_sorted2[idx_u]

            # Inverse: completeness → magnitude
            inv_interp_func = interp1d(
                comp_u, #originally comp_sorted2
                mag_u, #originally mag_sorted2
                bounds_error=False,
                #fill_value=(mag_sorted2[-1], mag_sorted2[0]) originally here
                fill_value=(mag_u[-1], mag_u[0])
            )
            interp_funcs[band_name] = inv_interp_func
            
            # #return_interp:
            # Inverse: magnitude → completeness

            comp_sorted = np.clip(comp_sorted, 0.0, 1.0)
            # enforce monotonic non-increasing with magnitude
            comp_sorted = np.maximum.accumulate(comp_sorted[::-1])[::-1]

            interp_func2 = interp1d(
                mag_sorted,
                comp_sorted,
                bounds_error=False,
                #fill_value=(mag_sorted[-1], mag_sorted[0])
                fill_value=(1.0, 0.0)
            )
            interp_mag_to_comp[band_name] = interp_func2

    # Create Astropy Table
    completeness_table = Table()
    for key, val in results.items():
        completeness_table[key] = val

    if return_interp:
        return completeness_table, interp_funcs, interp_mag_to_comp
        #return completeness_table, interp_funcs, interp_mag_to_comp, comp_sorted, mag_sorted, mag_sort2, comp_sort2
    else:
        return completeness_table


##### Plot AST Completeness #####

def plot_ast_completeness(completeness_tbl, dwarf):
    '''
    This function takes in the compelteness table (output array from the `compute_ast_completeness_table` function) and makes a basic plot to show the completeness plotted against magnitude

    Parameters
    ----------
    
    completeness_tbl : ndarray
    This is the resulting ndarray that includes the binned completeness, magnitude bin centers, binned bias, and binned scatter values. In this function, we only make use of `binned completeness` and ` magnitude bin centers` for both F606W and F814W

    curr_dwarf : Dwarf object
    This is meant to pull in the name of the target system instead of hard docing it. 
    '''
    
    plt.figure(figsize=(10, 5))

    plt.plot(completeness_tbl['mag_bin_center'], completeness_tbl['completeness_606'],
             label='F606W', color='blue')
    plt.plot(completeness_tbl['mag_bin_center'], completeness_tbl['completeness_814'],
             label='F814W', color='red')

    #plt.axhline(0.5, color='gray', linestyle='--', lw=1, label='80% completion')
    #plt.axhline(0.8, color='gray', linestyle=':', lw=1, label='50% completion')

    plt.xlabel("Input Magnitude")
    plt.ylabel("Completeness")
    plt.title(dwarf.name + " Completeness")
    plt.ylim(0, 1.2)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

### define the bias and scatter functions ####

#outputs for my interpolation functions should be magnitudes in F606, F814
def bias_and_scatter(ast_completeness_tbl):
    '''
    This function takes int he resulting ast_completeness table from the `compute_ast_completeness_table` function, and returns 4 interpolating functions; 1 for the bias as a function of magnitude, and one for the scatter as a function of magnitude each for F606W and F814W. This function also makes use of the scipy `interp1d` function. 

    Parameters
    ----------
    completeness_tbl : ndarray
    This is the resulting ndarray that includes the binned completeness, magnitude bin centers, binned bias, and binned scatter values. In this function, we only make use of `binned bias` and `binned scatter` for both F606W and F814W. 

    Returns
    -------
    bias_interp_814 :  scipy.interpolate._interpolate.interp1d
    This is the interpolation function for the F814W bias
    
    scatter_interp_814 :  scipy.interpolate._interpolate.interp1d
    This is the interpolation function for the F814W scatter
    
    bias_interp_606 :  scipy.interpolate._interpolate.interp1d
    This is the interpolation function for the F606W bias
    
    scatter_interp_606 :  scipy.interpolate._interpolate.interp1d
    This is the interpolation function for the F606W scatter
    '''
    bin_centers = ast_completeness_tbl['mag_bin_center']
    bias_interp_606 = interp1d(
                bin_centers,
                ast_completeness_tbl['bias_606'],
                bounds_error=False, #allows for extrapolation or not
                fill_value=(0.001, 1e-10) #what to extrapolate to
            )

    bias_interp_814 = interp1d(
                bin_centers,
                ast_completeness_tbl['bias_814'],
                bounds_error=False, #allows for extrapolation or not
                fill_value=(0.001, 1e-10) #what to extrapolate to
            )

    scatter_interp_606 = interp1d(
                bin_centers,
                ast_completeness_tbl['scatter_606'],
                bounds_error=False, #allows for extrapolation or not
                fill_value=(0.001, 1e-10) #what to extrapolate to
            )

    scatter_interp_814 = interp1d(
                bin_centers,
                ast_completeness_tbl['scatter_814'],
                bounds_error=False, #allows for extrapolation or not
                fill_value=(0.001, 1e-10) #what to extrapolate to
            )
    return bias_interp_814, scatter_interp_606, bias_interp_606, scatter_interp_814

##### Aplt AST Observational Affects ####

def apply_ast_probabilistic_interp_df(
    mock_df,
    interp_mag_to_comp, #comp_interp_814,
    bias_interp_814, scatter_interp_814,
    bias_interp_606, scatter_interp_606,
    seed=None
    ):
    rng = np.random.default_rng(seed)

    out = mock_df.copy(deep=True)

    # Ensure output columns exist
    out['F814W_obs'] = np.nan
    out['F606W_obs'] = np.nan

    m814 = out['F814Wmag'].to_numpy()
    m606 = out['F606Wmag'].to_numpy()

    # Completeness as function of m814
    comp = np.asarray(comp_interp_814(m814), dtype=float)
    comp = np.clip(comp, 0.0, 1.0)

    keep = rng.random(len(out)) <= comp

    # Bias/scatter as a function of magnitude
    b814 = np.asarray(bias_interp_814(m814), dtype=float)
    s814 = np.asarray(scatter_interp_814(m814), dtype=float)
    b606 = np.asarray(bias_interp_606(m814), dtype=float)      # often keyed on m814; if you prefer m606, swap input
    s606 = np.asarray(scatter_interp_606(m814), dtype=float)

    # draw noise only for kept stars
    n814 = rng.normal(loc=b814, scale=s814, size=len(out))
    n606 = rng.normal(loc=b606, scale=s606, size=len(out))

    out.loc[keep, 'F814W_obs'] = m814[keep] + n814[keep]
    out.loc[keep, 'F606W_obs'] = m606[keep] + n606[keep]

    return out

### fast version of Apply_AST_probabilistic ###

def apply_ast_probabilistic_fast(mock_df, interp_mag_to_comp,  bias_interp_814, scatter_interp_814, 
                                 bias_interp_606, scatter_interp_606, seed=None): #interp_mag_to_compwas originally comp814_interp,
    #set the numpy random seed for reproducability 
    np.random.seed(seed=seed)
    #copy the data (apparently weird things can happen if you modify the original dataframe
    cmd_realistic = mock_df.copy().reset_index(drop=True)
        
    #access only the magnitudes from the pandas dataframe
    m814_true = cmd_realistic['F814Wmag'].values
    m606_true = cmd_realistic['F606Wmag'].values
    
    #create empty columns of length n for pre-allocating; these will have values changed based on surviving stars. 
    n = len(cmd_realistic)
    f814_obs = np.full(n, np.nan)
    f606_obs = np.full(n, np.nan)
    
    #detmeriners for keeping values or changing them
    u814 = np.random.uniform(0., 1., size=n) #uniform draw
    c814 = interp_mag_to_comp['F814W'](m814_true) #814 completeness for each star

    #generate all the bias, scatter values; using the interpolating functions returns a numpy array
    b814 = bias_interp_814(m814_true)
    b606 = bias_interp_606(m606_true)
    
    s814 = scatter_interp_814(m814_true)
    s606 = scatter_interp_606(m606_true)
    
    #need element-wise comparison for the scatter
    min_814_scatter = np.maximum(s814, 1e-8)
    min_606_scatter = np.maximum (s606, 1.e-8)
    noise814 = np.random.normal(loc=b814, scale=min_814_scatter)
    noise606 = np.random.normal(loc=b606, scale=min_606_scatter)

    # Observed magnitudes = true mag + bias & scatter draw
    m814_obs = m814_true + noise814
    m606_obs = m606_true + noise606
 
    #now, I need to compare the draw vs. completeness
    keep = (u814 <= c814)

    #filter by completeness values on which ones to keep, and which ones to reject. 
    f814_obs[keep] = m814_obs[keep]
    f606_obs[keep] = m606_obs[keep]

    #Add the new columns to the Pandas dataframe
    cmd_realistic['F814W_obs'] = f814_obs
    cmd_realistic['F606W_obs'] = f606_obs

    return cmd_realistic


##### Make a mock population Data Frame (most complete 2/4) ###########

#def make_and_eval_mock_pop(fixed_age, met): #original function: (params):
#def make_and_eval_mock_pop(fixed_age, met, log_mass = 6.98, smooth=False, weight = 1.0): #original function: (params):
def make_and_eval_mock_pop(isoch, fixed_age, met, dmod=29.74, log_mass=6.98, interp_mag_to_comp=None, bias_interp_814=None,
    scatter_interp_814=None, bias_interp_606=None, scatter_interp_606=None, smooth=False, smooth_factor = 1.e1, weight=1.0, seed=None):

    """
    create and evaluate my mock population for a given age, metallicity
    """
    #fixed_age, met = params
    print(f"Running for logAge = {fixed_age}, MH = {met}")
    seed = 3
    try:
        if seed is not None:
            seed_iter = int(seed + round(fixed_age*1000) + round((met+10)*10000)) #generate a seed that is reporducable
        else:
            seed_iter = None
        
        # 1. Generate mock population
        outputs = ['F606Wmag', 'F814Wmag'] #can add more from isoch_table columns
        
        # base mass scaled by weight
        base_mass = 10**log_mass * (1./0.55) * weight

        # if smooth=True, increase sampling mass by 10**2 in order to smooth out the CDF
        mass_used = base_mass * (smooth_factor if smooth else 1.0)
        
        mock_df = mock_stars_simple_pops_pd(
            isoch, logage=fixed_age, met=met,
            mass=mass_used, minmass=0.55, outputs=outputs, seed=seed_iter
            )
        
        #3 Apply distance modulus
        dist_mod  = 29.87
        mock_pop_dmod = apply_dist_mod(mock_df, dist_mod)
        
        #mock_obs = apply_ast_probabilistic_interp(mock_pop_dmod, 
        #                             interp_mag_to_comp['F814W'], bias_interp_814, scatter_interp_814, 
        #                             bias_interp_606, scatter_interp_606, seed=3)

        mock_obs = apply_ast_probabilistic_fast(mock_pop_dmod, 
                                     interp_mag_to_comp, bias_interp_814, scatter_interp_814, 
                                     bias_interp_606, scatter_interp_606, seed=seed_iter)
        # 5. Compute observed color (drop unrecovered)
        mock_color_df = (mock_obs['F606W_obs'] - mock_obs['F814W_obs']).dropna()

        return mock_color_df

    except Exception as e:
        print(f"Error at age={fixed_age}, met={met}: {e}")
        # return empty series to match expected return type
        return pd.Series(dtype=float)

#def make_and_eval_mock_pop(fixed_age, met): #original function: (params): ###returns the dataframe, so I can manipulate it directly
#def make_and_eval_mock_pop2(isoch, fixed_age, met, dmod = 29.74, log_mass = 6.98, smooth=False, weight = 1.0): 

def make_and_eval_mock_pop2(isoch, fixed_age, met, dmod=29.74, log_mass=6.98, interp_mag_to_comp=None, bias_interp_814=None,
    scatter_interp_814=None, bias_interp_606=None, scatter_interp_606=None, smooth=False, smooth_factor=1.e1, weight=1.0, seed=None):

    """
    create and evaluate my mock population for a given age, metallicity. Reutrns a pandas dataframe
    """
    #fixed_age, met = params
    print(f"Running for logAge = {fixed_age}, MH = {met}")
    #seed = 3
    try:
        if seed is not None:
            seed_iter = int(seed + round(fixed_age*1000) + round((met+10)*10000)) #generate a seed that is reporducable
        else:
            seed_iter = None
        
        # 1. Generate mock population
        outputs = ['F606Wmag', 'F814Wmag'] #can add more from isoch_table columns
        
        # base mass scaled by weight
        base_mass = 10**log_mass * (1./0.55) * weight

        # if smooth=True, increase sampling mass by 10**2 in order to smooth out the CDF
        mass_used = base_mass * (smooth_factor if smooth else 1.0)
        
        mock_df = mock_stars_simple_pops_pd(
            isoch, logage=fixed_age, met=met,
            mass=mass_used, minmass=0.55, outputs=outputs, seed=seed_iter
            )
        
        #3 Apply distance modulus
        dist_mod  = dmod
        mock_pop_dmod = apply_dist_mod(mock_df, dist_mod)
        
        #mock_obs = apply_ast_probabilistic_interp(mock_pop_dmod, 
        #                             interp_mag_to_comp['F814W'], bias_interp_814, scatter_interp_814, 
        #                             bias_interp_606, scatter_interp_606, seed=3)

        mock_obs = apply_ast_probabilistic_fast(mock_pop_dmod, 
                                     interp_mag_to_comp, bias_interp_814, scatter_interp_814, 
                                     bias_interp_606, scatter_interp_606, seed=seed_iter)
        #previously had the seed=3 here
        

        # 5. Compute observed color (drop unrecovered)
        return mock_obs
        #mock_color_df = (mock_obs['F606W_obs'] - mock_obs['F814W_obs']).dropna()

        #return mock_color_df

    except Exception as e:
        print(f"Error at age={fixed_age}, met={met}: {e}")
        # return empty series to match expected return type
        return pd.Series(dtype=float)

def ks2_samp(sample1, sample2, direction='two-sided'):
    s1 = np.asarray(sample1)
    s2 = np.asarray(sample2)

    #remove NaNs
    mask1 = np.isfinite(s1); mask2 = np.isfinite(s2)
    s1_clean = s1[mask1]
    s2_clean = s2[mask2]
    
    D, p = stats.ks_2samp(s1_clean, s2_clean, alternative=direction)
    print(f'D_KS = {D:.3e} & P_KS = {p:.3e}')
    return D, p


#### Running One-One to find the best metallicity for the satellite  ######

def run_one(params):
    age, met = params
    print(f"Running for logAge = {age}, MH = {met}")
    seed = 3
    try:
        seed_iter = int(seed + round(age*1000) + round((met+10)*10000))
        # 1. Generate mock population
        mock_pop_apt = mock_stars_simple_pops(
            isoch_table, logage=age, met=met,
            mass=10**6.65*(1./0.55), minmass=0.5, outputs=outputs, seed=seed_iter
        )
        #mass = 10**6.65*(1./0.55)
        
        # 2. Convert to pandas for AST application
        mock_df = mock_pop_apt.to_pandas()
        
        #3 Apply distance modulus
        mock_pop_dmod = apply_dist_mod(mock_df, dist_mod)
        
        #mock_pop_with_dmod_df
        # 4. Apply your current AST completeness function
        mock_obs = apply_ast_probabilistic_interp(mock_pop_dmod, 
                                     interp_mag_to_comp['F814W'], bias_interp_814, scatter_interp_814, 
                                     bias_interp_606, scatter_interp_606,
                                   seed=3)


        # 5. Compute observed color (drop unrecovered)
        mock_color_df = (mock_obs['F606W_obs'] - mock_obs['F814W_obs']).dropna()
        #data_color_apt = (data_apt_clean['acs_f606w_vega'] - data_apt_clean['acs_f814w_vega'])
        #data_color_apt = (f606_rgb - f814_rgb)


        #print(len(mock_color_df))
        #print(len(data_color_apt))
        
        # 6. Run KS test
        #D, p = stats.ks_2samp(data_color_apt, mock_color_df)
        D, p = stats.ks_2samp(data_color_np, mock_color_df)

        # Return results tuple
        return (age, met, D, p)

    except Exception as e:
        # Handle unexpected errors without crashing pool
        print(f"Error at age={age}, met={met}: {e}")
        return (age, met, np.nan, np.nan)

    #     #5
    #     mock_color_df = (mock_obs['F606W_obs'] - mock_obs['F814W_obs']).dropna()
    #     data_color_apt = np.asarray(f606_rgb - f814_rgb, dtype=float)
    #     data_color_apt = data_color_apt[np.isfinite(data_color_apt)]
        
    #     n_data = len(data_color_apt)
    #     n_mock = len(mock_color_df)

    #     if (n_data < 2) or (n_mock < 2):
    #         # return counts so you can see what's failing
    #         return (age, met, np.nan, np.nan, n_data, n_mock)
    #     #6
    #     D, p = stats.ks_2samp(data_color_apt, mock_color_df)
    #     return (age, met, D, p, n_data, n_mock)

    # except Exception as e:
    #     print(f"Error at age={age}, met={met}: {e}")
    #     return (age, met, np.nan, np.nan, -1, -1)

# ---------------------------
# ECDF-on-grid helper (pure NumPy)
# ---------------------------
def ecdf_on_grid(sorted_sample, x_grid):
    n = len(sorted_sample)
    counts = np.searchsorted(sorted_sample, x_grid, side="right")
    return counts / n

### Color spread ###

def color_spread_weighting(
    isoch,
    data_color,
    interp_mag_to_comp,
    bias_interp_814,
    scatter_interp_814,
    bias_interp_606,
    scatter_interp_606,
    dmod,
    log_mass,
    fixed_age=9.9,
    mets=[-2.19174, -0.5],
    smooth=False,
    smooth_factor=1.e1,
    seed=None
):
    start_time = time.time()

    blue_color = make_and_eval_mock_pop(
        isoch, fixed_age, mets[0],
        dmod=dmod,
        log_mass=log_mass,
        interp_mag_to_comp=interp_mag_to_comp,
        bias_interp_814=bias_interp_814,
        scatter_interp_814=scatter_interp_814,
        bias_interp_606=bias_interp_606,
        scatter_interp_606=scatter_interp_606,
        smooth=smooth,
        smooth_factor=smooth_factor,
        weight=1.0,
        seed=seed
    )

    red_color = make_and_eval_mock_pop(
        isoch, fixed_age, mets[1],
        dmod=dmod,
        log_mass=log_mass,
        interp_mag_to_comp=interp_mag_to_comp,
        bias_interp_814=bias_interp_814,
        scatter_interp_814=scatter_interp_814,
        bias_interp_606=bias_interp_606,
        scatter_interp_606=scatter_interp_606,
        smooth=smooth,
        smooth_factor=smooth_factor,
        weight=1.0,
        seed=seed
    )

    # ---- Clean + sort samples ----
    blue_stars = np.sort(np.asarray(blue_color))
    red_stars  = np.sort(np.asarray(red_color))
    data_stars = np.sort(np.asarray(data_color))

    blue_stars = blue_stars[np.isfinite(blue_stars)]
    red_stars  = red_stars[np.isfinite(red_stars)]
    data_stars = data_stars[np.isfinite(data_stars)]

    if len(blue_stars) < 2 or len(red_stars) < 2 or len(data_stars) < 2:
        raise ValueError("One of your samples has <2 finite values. Cannot build ECDFs reliably.")

    x_grid = data_stars

    F_blue = ecdf_on_grid(blue_stars, x_grid)
    F_red  = ecdf_on_grid(red_stars, x_grid)
    F_data_on_grid = ecdf_on_grid(data_stars, x_grid)

    weights = np.arange(0.0, 1.01, 0.01)
    w_list, D_list, p_list = [], [], []

    for w in weights:
        F_mix = w * F_blue + (1.0 - w) * F_red

        Fmix_callable = interp1d(
            x_grid, F_mix,
            bounds_error=False,
            fill_value=(0.0, 1.0),
            assume_sorted=True
        )

        D, p = stats.kstest(
            data_stars,
            Fmix_callable,
            alternative="two-sided",
            mode="asymp"
        )

        w_list.append(w)
        D_list.append(D)
        p_list.append(p)

    res = pd.DataFrame({
        "weights": w_list,
        "KS_D_1samp": D_list,
        "KS_p_1samp": p_list
    })

    best_idx = int(np.argmin(res["KS_D_1samp"].values))
    w_best = float(res.loc[best_idx, "weights"])
    D_best = float(res.loc[best_idx, "KS_D_1samp"])
    p_best = float(res.loc[best_idx, "KS_p_1samp"])

    print("Best weight (w_best):", w_best)
    print("Best 1-sample KS D:", D_best)
    print("1-sample KS p-value at w_best:", p_best)

    plt.figure(figsize=(7, 4))
    plt.plot(res["weights"], res["KS_D_1samp"], lw=2)
    plt.xlabel("w (fraction of BLUE component)")
    plt.ylabel("1-sample KS D")
    plt.title("1-sample KS distance vs mixture weight")
    plt.grid(alpha=0.25)
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(res["weights"], res["KS_p_1samp"], lw=2)
    plt.xlabel("w (fraction of BLUE component)")
    plt.ylabel("1-sample KS p-value")
    plt.yscale("log")
    plt.title("1-sample KS p-value vs mixture weight")
    plt.grid(alpha=0.25)
    plt.show()

    F_mix_best = w_best * F_blue + (1.0 - w_best) * F_red

    plt.figure(figsize=(8, 6))
    plt.plot(x_grid, F_data_on_grid, "k-", lw=2.5, label="Data ECDF (on data grid)")
    plt.plot(x_grid, F_mix_best, "r-", lw=2.5, label=f"Best mixture CDF (w={w_best:.2f})")
    plt.plot(x_grid, F_blue, "--", lw=2, label="Blue CDF (on data grid)")
    plt.plot(x_grid, F_red, "--", lw=2, label="Red CDF (on data grid)")
    plt.xlabel("Color (F606W - F814W)")
    plt.ylabel("CDF")
    plt.title("Best-fit mixture CDF vs data")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.show()

    print('execution time = %.3f seconds' % (time.time() - start_time))
    print('execution time = %.3f minutes' % ((time.time() - start_time) / 60.0))

    return w_best

def build_final_weighted_sample(
    isoch,
    fixed_age,
    mets,
    w_best,
    dmod,
    log_mass,
    interp_mag_to_comp,
    bias_interp_814,
    scatter_interp_814,
    bias_interp_606,
    scatter_interp_606,
    smooth=False,
    smooth_factor=1.e1,
    seed=None
):
    blue_weight = w_best
    red_weight  = 1.0 - w_best

    sample_parts = []

    # build blue only if needed
    if blue_weight > 0.0:
        blue_sample = make_and_eval_mock_pop(
            isoch, fixed_age, mets[0],
            dmod=dmod,
            log_mass=log_mass,
            interp_mag_to_comp=interp_mag_to_comp,
            bias_interp_814=bias_interp_814,
            scatter_interp_814=scatter_interp_814,
            bias_interp_606=bias_interp_606,
            scatter_interp_606=scatter_interp_606,
            smooth=smooth,
            smooth_factor=smooth_factor,
            weight=blue_weight,
            seed=seed
        )

        blue_sample = np.asarray(blue_sample)
        blue_sample = blue_sample[np.isfinite(blue_sample)]

        if len(blue_sample) > 0:
            sample_parts.append(blue_sample)

    # build red only if needed
    if red_weight > 0.0:
        red_sample = make_and_eval_mock_pop(
            isoch, fixed_age, mets[1],
            dmod=dmod,
            log_mass=log_mass,
            interp_mag_to_comp=interp_mag_to_comp,
            bias_interp_814=bias_interp_814,
            scatter_interp_814=scatter_interp_814,
            bias_interp_606=bias_interp_606,
            scatter_interp_606=scatter_interp_606,
            smooth=smooth,
            smooth_factor=smooth_factor,
            weight=red_weight,
            seed=seed
        )

        red_sample = np.asarray(red_sample)
        red_sample = red_sample[np.isfinite(red_sample)]

        if len(red_sample) > 0:
            sample_parts.append(red_sample)

    if len(sample_parts) == 0:
        return np.array([])

    final_sample = np.concatenate(sample_parts)
    return np.sort(final_sample)


###### Running KDE  #########

# #import iqr for the inner quartile range
# from scipy.stats import iqr

# #Upgraded Epanechnikov KDE
# def adapt_hist_Epanechnikov(data, spacing, bandwidth='silverman'):
#     # Silverman's rule by default
#     if isinstance(bandwidth, str):
#         if bandwidth.lower() == 'silverman':
#             sigma = np.std(data, ddof=1)
#             n = len(data)
#             data_iqr = iqr(data)/1.34
#             # 1.06 * std * n**(-1/5)
#             bandwidth = 0.9*min(sigma, data_iqr)*len(data)**(-0.2)
#         else:
#             raise ValueError(f"Please amend input")

#     # Epanechnikov Kernel KDE
#     density = np.zeros(len(spacing))
#     for idx, xi in enumerate(spacing):
#         for value in data:    
#             u = (xi - value) / bandwidth
#             if np.abs(u) <= 1.0:
#                 density[idx] += 0.75 * (1 - u**2) / bandwidth

#     return (1 / len(data)) * density

##### return the Epanechnikov kernel ########
# #set the spacing intervals
# spacing_color = np.linspace(-0.5, 3.0, 2000)

# good = (~data_color.mask) & np.isfinite(data_color)          # keeps only unmasked + finite (no NaN/inf)
# col_clean = data_color[good]

# #data_color_pd = data_color_pd.dropna()
# mock_pop_color = (mock_pop_with_obs_df['F606W_obs'] - mock_pop_with_obs_df['F814W_obs']).dropna()

# #Metallicity KDE
# RGB_kde = adapt_hist_Epanechnikov(col_clean, spacing_color, 'silverman')
# mock_pop_kde = adapt_hist_Epanechnikov(mock_pop_color, spacing_color, 'silverman')

# #create a 1x2 plot for both quantities
# fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(12, 6))

# #plot Metallicity --> #plot the spacing vs. the KDE, not the array itself
# ax.plot(spacing_color, RGB_kde, color='steelblue', label='TRGB ')
# ax.plot(spacing_color, mock_pop_kde, color='red', label='Mock Population')
# ax.set_xlabel('color')
# ax.set_xlim(-0.5, 3.0)
# ax.set_ylabel('Normalized counts')
# ax.set_title('Epanechnikov Kernel Density Estimator for TRGB, Mock Population')
# ax.legend()

# plt.tight_layout()
# plt.show()
