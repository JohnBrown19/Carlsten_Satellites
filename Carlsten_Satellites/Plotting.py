### this file will contain the major plotting functions used, as well as some 

#### General Plotting to find the RGB
import matplotlib.pyplot as plt 
from matplotlib.path import Path
from matplotlib import patches
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.ticker as ticker


def Find_TRGB(data_apt_clean, dwarf, box_coords=None, output=False):
    '''
    This is a variable function used to find the approximate region for the TRGB. It also returns the 
    f606_rgb, f814_rgb, ra_rgb, and dec_rgb arrays for data manipulation and plotting to compare to 
    our mock populations later

    Parameters
    ----------
    
    data_apt_clean : ndarray
    This is the input dataframe or astropy table that is generated after extinction correction

    dwarf :  object(Dwarf)
    This is the dwarf object that our data is based on. 
    This is getting in mostly so I can name the plot appropriately

    box_coords : numpy array of tuple coordinates, optional
    This paraemter takes in a tuple of points [(ra1, dec1), (ra2, dec2), etc.] that is used to isolate the 
    RGB overdensity.

    output : bool, optional
    This is a boolean flag that doesn't return the rgb arrays (False, no returns; True, return f606_rgb and f814_rgb)
    
    Returns
    -------
    f606_rgb : ndarray
    This is the F606W magnitude of our RGB stars. Will get used later for our mock population comparison
    
    f814_rgb : ndarray
    This is the F814W magnitude of our RGB stars. Will get used later for our mock population comparison
    
    ra_rgb : ndarray
    This is used for the plot to ensure that the correctly-selected region is used for TRGB selection. 

    dec_rgb : ndarray
    This is used for the plot to ensure that the correctly-selected region is used for TRGB selection. 
    
    Notes
    -----
    I considered putting a user input argument so that I could more iteratively adjust the TRGB box coords, but I 
    didn't look into it at the moment. 
    
    '''

    #create ra, dec object to filter by
    ra = data_apt_clean['ra']; dec = data_apt_clean['dec']
    
    if box_coords is not None:
        #create the points around the 
        TRGB_box = Path(box_coords)
        plt.gca().add_patch(patches.PathPatch(TRGB_box, fc='none', ec='red', lw=1.4, linestyle = '-.', zorder=1,
                                alpha=0.9))
        #filter the region
        in_CMD_box = (TRGB_box.contains_points(np.vstack([ra, dec]).T) )

        #unfiltered version
        pts_sources_rgb = data_apt_clean[(in_CMD_box)]

        #fitler arrays for Ra, Dec, F606, F814
        f606_rgb = pts_sources_rgb['acs_f606w_vega']
        f814_rgb = pts_sources_rgb['acs_f814w_vega']
        ra_rgb = pts_sources_rgb['ra']
        dec_rgb = pts_sources_rgb['dec']

        #Photometry and RGB masked plots
        
        plt.scatter(ra_rgb, dec_rgb, alpha=0.9, color = 'red', s = 6, label = 'Isolated RGB')
        


    #plots
    #plt.scatter(data_apt['ra'], data_apt['dec'], alpha=0.5, label = 'Raw data')
    plt.scatter(data_apt_clean['ra'], data_apt_clean['dec'], color='black', s=6, alpha=0.5, label = 'Photometry masked')
    plt.gca().invert_xaxis()
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.legend(loc='best')
    plt.title(dwarf.name, size=16)
    plt.show()

    if output is True:
        return f606_rgb, f814_rgb #, ra_rgb, dec_rgb

def simple_cmd(data_apt_clean, f606_rgb, f814_rgb, ymin=22.0, ymax=28.0, xmin=-1.5, xmax=2.5):
    #real CMD
    plt.scatter(data_apt_clean['acs_f606w_vega'] - data_apt_clean['acs_f814w_vega'], 
            data_apt_clean['acs_f814w_vega'], c='blue', alpha=0.8, label='Observed Data', s=10)

    #Isolated RGB region
    plt.scatter(f606_rgb - f814_rgb, f814_rgb, c='red', alpha = 0.8, label='Selected Region', s= 10)

    #plot
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    plt.gca().invert_yaxis()
    plt.xlabel('F606W - F814W')
    plt.ylabel('F814W')
    plt.title('CMD for Observed Data vs. Mock Population')
    plt.legend(loc='best')
    plt.show()

def kde_2d(data_apt_clean, kind='kde', cmap='Blues'):
    '''
    Used to show the 2d Kernel Density Estimator for the potential TRGB. This should line up relatively 
    with the position of the satellite, and helps to identify the slected box to draw. 

    '''
    x = data_apt_clean['ra']
    y = data_apt_clean['dec']
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=x, y=y, cmap=cmap, cbar=True)
    plt.title('2D KDE Plot for Overdensities')
    plt.gca().invert_xaxis()
    plt.xticks(rotation=45)
    plt.show()
    sns.jointplot(x=x, y=y, kind=kind, fill=True, cmap='Blues')
    plt.xticks(rotation=45)
    plt.gca().invert_xaxis()
    plt.show()
    
#def cmd_compare(f606_rgb, f814_rgb, mock_pop_with_obs_df, isoch, age=fixed_age, met=mets, 
#                dmod=dmod, ymin = 22.0, ymax=28.0, xmin = -1.0, xmax = 2.5, name=dwarf):

def cmd_compare(f606_rgb, f814_rgb, mock_pop_with_obs_df, isoch, age, mets, dwarf,
                ymin = 22.0, ymax=28.0, xmin = -1.0, xmax = 2.5):
    #real CMD
    plt.scatter(f606_rgb - f814_rgb, f814_rgb, c='blue', label='Selected Region')

    #mock CMD
    plt.scatter(mock_pop_with_obs_df['F606W_obs'] - mock_pop_with_obs_df['F814W_obs'], mock_pop_with_obs_df['F814W_obs'], 
            alpha=0.5, c='red', label = 'Mock Population')
    #isochrone
    cmdfilt = (isoch['logAge'] == age) & (isoch['MH'] == mets[0]) & (isoch['label'] < 9)
    w = isoch[cmdfilt]
    cmdfilt2 = (isoch['logAge'] == age) & (isoch['MH'] == mets[1]) & (isoch['label'] < 9)
    w2 = isoch[cmdfilt2]

    plt.scatter(w['F606Wmag'] - w['F814Wmag'], w['F814Wmag'] + dwarf.dmod, color='k', alpha = 0.9, 
            label = f'logage = {age}, [M/H] = {mets[0]}')
    plt.scatter(w2['F606Wmag'] - w2['F814Wmag'], w2['F814Wmag'] + dwarf.dmod, color='purple', alpha = 0.9, 
            label = f'logage = {age}, [M/H] = {mets[1]}')

    #plot
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    plt.gca().invert_yaxis()
    plt.xlabel('F606W - F814W')
    plt.ylabel('F814W')
    plt.title('CMD for ' + dwarf.name)
    plt.legend(loc='best')
    plt.show()

def ks2_samp(data_color, mock_pop_color, direction='two-sided'):
    s1 = np.asarray(data_color)
    s2 = np.asarray(mock_pop_color)

    #remove NaNs
    mask1 = np.isfinite(s1); mask2 = np.isfinite(s2)
    s1_clean = s1[mask1]
    s2_clean = s2[mask2]
    
    D, p = stats.ks_2samp(s1_clean, s2_clean, alternative=direction)
    print(f'D_KS = {D:.3e} & P_KS = {p:.3e}')
    return D, p
    
def cdf_compare(data_color, mock_pop_color, dwarf):
    '''
    Compare the CDF of my data against my mock_population using the 2 sample ks-test

    Parameters
    ----------

    Returns
    -------
    '''
    #run 2-sample ks test
    D, p = stats.ks_2samp(data_color, mock_pop_color)

    #set_up the ecdf to plot like in the documentation
    ax = plt.subplot()

    #data
    ecdf_input = stats.ecdf(data_color)

    #mock pop
    ecdf_output = stats.ecdf(mock_pop_color)

    textplacement = np.min(data_color)

    #plot
    ecdf_input.cdf.plot(ax, label='Satellite RGB', color='blue')
    ecdf_output.cdf.plot(ax, label = 'Mock Population', color='red')
    plt.xlabel("Color")
    plt.ylabel("CDF")
    plt.annotate(f'Distance = {D:.3f}', (textplacement -.2, 0.85), fontsize=12)
    plt.annotate(f'p-value = {p:.3e}', (textplacement -.2, 0.8), fontsize=12)
    plt.title(dwarf.name + " CDF for RGB vs. Drawn Population")
    plt.legend()
    plt.show()

def kde_2d2(data_apt_clean, kind='kde', cmap='Blues'):
    '''
    Used to show the 2D Kernel Density Estimator for the potential TRGB.
    This should line up relatively with the position of the satellite, and
    helps to identify the selected box to draw.
    '''
    x = data_apt_clean['ra']
    y = data_apt_clean['dec']

    # First plot
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=x, y=y, cmap=cmap, cbar=True)

    ax = plt.gca()
    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.tick_params(axis='x', rotation=45)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    plt.title('2D KDE Plot for Overdensities')
    plt.xlabel("RA")
    plt.ylabel("Dec")
    plt.show()

    # Second plot
    if kind == 'kde':
        g = sns.jointplot(x=x, y=y, kind=kind, fill=True, cmap=cmap)
    elif kind == 'hist':
        g = sns.jointplot(x=x, y=y, kind=kind, cmap=cmap)
    else:
        g = sns.jointplot(x=x, y=y, kind=kind)

    g.ax_joint.invert_xaxis()
    g.ax_joint.set_aspect('equal')
    g.ax_joint.tick_params(axis='x', rotation=45)
    g.ax_joint.xaxis.set_major_locator(ticker.MaxNLocator(5))
    g.ax_joint.yaxis.set_major_locator(ticker.MaxNLocator(5))
    g.ax_joint.set_xlabel("RA")
    g.ax_joint.set_ylabel("Dec")

    plt.show()














    