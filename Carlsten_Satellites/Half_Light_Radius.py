### This is a compilation of code block from Markito that are used to find the RGB ###
### stars that exist within x half-light radii from a source ####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy
import astropy.units as u
import scipy.interpolate
import ezpadova
from ezpadova import parsec
from astropy.table import Table
from ezpadova import QuickInterpolator
from matplotlib import patches
from matplotlib.path import Path
import h5py
from matplotlib.patches import Ellipse, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from photutils.aperture import (ApertureStats, aperture_photometry)
from photutils.aperture import (CircularAnnulus, CircularAperture, EllipticalAnnulus, EllipticalAperture)
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import emcee

#import from other file
from . import colors

#### common Variables #####

####### PARSEC Query Parameters #######
photsys         = "YBC_tab_mag_odfnew/tab_mag_wfc3_202101_wide.dat"
photsys_acs     = "YBC_tab_mag_odfnew/tab_mag_acs_wfc_202101.dat"
photsys_version = 'YBCnewVega'
parsec_model    = 'parsec_CAF09_v1.2S'
track_colibri   = 'parsec_CAF09_v1.2S_S_LMC_08_web'
imf_file        = "tab_imf/imf_kroupa_orig.dat"
eta_reimers     = "0.3"

parsec_params = dict(parsec_model=parsec_model, track_colibri=track_colibri, photsys_file=photsys_acs, 
                     imf_file=imf_file, eta_reimers=eta_reimers)

####### PARSEC ACS/WFC Column Names #######
col_names_iso = ['Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe', 'logg', 'label',
                 'McoreTP', 'C_O', 'period0', 'period1', 'period2', 'period3', 'period4', 'pmode', 'Mloss', 
                 'tau1m','X', 'Y', 'Xc', 'Xn', 'Xo', 'Cexcess', 'Z', 'mbolmag','F218Wmag', 'F225Wmag', 
                 'F275Wmag', 'F336Wmag', 'F390Wmag', 'F438Wmag', 'F475Wmag', 'F555Wmag', 'F606Wmag', 
                 'F625Wmag','F775Wmag', 'F814Wmag', 'F105Wmag', 'F110Wmag', 'F125Wmag','F140Wmag', 'F160Wmag']

### common Labels #####
common_labels_text = [r"\alpha \, [^{\circ}]", r"\delta \, [^{\circ}]", r"r_h \, [']",
                      r"\epsilon", r"\theta \, [^{\circ}] "]
common_labels      = [r"$\alpha \, [^{\circ}]$", r"$\delta \, [^{\circ}]$", r"$r_h \, ['] $",
                      r"$\epsilon$", r"$\theta \, [^{\circ}] $"]
label_extras = {'sersic'   : {'labels_text': [r"N_*", r'n'],
                             'labels':       [r"$N_*$", r"$n$"],
                             'profile_txt': 'Sérsic',
                            },
                'exponential': {'labels_text': [r"N_*"],
                               'labels': [r"$N_*$"],
                               'profile_txt': 'Exponential',
                            }
               }

profile_x = 'exponential'  ## 'sersic'  ##
ndim = {'sersic': 7, 'exponential': 6}.get(profile_x, None)
labels_text   = common_labels_text + label_extras[profile_x]['labels_text']
labels        = common_labels + label_extras[profile_x]['labels']

def find_angular_seperation():
    pass

def get_separation(zeropoint, coordinates, unit='arcmins'):
    
    '''
    Inputs: 
        zeropoint (astropy.SkyCoord): reference point from which you want to calc separation
        coordinates (astropy.SkyCoords): list of coordinates you want to find separations for
        in_kpc: if you want separations in kpc, make this True. Else, get result in degrees.
    Returns: Array of transformed dec, ra separations in degrees from zeropoint coordinate
    '''
    import astropy.units as u
    
    sep = zeropoint.separation(coordinates)
    ang = zeropoint.position_angle(coordinates)
    # if unit=='kpc':
    #     delta_dec = sep.degnp.cos(ang.radian)*dgal(np.pi/180) # in kpc currently
    #     delta_ra = sep.degnp.sin(1.0ang.radian)dgal(np.pi/180)
    if unit=='degrees':
        delta_dec = ((sep.deg * np.cos(ang.radian)) * u.degree).value # in degrees
        delta_ra = ((sep.deg * np.sin(1.0*ang.radian)) * u.degree).value
    if unit=='arcmins':
        delta_dec = (((sep.deg * np.cos(ang.radian)) * u.degree).to(u.arcmin)).value # in arcmin
        delta_ra = (((sep.deg * np.sin(1.0*ang.radian)) * u.degree).to(u.arcmin)).value
    if unit == 'arcsec':
        delta_dec = (((sep.deg * np.cos(ang.radian)) * u.degree).to(u.arcsec)).value # in arcsec
        delta_ra = (((sep.deg * np.sin(1.0*ang.radian)) * u.degree).to(u.arcsec)).value               
    return delta_dec, delta_ra

#select RGB stars
def cut_isochrones_path(mag_1_data, mag_2_data, mag_1_err, mag_2_err, isochrones, distance_modulus, 
                        mag1_cut=[21.0, 28.5], mag2_cut=[21.0, 26.5], radius=0.1, err_factor=1.0, 
                        max_stage=6, include_hb=True):
    """
    Cut to identify objects within an array of isochrone cookie-cutters.
    mag_1, mag_2, mag_1_err, mag_2_err, = F606W (data), F814W (data), F606W err (ast), F814W err (ast)
    radius = isochrone intrinsic err (default = 0.1, other common = 0.07, 0.12, 0.15)
    """
    # Initialize a final cut mask to combine all isochrone cuts
    final_cut = np.zeros(len(mag_1_data), dtype=bool)
    
    mag_1_data = np.asarray(mag_1_data)
    mag_2_data = np.asarray(mag_2_data)
    for isochrone in isochrones:
        # Select ONLY RGB and HB stages
        irgb, = np.where((isochrone['label']>=1) & (isochrone['label']<=3) )
        ihb,  = np.where((isochrone['label']>=4) & (isochrone['label']<=max_stage) )

        #### Mags of RGB & HB stages
        mag_1_rgb = isochrone['F606Wmag'][irgb] + distance_modulus
        mag_2_rgb = isochrone['F814Wmag'][irgb] + distance_modulus
        mag_1_hb  = isochrone['F606Wmag'][ihb]  + distance_modulus
        mag_2_hb  = isochrone['F814Wmag'][ihb]  + distance_modulus

        # Reverse the arrays for interpolation
        mag_1_rgb = mag_1_rgb[::-1]
        mag_2_rgb = mag_2_rgb[::-1]
        mag_1_hb  = mag_1_hb[::-1]
        mag_2_hb  = mag_2_hb[::-1]

        ### Cut based on the first filter (F814W)
        f_isochrone_1    = scipy.interpolate.interp1d(mag_2_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False,
                                                      fill_value='extrapolate')
        f_isochrone_1_hb = scipy.interpolate.interp1d(mag_2_hb, mag_1_hb - mag_2_hb, bounds_error=False,
                                                      fill_value='extrapolate')
        
        color_diff_1    = np.fabs((mag_1_data - mag_2_data) - f_isochrone_1(mag_2_data))
        color_diff_1_hb = np.fabs((mag_1_data - mag_2_data) - f_isochrone_1_hb(mag_2_data))
        
        treshold_2 = np.sqrt(radius**2 + (err_factor*mag_2_err)**2 + (err_factor*mag_1_err)**2)
        cut_2_rgb  = (color_diff_1 < treshold_2)
        cut_2_hb   = (color_diff_1_hb < treshold_2)
        
        if include_hb: 
            cut_2 = np.logical_or(cut_2_rgb, cut_2_hb)
        else:
            cut_2 = cut_2_rgb
        ###---###

        ### Cut based on the second filter (F606W)
        f_isochrone_2    = scipy.interpolate.interp1d(mag_1_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False,
                                                      fill_value='extrapolate')
        f_isochrone_2_hb = scipy.interpolate.interp1d(mag_1_hb, mag_1_hb - mag_2_hb, bounds_error=False,
                                                      fill_value='extrapolate')
        
        color_diff_2    = np.fabs((mag_1_data - mag_2_data) - f_isochrone_2(mag_1_data))
        color_diff_2_hb = np.fabs((mag_1_data - mag_2_data) - f_isochrone_2_hb(mag_1_data))
        
        treshold_1 = np.sqrt(radius**2 + (err_factor*mag_2_err)**2 + (err_factor*mag_1_err)**2)
        cut_1_rgb  = (color_diff_2 < treshold_1)
        cut_1_hb   = (color_diff_2_hb < treshold_1)
        
        if include_hb: 
            cut_1 = np.logical_or(cut_1_rgb, cut_1_hb)
        else:
            cut_1 = cut_1_rgb
        
        ### Combine the cuts for this isochrone
        cut = np.logical_or(cut_1, cut_2)

        ### Update the final cut by combining with previous isochrone cuts
        # final_cut = (np.logical_or(final_cut, cut) & (mag_2_data<= mag2_cut) & (mag_1_data<= mag1_cut))
        final_cut = ( np.logical_or(final_cut, cut) 
                     &(mag_2_data<= mag2_cut[1]) &(mag_2_data>= mag2_cut[0]) 
                     &(mag_1_data<= mag1_cut[1]) &(mag_1_data>= mag1_cut[0]) )
    ###---###
    ### Calculate magnitude bins and errors
    return final_cut

def load_reader(filename):
    return emcee.backends.HDFBackend(filename, read_only=True)

def load_reader_h5(filename):
    file = emcee.backends.HDFBackend(filename, read_only=True) 
    file = h5py.File(filename, 'r')
    return file

########################################################################### 
def rh_analytical_king(rc, rt):
    c_k = np.log10(rt/rc)
    r_e = (0.5439 + 0.1044 * c_k + 1.5618 * c_k**2 - 0.7559 * c_k**3 + 0.2572 * c_k **4 ) * rc
    return r_e

########################################################################### 
def get_mcmc_structural_params(profile_x, gal_name, output='bstval', rh_output=False, reader='mcmc'):
    
    # ndim_dict = {'sersic': 7, 'exponential': 6, 'plummer': 6, 'king': 7}
    
    profile_dict = {'sersic':      {'ndim': 7, 'rh_idx': 2, 'e_idx': 3, 'PA_idx': 4}, 
                    'exponential': {'ndim': 6, 'rh_idx': 2, 'e_idx': 3, 'PA_idx': 4}, 
                    'plummer':     {'ndim': 6, 'rh_idx': 2, 'e_idx': 3, 'PA_idx': 4}, 
                    'king':        {'ndim': 7, 'rh_idx': None, 'e_idx': 4, 'PA_idx': 5}, 
                    }
    profile_idxs = profile_dict.get(profile_x, None)
    
    ndim   = profile_idxs['ndim']
    rh_idx = profile_idxs['rh_idx']
    e_idx  = profile_idxs['e_idx']
    PA_idx = profile_idxs['PA_idx']
    
    if reader == 'mcmc':
        filename = "{}.h5".format(gal_name + "_" + profile_x + "_arcmin") 
        reader   = load_reader(filename)
    
        tau           = reader.get_autocorr_time(tol=ndim, quiet=True)
        discard, thin = int(20 * np.min(tau)), int(0.5 * np.min(tau))
        flat_samples  = reader.get_chain(discard=discard, thin=thin, flat=True)
    elif reader == 'h5':
        filename = f"sampler_run_{gal_name}.h5"
        file     = load_reader_h5(filename)
        flat_samples = file['chain'][()]
    ###----####
    
    wrapped_PA      = estimate_position_angle_samples(flat_samples[:, PA_idx] * 180./np.pi, method='wrap')
    peak_pa         = kde_peak(flat_samples[:, PA_idx] * 180./np.pi)
    peak_wrapped_PA = kde_peak(wrapped_PA)
    flat_samples[:, PA_idx] = wrapped_PA + (peak_pa - peak_wrapped_PA)
    
    if rh_output:
        if profile_x =='king':
            ah_king = rh_analytical_king(flat_samples[:, 2], flat_samples[:, 3]) 
            ah_king = ah_king * np.sqrt(1.0 - flat_samples[:, e_idx]) 
            flat_samples = np.insert(flat_samples, 3, ah_king, axis=1)
        else:
            flat_samples[:, rh_idx] = flat_samples[:, rh_idx] * np.sqrt(1.0 - flat_samples[:, e_idx]) 
    ###---###
    bstval = np.zeros(np.shape(flat_samples)[1])
    param_errs = []
    for i in range(np.shape(flat_samples)[1]):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        param_errs.append(q)
        bstval[i] = mcmc[1]
    ###---###
    if output == 'bstval':
        return bstval, param_errs
    elif output == 'flat_samples':
        return flat_samples
#######----#######
    
#####################################
def plot_density_ellipse(ax, profile_x, bstval, ellipse_kwargs=None, radii=[2, 3]):
    
    if ellipse_kwargs is None:
        ellipse_kwargs = dict(facecolor='none', linestyle="-", linewidth=0.5, alpha=0.99, 
                              edgecolor='black')
    # Unpack parameters
    if profile_x == 'sersic':
        x0, y0, rh, e, PA, _, _ = bstval
    elif profile_x in ['exponential', 'plummer']:
        x0, y0, rh, e, PA, _ = bstval
    elif profile_x == 'king':
        x0, y0, rc, rt, e, PA, _ = bstval
        rh = rh_analytical_king(rc, rt)
    
    angle = 90.0 - PA
    if profile_x == 'king':
        # Add rc and rt ellipses in purple
        a_c, a_t = 2 * rc, 2 * rt
        b_c, b_t = a_c * (1 - e), a_t * (1 - e)
        for a_, b_ in zip([a_c, a_t], [b_c, b_t]):
            ax.add_artist(Ellipse([x0, y0], a_, b_, angle=angle, **ellipse_kwargs))
    else:
        for i in radii:
            a = i * rh
            b = a * (1 - e)

            ellipse = Ellipse([x0, y0], a * 2, b * 2, angle=angle, **ellipse_kwargs)
            ax.add_artist(ellipse)
#####################################

def rho_dwarf(theta, x, y, profile='sersic'):
    if profile == 'sersic':
        x0, y0, rh, e, PA, N_star, n = theta

        x_off = (x - x0)
        y_off = (y - y0)
        
        x_r = (x_off * np.cos(PA) - y_off * np.sin(PA) ) / (1.0 - e)
        y_r = (x_off * np.sin(PA) + y_off * np.cos(PA) )
        
        r = np.sqrt(x_r**2 + y_r**2)

        bn = 1.9992 * n - 0.3271
        #rho_0 = bn**(2*n) /(2 * np.pi * rh**2 * (1-e) * n * scipy.special.gamma(2*n) ) 

        log_rho_0 = ( 2.0 * n * np.log(bn) - np.log(2.0 * np.pi * rh**2 * (1.0 - e) * n)
                     - scipy.special.gammaln(2.0 * n) ) 
        
        rho_0 = np.exp(log_rho_0)
        rho_gal = rho_0 * N_star * np.exp(-bn * (r / rh)**(1.0 / n) )
        return rho_gal
    elif profile == 'exponential':
        x0, y0, rh, e, PA, N_star = theta
        
        x_off = (x - x0)
        y_off = (y - y0)
        
        x_r = (x_off * np.cos(PA) - y_off * np.sin(PA) ) / (1.0 - e)
        y_r = (x_off * np.sin(PA) + y_off * np.cos(PA) )
        
        r = np.sqrt(x_r**2 + y_r**2)
        
        rho_0 = 1.68**2 * N_star / ( 2.0 * np.pi * rh**2 * (1.0 - e) ) 
        rho_gal = rho_0 * np.exp(-1.68 * r / rh) 
        return rho_gal
    elif profile == 'plummer':
        x0, y0, rp, e, PA, N_star = theta
        
        x_off = (x - x0)
        y_off = (y - y0)
        
        x_r = (x_off * np.cos(PA) - y_off * np.sin(PA) ) / (1.0 - e)
        y_r = (x_off * np.sin(PA) + y_off * np.cos(PA) )
        
        r = np.sqrt(x_r**2 + y_r**2)
        
        rho_0   = N_star / ( np.pi * rp**2 *  (1.0 - e) )
        rho_gal = rho_0 * 1.0 / ( 1.0 + r**2 / rp**2 )**2
        return rho_gal
    elif profile == 'king':
        x0, y0, rc, rt, e, PA, N_star = theta
        
        x_off = (x - x0)
        y_off = (y - y0)
        
        x_r = (x_off * np.cos(PA) - y_off * np.sin(PA) ) / (1.0 - e)
        y_r = (x_off * np.sin(PA) + y_off * np.cos(PA) )
        
        r = np.sqrt(x_r**2 + y_r**2)
        
        c = rt/rc
        rho_0  =  N_star/( np.pi * rc**2 *(1.0 - e) ) * (1.0/( np.log(1.0 + c**2) 
                  - ( ( (3.0 * (1.0 + c**2)**(1/2) - 1.0) * ((1.0 + c**2)**(1/2) - 1.0) )/(1.0 + c**2) ) ) )
        
        rho_gal = rho_0 *((1.0 + (r/rc)**2)**(-1/2) - (1.0 + (c)**2)**(-1/2))**2
        return rho_gal
    else:
        raise ValueError(f"Unknown method '{method}' for Dwarf model.")
#######----#######

def integ(theta, xgrid, ygrid, px_per_bin, area_per_px, profile='sersic'):
    ### `px_per_bin` is number of pixels in bin (in arcmin^2)
    dA = (px_per_bin * area_per_px).T 
    Integral = np.sum(dA * rho_dwarf(theta, ygrid, xgrid, profile=profile))
    return Integral

########################|---------- Position Angle Functions ----------|########################
### NOTE: - Adapted from ugali package, but modified to handle the periodicity of angles and to 
###         be more robust for cases with two peaks at ~0 and ~180.
###       - THESE FUNCTIONS BELOW ARE MORE OR LESS THE SAME AS USING Astropy's wrat_at angle 
###         method; but realized Astropy could do this later on, so I stick to this for now. 
#######--#######--#######

_alpha   = 0.32
_nbins   = 300
_npoints = 1000

def interval(best,lo=np.nan,hi=np.nan):
    """
    Pythonized interval for easy output to yaml

    Parameters
    ----------
    best : best-fit estimate of the parameter
    lo   : lower value
    hi   : higher value

    Returns
    -------
    [best, [lo, hi]] : list of values (Conidence interval)
    """
    return [float(best),[float(lo),float(hi)]]

def peak(data, bins=_nbins):
    """
    Bin the distribution and find the mode

    Parameters:
    -----------
    data  : The 1d data sample
    bins  : Number of bins

    Returns
    -------
    peak : peak of the kde
    """
    num,edges = np.histogram(data,bins=bins)
    centers = (edges[1:]+edges[:-1])/2.
    return centers[np.argmax(num)]

def gauss_kde(data, npoints=_npoints, clip=5.0):
    """
    Identify peak using Gaussian kernel density estimator.
    
    Parameters:
    -----------
    data    : The 1d data sample
    npoints : The number of kde points to evaluate
    clip    : The Normalized Median Absolute Deviation (NMAD) to clip

    Returns
    -------
    peak : peak of the kde
    """

    # Clipping of severe outliers to concentrate more KDE samples
    # in the parameter range of interest
    mad = np.median(np.fabs(np.median(data) - data))
    if clip > 0:
        cut  = (data > np.median(data) - clip * mad)
        cut &= (data < np.median(data) + clip * mad)
        x = data[cut]
    else:
        x = data
    kde = scipy.stats.gaussian_kde(x)
    # No penalty for using a finer sampling for KDE evaluation
    # except computation time
    values = np.linspace(np.min(x), np.max(x), npoints)
    kde_values = kde.evaluate(values)
    peak = values[np.argmax(kde_values)]
    return peak, kde.evaluate(peak)

def kde_peak(data, npoints=_npoints, clip=5.0):
    """
    Identify peak using Gaussian kernel density estimator.

    Parameters:
    -----------
    data    : The 1d data sample
    npoints : The number of kde points to evaluate
    clip    : The Normalized Median Absolute Deviation (NMAD) to clip

    Returns
    -------
    peak : peak of the kde
    """
    return gauss_kde(data,npoints,clip)[0]

def peak_interval(data, alpha=_alpha, npoints=_npoints, clip =5.0 ):
    """Identify minimum interval containing the peak of the posterior as
    determined by a Gaussian kernel density estimator.

    Parameters
    ----------
    data   : the 1d data sample
    alpha  : the confidence interval
    npoints: number of kde points to evaluate

    Returns
    -------
    interval : the minimum interval containing the peak
    """
    peak = kde_peak(data, npoints, clip)
    x = np.sort(data.flat); n = len(x) ## x: same len as flat samples
    # The number of entries in the interval
    window = int(np.rint((1.0-alpha)*n))
    # The start, stop, and width of all possible intervals
    starts = x[:n-window]; ends = x[window:]
    widths = ends - starts
    # Just the intervals containing the peak
    select = (peak >= starts) & (peak <= ends)
    widths = widths[select]
    len(widths)
    if len(widths) == 0:
        raise ValueError('Too few elements for interval calculation')
    min_idx = np.argmin(widths)
    lo = starts[select][min_idx]
    hi = ends[select][min_idx]
    return interval(peak,lo,hi)

def estimate_position_angle(pa, npoints=_npoints, clip=5.0, alpha=0.32):
    """ Estimate the position angle from the posterior dealing
    with periodicity.
    pa = flat samples after burn in (must be in degrees)
    """
    # Transform so peak in the middle of the distribution
    peak = kde_peak(pa, npoints, clip)
    shift = 180.*((pa+90-peak)>180)
    pa -= shift
    # Get the kde interval
    ret = peak_interval(pa, alpha, npoints)
    if ret[0] < 0: 
        ret[0] += 180.; ret[1][0] += 180.; ret[1][1] += 180.;
    return ret

def estimate_position_angle_samples(pa, npoints=_npoints, clip=5.0, method='KDE'):
    """ 
    Estimate the position angle from the posterior dealing with periodicity.
    
    Parameters:
    -----------
    pa      : ndarray
              Flat samples after burn-in (must be in degrees).
    npoints : int, optional
              Number of points for KDE, by default _npoints.
    clip    : float, optional
              Clipping value for KDE, by default 10.0.
    method  : str, optional
              Method to use for estimating position angle ('KDE', 'wrap', or 'fold').
    
    Returns:
    --------
    pa_corrected : ndarray
                   Corrected position angles based on the selected method.
    """

    ### Method 1: KDE-based peak shift (works when there are not two peaks)
    # Determine the KDE peak considering angle periodicity
    peak = kde_peak(pa, npoints, clip)
    # Transform so peak is in the middle of the distribution
    shift = 180. * ((pa + 90 - peak) > 180)
    pa_shifted = pa - shift
    
    ### Method 2: Wrapping angles to handle periodicity
    counts, bin_edges = np.histogram(pa, bins=25, density=True)
    hist_peak = 0.5 * (bin_edges[np.argmax(counts)] + bin_edges[np.argmax(counts) + 1])
    
    if hist_peak > 90:  # If the peak is closer to 180, shift to [90, 270]
        ### tested it! works amazing for when highest peak is very close to 180.
        wrapped_shifted_pa = pa - hist_peak + 90.0
        wrapped_pa = np.mod(wrapped_shifted_pa, 180.0) + 90.0 - 2*np.abs(180.0-hist_peak)
        
    else:  # If the peak is closer to 0, shift to [-90, 90]
        wrapped_shifted_pa = pa - hist_peak +90.0
        wrapped_pa = np.mod(wrapped_shifted_pa, 180.0) - 90.0 + 1*np.abs(0.0-hist_peak)

    ### Method 3: Folding angles into [-90, 90] directly (handle two peaks at ~0 and ~180)
    folded_angle = np.mod(pa, 180)
    folded_angle = np.where(folded_angle > 90, folded_angle - 180, folded_angle)

    if method == 'KDE':
        return pa_shifted
    elif method == 'wrap':
        print('highest hist peak:', hist_peak)
        return wrapped_pa
    elif method == 'fold':
        return folded_angle
    else:
        raise ValueError("Invalid method. Choose from 'KDE', 'wrap', or 'fold'.")

##### Probability Density Functions #####

def rho_dwarf_kernel_pdf(r, theta, profile='sersic'):
    if profile == 'sersic':
        x0, y0, rh, e, PA, N_star, n = theta

        bn = 1.9992 * n - 0.3271
        #rho_0 = bn**(2*n) /(2 * np.pi * rh**2 * (1-e) * n * scipy.special.gamma(2*n) ) 
        log_rho_0 = ( 2.0 * n * np.log(bn) - np.log(2.0 * np.pi * rh**2 * (1.0 - e) * n)
                     - scipy.special.gammaln(2.0 * n) ) 
        
        rho_0 = np.exp(log_rho_0)
        rho_gal = rho_0 * N_star * np.exp(-bn * (r / rh)**(1.0 / n) )
        return rho_gal
    elif profile == 'exponential':
        x0, y0, rh, e, PA, N_star = theta
        
        rho_0 = 1.68**2 * N_star / ( 2.0 * np.pi * rh**2 * (1.0 - e) ) 
        rho_gal = rho_0 * np.exp(-1.68 * r / rh) 
        return rho_gal
    elif profile == 'plummer':
        x0, y0, rp, e, PA, N_star = theta
        
        rho_0   = N_star / ( np.pi * rp**2 *  (1.0 - e) )
        rho_gal = rho_0 * 1.0 / ( 1.0 + r**2 / rp**2 )**2
        return rho_gal
    elif profile == 'king':
        x0, y0, rc, rt, e, PA, N_star = theta
        
        c = rt/rc
        rho_0  =  N_star/( np.pi * rc**2 *(1.0 - e) ) * (1.0/( np.log(1.0 + c**2) 
                  - ( ( (3.0 * (1.0 + c**2)**(1/2) - 1.0) * ((1.0 + c**2)**(1/2) - 1.0) )/(1.0 + c**2) ) ) )
        
        rho_gal = rho_0 *((1.0 + (r/rc)**2)**(-1/2) - (1.0 + (c)**2)**(-1/2))**2
        return rho_gal
    else:
        raise ValueError(f"Unknown method '{method}' for Dwarf model.")
#######----#######

def kernel_integrate(bstval, profile='exponential', r_range=[0, np.inf]):
    """
    Calculate the 2D integral of the 1D surface brightness profile 
    (i.e, the flux) between rmin and rmax (elliptical radii). 

    Parameters:
    ----------
    rmin : minimum integration radius (deg)
    rmax : maximum integration radius (deg)

    Returns:
    -------
    integral : Solid angle integral (deg^2)
    """
    rmin, rmax = r_range
    if rmin < 0: raise Exception('rmin must be >= 0')
    integrand = lambda r: rho_dwarf_kernel_pdf(r, bstval, profile=profile) * 2*np.pi * r
    return scipy.integrate.quad(integrand, rmin, rmax, full_output=True, epsabs=0)[0]
#######----#######

def density_profile_1D(bstval, profile='sersic'):
    if profile =='sersic':
        x0, y0, rh, e, PA, N_star, n = bstval
        best_vals = np.array([x0, y0, rh, e, np.radians(PA), N_star, n])
    elif profile =='exponential':
        x0, y0, rh, e, PA, N_star = bstval
        best_vals = np.array([x0, y0, rh, e, np.radians(PA), N_star])
    elif profile =='plummer':
        x0, y0, rp, e, PA, N_star = bstval
        best_vals = np.array([x0, y0, rp, e, np.radians(PA), N_star])
    elif profile == 'king':
        x0, y0, rc, rt, e, PA, N_star = bstval
        best_vals = np.array([x0, y0, rc, rt, e, np.radians(PA), N_star])
    
    rx = np.linspace(x0, 6, 1000)
    ry = np.linspace(y0, 6, 1000)

    N_tot = len(deltara)
    area = (area_per_px * num_px)
    sigma_bk = (N_tot - integ(best_vals, profile=profile)) / area
    
    r_elliptical = elliptical_radius(rx, ry, bstval, profile=profile)
    contplot = rho_dwarf(best_vals, rx, ry, profile=profile)
    
    if profile == 'king':
        contplot = np.where(r_elliptical <= rt, contplot, 0.0 )  
        
    return r_elliptical, contplot, sigma_bk
#######----#######

def elliptical_radius(x, y, bstval, profile='sersic'):
    '''
    Takes in deltara, deltadec coords
    Returns elliptical radius (major axis)
    '''
    if profile == 'sersic':
        x0, y0, rh, e, PA, N_star, n = bstval 
    elif profile == 'exponential':
        x0, y0, rh, e, PA, N_star = bstval 
    elif profile == 'plummer':
        x0, y0, rp, e, PA, N_star = bstval
    elif profile == 'king':
        x0, y0, rc, rt, e, PA, N_star = bstval       

    PA_rad = np.radians(PA) ## input PA must be in cartesian (as given by mcmc posterior)
    
    #center the coordinates on the candidate's coordinates
    x_off = (x - x0)
    y_off = (y - y0)

    #use the rotation matrix to transform from non-rotated coords into coords that align with the satellite
    x_r = (x_off * np.cos(PA_rad) - y_off * np.sin(PA_rad) ) / (1.0 - e)
    y_r = (x_off * np.sin(PA_rad) + y_off * np.cos(PA_rad) )
    
    # # elliptical radius, with q = 1 - e
    # q = 1.0 - e
    # r_ell = np.sqrt(x_r**2 + (y_r / q)**2)

    # return r_ell
        
    r = np.sqrt(x_r**2 + y_r**2)
    
    return r
    
#######----#######

def Select_stars_inside_ellipse(deltara_stars, deltadec_stars, array, bstval, ufd_coords, factor=2.0, 
                                method='elliptical radius', profile='sersic', coords_cols=['RA', 'Dec']):
    """
    Select sources inside ellipse of size: `factor` x a_h (elliptical half-light radius)

    Parameters
    ----------
    array (astropy table-like array): array of point sources
    deltara_stars, deltadec_stars: relative positions of stars in `array` w.r.t. centroid of system
    bstval (1D numpy array): structural parameters of system
    ufd_coords: astropy coordinates of system
    factor: factor * r_h; size of ellipse where inside points would be selected (e.g., 2rh)
    method: `ellptical radius` or `ellipse`; method to select inside sources

    Returns
    -------
    deltara_irh, deltadec_irh: relative positions of stars inside system
    ufd_array: sub-array from `array` of stars inside ellipse size:
              `factor` x a_h (elliptical half-light radius)
    """
    import astropy.units as u
    
    if profile == 'sersic':
        x0, y0, rh, e, PA, f_bk, n = bstval #NOTE: PA here is in degrees shifted as 90 - KDE PA_algorith
    elif profile == 'exponential':
        x0, y0, rh, e, PA, f_bk = bstval #NOTE: PA here is in degrees shifted as 90 - KDE PA_algorith
    elif profile == 'plummer':
        x0, y0, rp, e, PA, N_star = bstval
        rh = rp
    elif profile == 'king':
        x0, y0, rc, rt, e, PA, N_star = bstval 
        rh = rh_analytical_king(rc, rt)
    
    elliptical_rad_pts = elliptical_radius(deltara_stars, deltadec_stars, bstval, profile=profile)
    
    a = factor * rh           #Semi-major axis
    b = factor * rh * (1 - e) #Semi-minor axis
    x_center, y_center = x0, y0
    PA_rad = np.radians(PA)
    
    # Rotate the coordinate system by the position angle
    cos_angle = np.cos(PA_rad)
    sin_angle = np.sin(PA_rad)

    # Center the points relative to the ellipse's center
    x_rel = deltara_stars - x_center
    y_rel = deltadec_stars - y_center
    
    # Rotate points to align with the ellipse's axes
    x_rot = cos_angle * x_rel + sin_angle * y_rel
    y_rot = -sin_angle * x_rel + cos_angle * y_rel

    # if method=='ellipse':
    #     # Check if the points are within the ellipse
    #     inside = ((x_rot**2 / a**2) + (y_rot**2 / b**2)) <= 1.0
        
    # elif method=='elliptical radius': 
    #     inside = np.where((elliptical_rad_pts<=(factor*rh)))

    # # # Select points inside the ellipse
    # # ufd_array = array[inside]
    
    # if isinstance(array, pd.DataFrame):
    #     ufd_array = array.loc[inside].copy()
    # else:
    #     ufd_array = array[inside]

    if method == 'ellipse':
        inside = ((x_rot**2 / a**2) + (y_rot**2 / b**2)) <= 1.0

    elif method == 'elliptical radius': 
        inside = elliptical_rad_pts <= (factor * rh)

    else:
        raise ValueError("method must be 'ellipse' or 'elliptical radius'")

    if isinstance(array, pd.DataFrame):
        ufd_array = array.loc[inside].copy()
    else:
        ufd_array = array[inside]

    ### Candidate Coordinates 
    coords_rh_pts = astropy.coordinates.SkyCoord(ufd_array[coords_cols[0]], ufd_array[coords_cols[1]],
                                                 unit=(u.deg, u.deg), frame='icrs')
    deltadec, deltara = get_separation(ufd_coords, coords_rh_pts)
    return deltadec, deltara, ufd_array

#######----#######


### sCrap code from em reworking stuff ###

#     ###From markito on making the delta ra, delrta dec image of the satellite
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

# cands_coords = {'DW0846P3300': (131.559, 32.997), #from the spreadsheet, --> Carlsten and or the proposal
#                 'DW0846P3348': (131.63, 33.811),
#                 'DW0852P3249': (133.199, 32.829),
#                 'DW0854P3314': (133.585, 33.249)}

# # Cands_coords = [(131.559, 32.997), (131.63, 33.811), (133.199, 32.829), (133.585, 33.249)]
# gal_name = 'DW0854P3314'
# ra_x0, dec_y0 = cands_coords[gal_name]

# ### Candidate Coordinates 
# Cand_coords = SkyCoord(ra_x0, dec_y0,  unit=['deg', 'deg'], frame='icrs')
# #### All Point sources Coordinates 
# #original
# # coords = SkyCoord(pts_sources_rgb['ra'], pts_sources_rgb['dec'], unit=(u.deg, u.deg), 
# #                   frame='icrs')
# # Cand_stars = SkyCoord(pts_sources['ra'], pts_sources['dec'], unit=(u.deg, u.deg), frame='icrs')

# #trying to mold his to mine
# coords = SkyCoord(ra_rgb, dec_rgb, unit=(u.deg, u.deg), 
#                   frame='icrs') #my orange points

# Cand_stars = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs') #my red point


# #kept his naming convention going forward
# deltadec, deltara = get_separation(Cand_coords, coords)
# deltadec_stars, deltara_stars = get_separation(Cand_coords, Cand_stars)

# mrk_s, mrk_lw, mrk_alpha = 7, 0.2, 0.6
# mrk_fc, mrk_ec = 'silver', 'black' 

# ax.scatter(deltara_stars , deltadec_stars, s=0.8*mrk_s, alpha=0.9*mrk_alpha, fc=mrk_fc, ec=mrk_ec, lw=mrk_lw)
# ax.scatter(deltara, deltadec,  s=mrk_s, alpha=mrk_alpha, fc='pink', ec='red', lw=mrk_lw)
# ax.text(0.1, 0.95, gal_name, transform=ax.transAxes, fontsize=20, 
#         bbox =dict(facecolor='white', edgecolor= 'red', #xkcd_fireengine_red, 
#                    boxstyle='round,pad=0.5'), 
#         verticalalignment='top', fontfamily ='serif', zorder = 11)

# ax.set_xlim(-1.5, 3.0)
# ax.set_ylim(-2.25, 2.25)
# ax.invert_xaxis()
# ax.set_ylabel(r"$\Delta \delta$ (arcmin)", fontsize=12)
# ax.set_xlabel(r"$\Delta \alpha$ (arcmin) ", fontsize=12)
# ax.set_aspect('equal')
# ax.tick_params(labelsize=8)

# plt.show()