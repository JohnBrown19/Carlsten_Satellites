#Imports and data Loading
from astropy.io import ascii
import numpy as np
import pandas as pd
import ezpadova #isochrones
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, poisson
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
from astropy.coordinates import SkyCoord
from pathlib import Path
from matplotlib.path import Path as MplPath

#import from other file
from .Analysis import cull_data, extinction_correction, clean_asts, pc_to_arcmin, Poisson_stats
from .Half_Light_Radius import get_separation, elliptical_radius, get_mcmc_structural_params, plot_density_ellipse, cut_isochrones_path

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
    def __init__(self, name, coords, Host, distance, dmod, logmass, e_logmass, effective_radius, profile, base_path=None, results_path=None):
        ## Data Attributes
        #self.name = str.upper(name)
        self.name = str(name).strip().upper()
        self.coords = np.array(coords)
        self.Host = str(Host)
        self.distance = float(distance)
        self.dmod = float(dmod)
        self.logmass = float(logmass)
        self.e_logmass =float(e_logmass)
        self.eff_rad_pc = float(effective_radius)
        self.half_light_radius = None
        #self.profile = str(profile).lower()
        self.profile = self.clean_profile(profile)
        self.asts = None
        self.data = None
        self.data_masked = None
        self.pts_sources_rgb = None
        self.pixel_mask=None
        self.wcs_mask = None
        self.header=None
        self.SkyCoord = None
        self.cand_center = None
        self.deltadec_masked = None
        self.deltara_masked = None
        self.eff_rad_arcmin = None
        self.inside_radius = None
        self.pixs_in_rad = None
        self.good_pixels = None
        self.target_px = None
        self.bg_px = None
        self.N_target_px = None
        self.N_bg_px = None
        self.target_area = None
        self.bk_area = None
        self.total_area = None
        self.deltadec_stars = None
        self.deltara_stars = None
        self.p_val = None
        self.deltadec_rgb = None
        self.deltara_rgb = None
        self.deltadec_all = None
        self.deltara_all = None
        self.bstval = None
        self.param_errs = None
        self.obs_below_trgb = None
        
        # ##Path attributes
        # ##If base_path is None, use the current working directory
        # # self.base_path = Path.cwd() if base_path is None else Path(base_path)
        # # work_dir = '/Users/jonatbro/Documents/Research-backup/GHOSTS-data/17797'
        # # self.base_path = Path(work_dir) if base_path is None else Path(base_path)

        # # If results_path is None, use the base_path to build / use the results_path
        # if results_path is None:
        #     self.results_path = (
        #         self.base_path
        #         #/ "17797"
        #         / "results"
        #         / self.name
        #     )
        # else:
        #     self.results_path = Path(results_path) / self.name

        # self.results_path.mkdir(parents=True, exist_ok=True)

        # self.photometry_path = (
        #     self.base_path 
        #     #/ "17797" 
        #     / "fake-results2" 
        #     / f"17797_{self.name}" 
        #     / "proc_default_deepCR"
        # )

        # self.ast_path = (
        #     self.base_path 
        #     #/ "17797" 
        #     / "merged-results"
        # )
        
        # /Users/jonatbro/Documents/Research-backup/GHOSTS-data/17797
        self.base_path = Path.cwd() if base_path is None else Path(base_path)
        #default_base_path = Path("/Users/jonatbro/Documents/Research-backup/GHOSTS-data/17797")
        #self.base_path = default_base_path if base_path is None else Path(base_path)
        
        if results_path is None:
            self.results_path = (
                self.base_path
                / "results"
                / self.name
            )
        else:
            self.results_path = Path(results_path) / self.name
        
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        self.photometry_path = (
            self.base_path
            / "fake-results2"
            / f"17797_{self.name}"
            / "proc_default_deepCR"
        )
        
        self.photometry_file = (
            self.photometry_path
            / f"17797_{self.name}.phot_full.hdf5"
        )
        
        self.ast_path = (
            self.base_path
            / "merged-results"
        )


        
        #base_path="/Users/jonatbro/Documents/Research-backup/GHOSTS-data",
        #results_path="/Users/jonatbro/Documents/Research-backup/results"
        
    #below here are methods
    #you access attributes using object.attribute()
    # def load_asts(self):
    #     try: 
    #         self.asts = ascii.read("./17797/merged-results/17797_" + self.name 
    #                             + "_fakestars.dat", names=self.ast_cols)
    #         self.sats_type = type(self.asts)
    #         print(f"AST data for {self.name} loaded sucessfully")
    #     except Exception as e:
    #         print(f"could not load usable ASTs for {self.name}. Using ASTS from DW0846P3300 as a stand-in")
    #         print(f"Reason: {e}")
    #         self.asts = ascii.read("./17797/merged-results/17797_" + 'DW0846P3300' 
    #                             + "_fakestars.dat", names=self.ast_cols)
    #         print(f"AST data for {DW0846P3300} loaded sucessfully")

    def clean_profile(self, profile):
        """
        Normalize profile labels from external tables.
        """

        if pd.isna(profile):
            return None

        profile = str(profile).strip().lower()

        profile_map = {
            "exponential": "exponential",
            "expoential": "exponential",   # typo correction
            "exp": "exponential",
            "sersic": "sersic",
            "sérsic": "sersic",
            "plummer": "plummer",
            "king": "king",
            "": None,
            "none": None,
            "nan": None,
            "--": None}

        return profile_map.get(profile, None)

    # def load_asts(self): ##old tried and true
    #     ast_file = self.ast_path / f"17797_{self.name}_fakestars.dat"

    #     try:
    #         self.asts = ascii.read(ast_file, names=self.ast_cols)
    #         self.sats_type = type(self.asts)
    #         print(f"AST data for {self.name} loaded successfully from:")
    #         print(ast_file)

    #     except Exception as e:
    #         print(f"Could not load usable ASTs for {self.name}.")
    #         print(f"Reason: {e}")

    #         fallback_name = "dw0846p3300"
    #         fallback_file = self.ast_path / f"17797_{fallback_name}_fakestars.dat"

    #         self.asts = ascii.read(fallback_file, names=self.ast_cols)
    #         print(f"Using AST data from {fallback_name}:")
    #         print(fallback_file)
    # def load_asts(self):
    #     """
    #     Load raw ASTs only.
    
    #     Culling happens later in clean_asts().
    #     """
    
    #     standard_exposure_cols = [
    #         "m606_exp1_chip1",
    #         "m606_exp1_chip2",
    #         "m814_exp1_chip1",
    #         "m814_exp1_chip2",
    #         "m814_exp2_chip1",
    #         "m814_exp2_chip2",
    #         "m606_exp2_chip1",
    #         "m606_exp2_chip2",
    #     ]
    
    #     short_snap_exposure_cols = [
    #         "m814_exp1_chip1",
    #         "m814_exp1_chip2",
    #         "m606_exp1_chip1",
    #         "m606_exp1_chip2",
    #         "m814_exp2_chip1",
    #         "m814_exp2_chip2",
    #     ]
    
    #     short_snap_names = {
    #         "DW1234P3952",
    #     }
    
    #     def get_column_names(dwarf_name):
    #         dwarf_name_clean = str(dwarf_name).strip().upper()
    
    #         if dwarf_name_clean in short_snap_names:
    #             ast_column_names = self.ast_cols + short_snap_exposure_cols
    #             layout_name = "short-snap"
    #         else:
    #             ast_column_names = self.ast_cols + standard_exposure_cols
    #             layout_name = "standard"
    
    #         return ast_column_names, layout_name
    
    #     def read_ast_file(ast_file, dwarf_name):
    #         ast_column_names, layout_name = get_column_names(dwarf_name)
    
    #         ast_table = ascii.read(
    #             str(ast_file),
    #             names=ast_column_names,
    #             # format="no_header",
    #             # delimiter="\t",
    #             # guess=False,
    #             # fast_reader=False,
    #         )
    
    #         return ast_table, layout_name
    
    #     ast_file = self.ast_path / f"17797_{self.name}_fakestars.dat"
    
    #     try:
    #         self.asts, ast_layout = read_ast_file(ast_file, self.name)
    #         self.sats_type = type(self.asts)
    
    #         print(f"Raw AST data for {self.name} loaded successfully.")
    #         print(f"Layout: {ast_layout}")
    #         print(f"File: {ast_file}")
    #         print(f"Raw AST rows: {len(self.asts)}")
    #         print(f"Raw AST columns: {len(self.asts.colnames)}")
    
    #     except Exception as error:
    #         print(f"Could not load raw ASTs for {self.name}.")
    #         print(f"Reason: {type(error).__name__}: {error}")
    
    #         fallback_name = "DW0846P3300"
    #         fallback_file = (
    #             self.ast_path
    #             / f"17797_{fallback_name}_fakestars.dat"
    #         )
    
    #         self.asts, fallback_layout = read_ast_file(
    #             fallback_file,
    #             fallback_name,
    #         )
    
    #         self.sats_type = type(self.asts)
    
    #         print(f"Using fallback AST data from {fallback_name}.")
    #         print(f"Layout: {fallback_layout}")
    #         print(f"File: {fallback_file}")
    #         print(f"Raw fallback AST rows: {len(self.asts)}")
    #         print(f"Raw fallback AST columns: {len(self.asts.colnames)}")
    def load_asts(self):
        """
        Load raw ASTs only.
    
        Culling happens later in clean_asts().
        """
    
        standard_exposure_cols = [
            "m606_exp1_chip1",
            "m606_exp1_chip2",
            "m814_exp1_chip1",
            "m814_exp1_chip2",
            "m814_exp2_chip1",
            "m814_exp2_chip2",
            "m606_exp2_chip1",
            "m606_exp2_chip2",
        ]
    
        short_snap_exposure_cols = [
            "m814_exp1_chip1",
            "m814_exp1_chip2",
            "m606_exp1_chip1",
            "m606_exp1_chip2",
            "m814_exp2_chip1",
            "m814_exp2_chip2",
        ]
    
        short_snap_names = {
            "DW1234P3952",
        }
    
        def get_column_names(dwarf_name):
            dwarf_name_clean = str(dwarf_name).strip().upper()
    
            if dwarf_name_clean in short_snap_names:
                ast_column_names = self.ast_cols + short_snap_exposure_cols
                layout_name = "short-snap"
            else:
                ast_column_names = self.ast_cols + standard_exposure_cols
                layout_name = "standard"
    
            return ast_column_names, layout_name
    
        def read_ast_file(ast_file, dwarf_name):
            ast_column_names, layout_name = get_column_names(dwarf_name)
    
            ast_table = ascii.read(
                str(ast_file),
                names=ast_column_names,
                # format="no_header",
                # delimiter="\t",
                # guess=False,
                # fast_reader=False,
            )
    
            return ast_table, layout_name

        # --------------------------------------------------
        # Candidate-specific AST substitutions
        # --------------------------------------------------
        ast_source_overrides = {
            "DW1121P1411": "DW0241P3829",
        }

        requested_name = str(self.name).strip().upper()

        ast_source_name = ast_source_overrides.get(
            requested_name,
            requested_name,
        )
    
        ast_file = self.ast_path / f"17797_{ast_source_name}_fakestars.dat"
    
        try:
            self.asts, ast_layout = read_ast_file(
                ast_file,
                ast_source_name,
            )
            self.sats_type = type(self.asts)
    
            print(f"Raw AST data for {self.name} loaded successfully.")
            if ast_source_name != requested_name:
                print(
                    f"Using substitute ASTs from {ast_source_name} "
                    f"for {requested_name}."
                )
            print(f"Layout: {ast_layout}")
            print(f"File: {ast_file}")
            print(f"Raw AST rows: {len(self.asts)}")
            print(f"Raw AST columns: {len(self.asts.colnames)}")
    
        except Exception as error:
            print(f"Could not load raw ASTs for {self.name}.")
            print(f"Reason: {type(error).__name__}: {error}")
    
            fallback_name = "DW0846P3300"
            fallback_file = (
                self.ast_path
                / f"17797_{fallback_name}_fakestars.dat"
            )
    
            self.asts, fallback_layout = read_ast_file(
                fallback_file,
                fallback_name,
            )
    
            self.sats_type = type(self.asts)
    
            print(f"Using fallback AST data from {fallback_name}.")
            print(f"Layout: {fallback_layout}")
            print(f"File: {fallback_file}")
            print(f"Raw fallback AST rows: {len(self.asts)}")
            print(f"Raw fallback AST columns: {len(self.asts.colnames)}")
            
    def load_data(self):
        # self.data = pd.read_hdf("./17797/fake-results2/17797_" + self.name 
        #                         + "/proc_default_deepCR/17797_" + self.name 
        #                         + ".phot_full.hdf5", 
        #                         key='data')
        file_path = self.photometry_path / f"17797_{self.name}.phot_full.hdf5"
        self.data = pd.read_hdf(file_path, key="data")
        print(f"Data for {self.name} loaded sucessfully")

    # def load_data_masked(self): #defunct function? 
    #     self.data_masked = pd.read_hdf("./17797/fake-results2/17797_" + self.name 
    #                             + "/proc_default_deepCR/17797_" + self.name 
    #                             + ".phot_full_masked.hdf5", 
    #                             key='data')
    #     print(f"Masked data for {self.name} loaded sucessfully")

    def load_mask(self): #defunct function? 
        #mask_path = "./17797/fake-results2/17797_" + self.name + "/proc_default_deepCR/17797_" + self.name + "_mask.fits"
        mask_path = self.photometry_path / f"17797_{self.name}_mask.fits"
        with fits.open(mask_path) as hdul:
            # choose the HDU that actually holds the mask array; 0 or 1 are common
            mask_hdu = hdul[0] if hdul[0].data is not None else hdul[1]
            #mask_data = np.asarray(mask_hdu.data)
            self.wcs_mask = WCS(mask_hdu.header)
            self.pixel_mask = np.asarray(mask_hdu.data)
            self.header = mask_hdu.header
        print(f"Masks for {self.name} loaded sucessfully")

    def clean_asts(self):
        "remove bad pixels where the magnitude in the ASTs where the magnitude = 99.99"
        self.asts = clean_asts(self.asts)

    def clean_data(self):
        #1. Apply the photometric culls, extinction correction
        self.data = cull_data(self.data, Data=True) #apply photometric cuts as part of stap 2
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
        ny, nx = self.pixel_mask.shape
        in_bounds = (xpix >= 0) & (xpix < nx) & (ypix >= 0) & (ypix < ny)
        print('Found all valid pixel locations')

        # 4) Sample the mask at integer pixel indices (choose rounding mode)
        #floor is common; you can also use np.rint for nearest-neighbor
        xi = np.floor(xpix[in_bounds]).astype(int)
        yi = np.floor(ypix[in_bounds]).astype(int)
        print('Apply pixel limits from .fits file!')

        # 5) Define what "masked" means. Here: non-zero == BAD
        bad = np.zeros_like(in_bounds, dtype=bool) #create an array of True Values
        bad[in_bounds] = self.pixel_mask[yi, xi] == 0   # flip to ==0 if your mask encodes “good==1”
        print('Apply `Masked = 1` to the dataset')

        # 6) Final keep mask = not bad (and in bounds). If out-of-bounds, you can choose to drop or keep.
        keep = in_bounds & (~bad)
        #self.data = self.data[keep]
        self.data_masked = self.data.loc[keep].copy()
        print('Applied masks to dataset')

    # def apply_mask(self):
    #     """
    #     Apply:
    #     1. Existing FITS pixel mask
    #     2. Central chip-gap exclusion strip
    #     3. Candidate-specific spiral-arm polygon mask
    #     """
    
    #     # --------------------------------------------------
    #     # Convert catalog RA/Dec to mask-image pixel positions
    #     # --------------------------------------------------
    #     ra = np.asarray(self.data["ra"])
    #     dec = np.asarray(self.data["dec"])
    
    #     pix = self.wcs_mask.all_world2pix(
    #         np.column_stack([ra, dec]),
    #         0,  # zero-indexed pixel coordinates
    #     )
    
    #     xpix = pix[:, 0]
    #     ypix = pix[:, 1]
    
    #     print("Recovered X, Y pixels from the photometry")
    
    #     # --------------------------------------------------
    #     # Keep sources within FITS-mask boundaries
    #     # --------------------------------------------------
    #     ny, nx = self.pixel_mask.shape
    
    #     in_bounds = (
    #         (xpix >= 0)
    #         & (xpix < nx)
    #         & (ypix >= 0)
    #         & (ypix < ny)
    #     )
    
    #     print("Found valid pixel locations")
    
    #     # Integer pixel positions for looking up the FITS mask
    #     xi = np.floor(xpix[in_bounds]).astype(int)
    #     yi = np.floor(ypix[in_bounds]).astype(int)
    
    #     # --------------------------------------------------
    #     # Begin with every source marked as not bad
    #     # --------------------------------------------------
    #     bad = np.zeros(len(self.data), dtype=bool)
    
    #     # Existing FITS mask:
    #     # Assumes pixel_mask == 1 means usable/good
    #     # and pixel_mask == 0 means masked/bad.
    #     bad[in_bounds] = self.pixel_mask[yi, xi] == 0
    
    #     # --------------------------------------------------
    #     # Mask the central chip-gap / problematic strip -- Amend here later if I only want to do certain ones
    #     # --------------------------------------------------
    #     ylo, yhi = 1750, 2250
    
    #     in_chip_gap = (
    #         in_bounds
    #         & (ypix >= ylo)
    #         & (ypix <= yhi)
    #     )
    
    #     bad |= in_chip_gap
    
    #     print(
    #         f"Chip-gap strip mask: removed {in_chip_gap.sum()} "
    #         f"sources with {ylo} <= y <= {yhi}"
    #     )
    
    #     # --------------------------------------------------
    #     # Candidate-specific spiral-arm polygon mask
    #     # --------------------------------------------------
    #     if self.name == "DW1909M6341":
    
    #         spiral_arm_poly = np.array([
    #             [2250, 1750],
    #             [2750, 1750],
    #             [3000, 3500],
    #             [3000, 4500],
    #             [1750, 4000],
    #             [1750, 3250],
    #             [2750, 3000],
    #             [2500, 3000],
    #             [2000, 2000],
    #         ])
    
    #         arm_path = MplPath(spiral_arm_poly)
    
    #         # xpix and ypix are aligned one-to-one with self.data rows
    #         xy_points = np.column_stack([xpix, ypix])
    
    #         in_spiral_arm = (
    #             in_bounds
    #             & arm_path.contains_points(xy_points)
    #         )
    
    #         bad |= in_spiral_arm
    
    #         print(
    #             f"Spiral-arm polygon mask: removed "
    #             f"{in_spiral_arm.sum()} sources"
    #         )
    
    #     # --------------------------------------------------
    #     # Final retained catalog
    #     # --------------------------------------------------
    #     keep = in_bounds & (~bad)
    
    #     self.data_masked = self.data.loc[keep].copy()
    
    #     # Optional: retain pixel coordinates and mask flags for diagnostics
    #     self.data_masked["x_mask_pix"] = xpix[keep]
    #     self.data_masked["y_mask_pix"] = ypix[keep]
    
    #     self.mask_keep = keep
    #     self.mask_bad = bad
    #     self.xpix_all = xpix
    #     self.ypix_all = ypix
    
    #     print(f"Applied masks to dataset")
    #     print(f"Retained {keep.sum()} / {len(self.data)} sources")
    
    #     return self.data_masked

    # def apply_SkyCoord(self, frame='icrs'):
    #     if 
    #     self.SkyCoord = SkyCoord(self.coords[0], self.coords[1], unit=['deg', 'deg'], frame=frame)

    # def apply_SkyCoord(self, attr="coords", ra_col="ra", dec_col="dec", frame="icrs"):
    #     obj = getattr(self, attr)

    #     if attr == "coords":
    #         self.cand_center = SkyCoord(obj[0], obj[1], unit=("deg", "deg"), frame=frame)
    #     else:
    #         skycoord = SkyCoord(obj[ra_col], obj[dec_col], unit=("deg", "deg"), frame=frame)

    #     setattr(self, f"{attr}_SkyCoord", skycoord)

    #     #return skycoord

    def get_results_file(self, filename, subfolder=None):
        """
        Return a full output path inside this dwarf's results folder.

        Examples
        --------
        self.get_results_file("cmd.png")
        self.get_results_file("selected_rgb.csv", subfolder="tables")
        self.get_results_file("spatial_selection.png", subfolder="plots")
        """

        if subfolder is None:
            output_dir = self.results_path
        else:
            output_dir = self.results_path / subfolder

        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir / filename

    def save_dataframe(self, df, filename, subfolder="tables", index=False):
        path = self.get_results_file(filename, subfolder=subfolder)
        df.to_csv(path, index=index)
        print(f"Saved table to: {path}")
        return path

    def save_figure(self, fig, filename, subfolder="plots", dpi=300):
        path = self.get_results_file(filename, subfolder=subfolder)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to: {path}")
        return path
        
    def get_masked_coords(self):
        """
        Use the mask.fits files and the dwarf coordinates
        """
        # Build full pixel grid
        x = np.arange(self.header["NAXIS1"])
        y = np.arange(self.header["NAXIS2"])
        X, Y = np.meshgrid(x, y)

        # Convert every pixel to RA/Dec using the WCS Astropy Package
        ra_grid, dec_grid = self.wcs_mask.wcs_pix2world(X, Y, 0)

        # Center of candidate
        #cand_center = SkyCoord(curr_dwarf.coords[0], curr_dwarf.coords[1], unit="deg", frame="icrs")

        # SkyCoord for every pixel
        pix_coords = SkyCoord(ra_grid * u.deg, dec_grid * u.deg, frame="icrs")

        # Convert to offset coordinates around dwarf center
        ## assumes the speeration is already in arcminutes!
        self.deltadec_masked, self.deltara_masked = get_separation(
        self.cand_center, pix_coords
        )

    def find_pixs_in_radius(self, reader = 'h5', factor=2):
        """
        If dwarf.profile is valid, use MCMC structural fit and elliptical radius.
        Otherwise, use catalog effective radius as a circular fallback.

        delta_ra and delta_dec must be in the same units as the radius being used.
        """

        valid_profiles = {"sersic", "exponential", "plummer", "king"}

        # if self.profile in valid_profiles:

        #     # bstval, param_errs = get_mcmc_structural_params(
        #     # self.profile, self.name, output="bstval", rh_output=False,
        #     # reader=reader)

        #     self.bstval, self.param_errs = get_mcmc_structural_params(self.profile, self.name, output="bstval", 
        #                                                               rh_output=False, reader=reader)

        #     if self.profile == "sersic":
        #         x0, y0, rh, e, PA, N_star, n = self.bstval

        #     elif self.profile == "exponential":
        #         x0, y0, rh, e, PA, N_star = self.bstval

        #     elif self.profile == "plummer":
        #         x0, y0, rh, e, PA, N_star = self.bstval

        #     elif self.profile == "king":
        #         x0, y0, rc, rt, e, PA, N_star = self.bstval
        #         rh = rh_analytical_king(rc, rt)

        #     r = elliptical_radius(#deltara, deltadec,
        #         self.deltara_masked, self.deltadec_masked,
        #     self.bstval,
        #     profile=self.profile
        #     )

        #     self.half_light_radius = rh
        #     self.radius_mode = "structural_fit"
        #     self.radius_used_arcmin = rh
            
        #     self.inside_radius = r <= factor * self.half_light_radius

            #return inside #, {"mode": "structural_fit", "profile": dwarf.profile, "radius_used": rh, "bstval": self.bstval}

        # else:
        if self.eff_rad_pc is None or np.isnan(self.eff_rad_pc):
            raise ValueError(f"No structural fit or effective radius available for {self.name}")

        #r_circ = np.sqrt(deltara**2 + deltadec**2)
        r_circ = np.sqrt(self.deltara_masked**2 + self.deltadec_masked**2)

        self.radius_mode = "effective_radius_fallback"
        self.radius_used_arcmin = self.eff_rad_arcmin

        #convert the radius to arcminutes
        self.eff_rad_arcmin = pc_to_arcmin(self.eff_rad_pc, self.distance)
        
        #define my target region using the effective radius
        self.inside_radius = r_circ <= factor * self.eff_rad_arcmin #dwarf.effective_radius

            #return  self.pixs_in_rad #, {"mode": "effective_radius_fallback", "profile": None, 
                #"radius_used": dwarf.effective_radius, "bstval": None}

    def find_area_from_pixs(self):
        #define regions from pixels
        self.good_pixels = (self.pixel_mask != 0)
        self.target_px = self.good_pixels & self.inside_radius #inside_2rh_pixels
        self.bg_px = self.good_pixels & (~self.inside_radius) #(~inside_2rh_pixels)

        #count the number of useable pixels
        self.N_target_px = np.count_nonzero(self.target_px)  #(inside_2rh_pixels)
        self.N_bg_px = np.count_nonzero(self.bg_px)   #(background_pixels)

        #find the area in arcminutes squared per pixel using the rotation matrix from the header
        area_per_px = np.abs(self.header["CD1_1"] * self.header["CD2_2"] -
                             self.header["CD1_2"] * self.header["CD2_1"]) * 60.0**2 

        #find the total area in arcminutes square from pixels
        self.target_area = area_per_px * self.N_target_px
        self.bg_area = area_per_px * self.N_bg_px
        self.total_area = self.target_area + self.bg_area

        print(f"Target pixels      = {self.N_target_px}")
        print(f"Background pixels  = {self.N_bg_px}")
        print(f"Total pixels =       {self.N_target_px + self.N_bg_px}")
        print(f"Target area     = {self.target_area:.4f} arcmin^2")
        print(f"Background area = {self.bg_area:.4f} arcmin^2")
        print(f"Total area = {(self.bg_area + self.target_area):.4f} arcmin^2")

    def plot_target_and_bg_pixels(self):
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))

        axes[0].imshow(self.good_pixels, origin="lower")
        axes[0].set_title("Good HST pixels")

        #change title based on Profile radius vs. Effective radius
        # if self.profile is not None:
        if self._has_structural_profile():
            axes[1].imshow(self.target_px, origin="lower")
            axes[1].set_title(r"Target pixels: $R_{\rm ell} \leq 2r_h$")

            axes[2].imshow(self.bg_px, origin="lower")
            axes[2].set_title(r"Background pixels: $R_{\rm ell} > 2r_h$")
            
        else: 
            axes[1].imshow(self.target_px, origin="lower")
            axes[1].set_title(r"Target pixels: $R_{\rm circ} \leq 2r_eff$")

            axes[2].imshow(self.bg_px, origin="lower")
            axes[2].set_title(r"Background pixels: $R_{\rm circ} > 2r_eff$")
            
        for ax in axes:
            ax.set_xlabel("X [pix]")
            ax.set_ylabel("Y [pix]")
            ax.set_aspect("equal")

        fig.tight_layout()
        #plt.show()
        
        return fig, axes

    # def count_stars_in_regions(self, frame='icrs'):
    #         star_df = curr_dwarf.data_masked.copy()
        
    #         star_coords = SkyCoord(
    #                     star_df["ra"].values,
    #                     star_df["dec"].values,
    #                     unit="deg",
    #                     frame=frame)
        
    #     self.cand_center = SkyCoord(self.coords[0], self.coords[1], unit=['deg', 'deg'], frame=frame)

    #         deltadec_stars, deltara_stars = cs.Half_Light_Radius.get_separation(
    #                 self.,
    #             star_coords)

    def _has_structural_profile(self): #helper function ot better handle having a profile vs. no profile
        valid_profiles = {"sersic", "exponential", "plummer", "king"}
        return self.profile in valid_profiles

    def _get_rh_from_bstval(self, bstval):
        if self.profile == "sersic":
            x0, y0, rh, e, PA, N_star, n = self.bstval

        elif self.profile == "exponential":
            x0, y0, rh, e, PA, N_star = self.bstval

        elif self.profile == "plummer":
            x0, y0, rh, e, PA, N_star = self.bstval

        elif self.profile == "king":
            x0, y0, rc, rt, e, PA, N_star = self.bstval
            rh = rh_analytical_king(rc, rt)

        else:
            raise ValueError(f"Invalid profile: {self.profile}")

        return rh

    ### REworkign of count_Stars_in_regions
    #def count_stars_in_regions(self, factor=2.0, data_attr="data_masked", ra_col="ra", dec_col="dec", frame="icrs", reader="h5"):
    # def count_stars_in_regions(self, factor=2.0, data_attr="data_masked", ra_col="ra", dec_col="dec", frame="icrs", reader="h5"):
        
    #     """
    #     Count stars inside and outside the target region.

    #     If self.profile is valid, use the structural fit and elliptical radius.
    #     If self.profile is None/invalid, use circular radius from effective radius.
    #     """

    #     star_df = getattr(self, data_attr, None)

    #     if star_df is None:
    #         raise ValueError(f"self.{data_attr} is None. Run the step that creates it first.")

    #     star_df = star_df.copy()
    #     star_coords = SkyCoord( star_df[ra_col].values, star_df[dec_col].values, unit="deg", frame=frame)

    # # cand_center = SkyCoord(
    # #     self.coords[0],
    # #     self.coords[1],
    # #     unit="deg",
    # #     frame=frame
    # # )

    #     self.deltadec_stars, self.deltara_stars = get_separation(
    #         self.cand_center, star_coords)

    #     if self._has_structural_profile():

    #         self.bstval, self.param_errs = get_mcmc_structural_params(
    #         self.profile,
    #         self.name,
    #         output="bstval",
    #         rh_output=False,
    #         reader=reader)

    #         self.half_light_radius = self._get_rh_from_bstval(self.bstval)

    #         r_stars = elliptical_radius(
    #         self.deltara_stars,
    #         self.deltadec_stars,
    #         self.bstval,
    #         profile=self.profile)

    #         self.star_in_target = r_stars <= factor * self.half_light_radius
    #         radius_used = self.half_light_radius
    #         mode = "structural_fit"

    #     else:
    #         if self.eff_rad_arcmin is None:
    #             self.eff_rad_arcmin = pc_to_arcmin(self.eff_rad_pc, self.distance)

    #         r_stars = np.sqrt(self.deltara_stars**2 + self.deltadec_stars**2)

    #         self.star_in_target = r_stars <= factor * self.eff_rad_arcmin
    #         radius_used = self.eff_rad_arcmin
    #         mode = "effective_radius_fallback"

    #     self.star_in_background = ~self.star_in_target

    #     self.target_star_df = star_df.loc[self.star_in_target].copy()
    #     self.background_star_df = star_df.loc[self.star_in_background].copy()

    #     self.N_target_stars = len(self.target_star_df)
    #     self.N_background_stars = len(self.background_star_df)

    #     print(f"Mode used: {mode}")
    #     print(f"Radius used: {radius_used:.4f} arcmin")
    #     print(f"Stars inside {factor} radius     = {self.N_target_stars}")
    #     print(f"Stars outside {factor} radius    = {self.N_background_stars}")

    def count_stars_in_regions(self, factor=2.0, data_attr="data_masked", ra_col="ra", dec_col="dec", frame="icrs", reader="h5"):
        
        """
        Count stars inside and outside the target region.

        If self.profile is valid, use the structural fit and elliptical radius.
        If self.profile is None/invalid, use circular radius from effective radius.
        """

        star_df = getattr(self, data_attr, None)

        if star_df is None:
            raise ValueError(f"self.{data_attr} is None. Run the step that creates it first.")

        star_df = star_df.copy()
        star_coords = SkyCoord( star_df[ra_col].values, star_df[dec_col].values, unit="deg", frame=frame)

    # cand_center = SkyCoord(
    #     self.coords[0],
    #     self.coords[1],
    #     unit="deg",
    #     frame=frame
    # )

        self.deltadec_stars, self.deltara_stars = get_separation(
            self.cand_center, star_coords)

        # if self._has_structural_profile():

        #     self.bstval, self.param_errs = get_mcmc_structural_params(
        #     self.profile,
        #     self.name,
        #     output="bstval",
        #     rh_output=False,
        #     reader=reader)

        #     self.half_light_radius = self._get_rh_from_bstval(self.bstval)

        #     r_stars = elliptical_radius(
        #     self.deltara_stars,
        #     self.deltadec_stars,
        #     self.bstval,
        #     profile=self.profile)

        #     self.star_in_fit_target = r_stars <= factor * self.half_light_radius
        #     radius_used = self.half_light_radius
        #     mode = "structural_fit"

        # else:
        if self.eff_rad_arcmin is None:
            self.eff_rad_arcmin = pc_to_arcmin(self.eff_rad_pc, self.distance)

        r_stars = np.sqrt(self.deltara_stars**2 + self.deltadec_stars**2)

        self.star_in_target = r_stars <= factor * self.eff_rad_arcmin
        radius_used = self.eff_rad_arcmin
        mode = "effective_radius_fallback"

        self.star_in_background = ~self.star_in_target

        self.target_star_df = star_df.loc[self.star_in_target].copy()
        self.background_star_df = star_df.loc[self.star_in_background].copy()

        self.N_target_stars = len(self.target_star_df)
        self.N_background_stars = len(self.background_star_df)

        print(f"Mode used: {mode}")
        print(f"Radius used: {radius_used:.4f} arcmin")
        print(f"Stars inside {factor} radius     = {self.N_target_stars}")
        print(f"Stars outside {factor} radius    = {self.N_background_stars}")

    def filter_target_below_trgb(self, trgb_f814w_app, tolerance=0.5, f814_col="acs_f814w_vega"):
        """
        Filter stars inside the target aperture to keep only stars below/fainter
        than the TRGB in F814W.

        Requires self.target_star_df from self.count_stars_in_regions().

        Old tolerance: 0.2
        More realistic tolerance: 0.5
        """

        if not hasattr(self, "target_star_df"):
            raise ValueError("Run self.count_stars_in_regions() first.")

        below_trgb_filter = self.target_star_df[f814_col] >= (trgb_f814w_app - tolerance)

        self.target_below_trgb_df = self.target_star_df.loc[below_trgb_filter].copy()
        self.trgb_f814w_app = trgb_f814w_app
        self.N_target_below_trgb = len(self.target_below_trgb_df)

        print(f"TRGB F814W apparent magnitude: {self.trgb_f814w_app:.3f}")
        print(f"Stars inside target aperture: {len(self.target_star_df)}")
        print(f"Stars below TRGB: {self.N_target_below_trgb}")

        return self.target_below_trgb_df

    def filter_all_below_trgb(self, trgb_f814w_app, tolerance=0.5, f814_col="acs_f814w_vega"):
        """
        Filter all cleaned/masked stars to keep only stars fainter than/below
        the TRGB limit.
    
        This creates self.obs_below_trgb, which can then be passed into
        count_stars_in_regions(data_attr="obs_below_trgb").
        """
    
        if self.data_masked is None:
            raise ValueError("self.data_masked is None. Run clean_data() and apply_mask() first.")
    
        below_trgb_filter = self.data_masked[f814_col] >= (trgb_f814w_app - tolerance)
    
        self.obs_below_trgb = self.data_masked.loc[below_trgb_filter].copy()
        self.trgb_f814w_app = trgb_f814w_app
        self.N_obs_below_trgb = len(self.obs_below_trgb)
    
        print(f"TRGB F814W apparent magnitude: {self.trgb_f814w_app:.3f}")
        print(f"Total stars after photometric culls/masking: {len(self.data_masked)}")
        print(f"Total stars below TRGB: {self.N_obs_below_trgb}")
    
        return self.obs_below_trgb

    # def plot_spatial_rgb_selection(self, factor=2.0):
    #     """
    #     Plot all spatially selected stars and potential RGB stars in separation space.

    #     Requires:
    #     - self.deltara_stars, self.deltadec_stars from count_stars_in_regions()
    #     - self.deltara_rgb, self.deltadec_rgb from selecting potential RGB stars
    #     - self.bstval if a structural profile exists
    #     """

    #     fig, ax = plt.subplots(figsize=(9, 8))

    #     pts_all_kwargs = dict(s=10.0, lw=0.2, fc="silver", ec="black", alpha=0.7, zorder=10)

    #     pts_rgb_kwargs = dict(s=10.0, lw=0.2, fc="magenta", ec="red", alpha=0.85, zorder=11)

    #     # ax.scatter(self.deltara_stars, self.deltadec_stars, label="All stars", **pts_all_kwargs)
    #     all_coords = SkyCoord(self.data_masked["ra"], self.data_masked["dec"], unit=("deg", "deg"), frame="icrs")

    #     self.deltadec_all, self.deltara_all = get_separation(self.cand_center, all_coords)

    #     ax.scatter(self.deltara_all, self.deltadec_all, label="All stars", **pts_all_kwargs)

    #     if hasattr(self, "deltara_rgb") and hasattr(self, "deltadec_rgb"):
    #         ax.scatter(self.deltara_rgb, self.deltadec_rgb, label="Potential RGB stars", **pts_rgb_kwargs)

    #     # Plot structural ellipse if available
    #     if self._has_structural_profile() and hasattr(self, "bstval"):
    #         ellipse_kwargs = dict(fc="none", ls="-", lw=1.5, alpha=0.95, zorder=9, ec="tab:blue")

    #         plot_density_ellipse(ax, self.profile, self.bstval, ellipse_kwargs=ellipse_kwargs, radii=[factor])

    #         profile_label = f"{self.profile} fit"

    #     else:
    #         # Circular effective-radius fallback
    #         if self.eff_rad_arcmin is None:
    #             self.eff_rad_arcmin = pc_to_arcmin(self.eff_rad_pc, self.distance)

    #         circ = plt.Circle((0, 0), factor * self.eff_rad_arcmin, fill=False, lw=1.5, alpha=0.95, ec="tab:blue", zorder=9)

    #         ax.add_patch(circ)
    #         profile_label = "effective radius fallback"

    #     ax.text(0.05, 0.95, self.name, transform=ax.transAxes, fontsize=16, 
    #             bbox=dict(facecolor="white", edgecolor="red", boxstyle="round,pad=0.4"), verticalalignment="top",zorder=12)

    #     ax.text(0.05, 0.88, profile_label, transform=ax.transAxes, fontsize=12,
    #         bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"), verticalalignment="top", zorder=12)

    #     ax.set_xlabel(r"$\Delta \alpha$ (arcmin)")
    #     ax.set_ylabel(r"$\Delta \delta$ (arcmin)")
    #     ax.set_aspect("equal")
    #     ax.invert_xaxis()
    #     ax.legend()

    #     fig.tight_layout()

    #     return fig, ax

    def select_potential_rgb_stars(self, isochs_filter, bias_interp_814, scatter_interp_814, bias_interp_606, scatter_interp_606,
        mag1_cut=(25.5, 28.0), mag2_cut=(24.5, 27.5), radius=0.1, err_factor=1.0, max_stage=6, include_hb=True, frame="icrs"):
        """
        Select potential RGB stars using photometric uncertainties and isochrone path.
        """

        if self.data_masked is None:
            raise ValueError("self.data_masked is None. Run clean_data() and apply_mask() first.")

        F814W_fit_sys = bias_interp_814(self.data_masked["acs_f814w_vega"])
        F814W_fit_stat = scatter_interp_814(self.data_masked["acs_f814w_vega"])
        F606W_fit_sys = bias_interp_606(self.data_masked["acs_f606w_vega"])
        F606W_fit_stat = scatter_interp_606(self.data_masked["acs_f606w_vega"])

        F814W_tot_err = np.sqrt(F814W_fit_sys**2 + F814W_fit_stat**2)
        F606W_tot_err = np.sqrt(F606W_fit_sys**2 + F606W_fit_stat**2)

        isochs_filter_kwargs = dict(mag1_cut=mag1_cut, mag2_cut=mag2_cut, radius=radius, err_factor=err_factor, max_stage=max_stage,
        include_hb=include_hb)

        iso_filter = cut_isochrones_path(self.data_masked["acs_f606w_vega"], self.data_masked["acs_f814w_vega"], F606W_tot_err,
        F814W_tot_err, isochs_filter, self.dmod, **isochs_filter_kwargs)

        self.pts_sources_rgb = self.data_masked.loc[iso_filter].copy()

        print(
            f"Potential RGB stars based on systematic/statistical errors: "
            f"{len(self.pts_sources_rgb)}")

        potential_rgb_coords = SkyCoord(self.pts_sources_rgb["ra"], self.pts_sources_rgb["dec"], unit=("deg", "deg"), frame=frame)

        self.deltadec_rgb, self.deltara_rgb = get_separation(self.cand_center, potential_rgb_coords)

        return self.pts_sources_rgb

    # return {
    #     "mode": mode,
    #     "radius_used_arcmin": radius_used,
    #     "factor": factor,
    #     "N_target_stars": self.N_target_stars,
    #     "N_background_stars": self.N_background_stars,
    #     "target_star_df": self.target_star_df,
    #     "background_star_df": self.background_star_df
    # }

    def get_statistics(self):
        self.bgnd_density = self.N_background_stars / self.bg_area
        self.expected_bg_in_target = self.bgnd_density * self.target_area
        
        print(f"We expected {self.expected_bg_in_target} stars from the background density, but we actually see {self.N_target_stars} stars")

        #Use the Poisson stats function to get the final probability
        self.p_val = Poisson_stats(self.N_target_stars, self.expected_bg_in_target)

    
    def plot_spatial_rgb_selection(self, factor=2.0):
        """
        Plot spatial distribution of non-RGB stars and potential RGB stars
        in separation space.
    
        This avoids plotting all stars underneath RGB stars, which can make
        the whole plot look magenta when the RGB-selected fraction is large.
    
        Requires:
        - self.data_masked
        - self.pts_sources_rgb from select_potential_rgb_stars()
        - self.deltara_rgb, self.deltadec_rgb from select_potential_rgb_stars()
        - self.bstval if a structural profile exists
        """
    
        fig, ax = plt.subplots(figsize=(9, 8))
    
        pts_non_rgb_kwargs = dict(
            s=10.0,
            lw=0.2,
            fc="silver",
            ec="black",
            alpha=0.75,
            zorder=10,
        )
    
        pts_rgb_kwargs = dict(
            s=10.0,
            lw=0.2,
            fc="magenta",
            ec="red",
            alpha=0.80,
            zorder=11,
        )
    
        # Build non-RGB catalog as data_masked minus pts_sources_rgb
        if hasattr(self, "pts_sources_rgb"):
            rgb_idx = self.pts_sources_rgb.index
            non_rgb_df = self.data_masked.loc[
                ~self.data_masked.index.isin(rgb_idx)
            ].copy()
        else:
            non_rgb_df = self.data_masked.copy()
    
        # Plot non-RGB stars
        if len(non_rgb_df) > 0:
            non_rgb_coords = SkyCoord(
                non_rgb_df["ra"],
                non_rgb_df["dec"],
                unit=("deg", "deg"),
                frame="icrs",
            )
    
            self.deltadec_non_rgb, self.deltara_non_rgb = get_separation(
                self.cand_center,
                non_rgb_coords,
            )
    
            ax.scatter(
                self.deltara_non_rgb,
                self.deltadec_non_rgb,
                label="Non-RGB stars",
                **pts_non_rgb_kwargs,
            )
    
        # Plot potential RGB stars
        if hasattr(self, "deltara_rgb") and hasattr(self, "deltadec_rgb"):
            ax.scatter(
                self.deltara_rgb,
                self.deltadec_rgb,
                label="Potential RGB stars",
                **pts_rgb_kwargs,
            )
    
        # Plot structural ellipse if available
        # if self._has_structural_profile() and hasattr(self, "bstval"):
        if (self._has_structural_profile() and getattr(self, "bstval", None) is not None):
            ellipse_kwargs = dict(
                fc="none",
                ls="-",
                lw=1.5,
                alpha=0.95,
                zorder=9,
                ec="tab:blue",
            )
    
            plot_density_ellipse(
                ax,
                self.profile,
                self.bstval,
                ellipse_kwargs=ellipse_kwargs,
                radii=[factor],
            )
    
            profile_label = f"{self.profile} fit"
    
        else:
            # Circular effective-radius fallback
            if self.eff_rad_arcmin is None:
                self.eff_rad_arcmin = pc_to_arcmin(
                    self.eff_rad_pc,
                    self.distance,
                )
    
            circ = plt.Circle(
                (0, 0),
                factor * self.eff_rad_arcmin,
                fill=False,
                lw=1.5,
                alpha=0.95,
                ec="tab:blue",
                zorder=9,
            )
    
            ax.add_patch(circ)
            profile_label = "effective radius fallback"
    
        # Text labels
        ax.text(
            0.05,
            0.95,
            self.name,
            transform=ax.transAxes,
            fontsize=16,
            bbox=dict(
                facecolor="white",
                edgecolor="red",
                boxstyle="round,pad=0.4",
            ),
            verticalalignment="top",
            zorder=12,
        )
    
        ax.text(
            0.05,
            0.88,
            profile_label,
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="round,pad=0.3",
            ),
            verticalalignment="top",
            zorder=12,
        )
    
        # Optional count label
        n_rgb = len(self.pts_sources_rgb) if hasattr(self, "pts_sources_rgb") else 0
        n_non_rgb = len(non_rgb_df)
    
        ax.text(
            0.05,
            0.81,
            f"N RGB = {n_rgb}\nN non-RGB = {n_non_rgb}",
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="round,pad=0.3",
            ),
            verticalalignment="top",
            zorder=12,
        )
    
        ax.set_xlabel(r"$\Delta \alpha$ (arcmin)")
        ax.set_ylabel(r"$\Delta \delta$ (arcmin)")
        ax.set_aspect("equal")
        ax.invert_xaxis()
        ax.legend()
    
        fig.tight_layout()
    
        return fig, ax

    def plot_spatial_counts(self, factor=2.0):
        """
        Plot all masked/cleaned stars in the field, then overplot all stars
        below the TRGB.
    
        Requires:
        - self.data_masked from clean_data() + apply_mask()
        - self.obs_below_trgb from filter_all_below_trgb()
        - self.count_stars_in_regions(data_attr="obs_below_trgb") if you want
          N_target, N_expected, and p-value in the title.
        """
    
        if self.data_masked is None:
            raise ValueError("self.data_masked is None. Run clean_data() and apply_mask() first.")
    
        if not hasattr(self, "obs_below_trgb") or self.obs_below_trgb is None:
            raise ValueError("Run filter_all_below_trgb() before plot_spatial_counts().")
    
        fig, ax = plt.subplots(figsize=(10, 7))
    
        # --------------------------------------------------
        # 1. All masked/cleaned stars in the HST field
        # --------------------------------------------------
        all_coords = SkyCoord(
            self.data_masked["ra"],
            self.data_masked["dec"],
            unit=("deg", "deg"),
            frame="icrs",
        )
    
        deltadec_all, deltara_all = get_separation(
            self.cand_center,
            all_coords,
        )
    
        ax.scatter(
            deltara_all,
            deltadec_all,
            s=10,
            c="lightgray",
            edgecolor="k",
            linewidth=0.2,
            alpha=0.7,
            label=f"All masked stars: N={len(self.data_masked)}",
        )
    
        # --------------------------------------------------
        # 2. All stars below the TRGB
        # --------------------------------------------------
        below_trgb_coords = SkyCoord(
            self.obs_below_trgb["ra"],
            self.obs_below_trgb["dec"],
            unit=("deg", "deg"),
            frame="icrs",
        )
    
        deltadec_below_trgb, deltara_below_trgb = get_separation(
            self.cand_center,
            below_trgb_coords,
        )
    
        ax.scatter(
            deltara_below_trgb,
            deltadec_below_trgb,
            s=16,
            c="magenta",
            edgecolor="red",
            linewidth=0.3,
            alpha=0.9,
            label=f"All stars below TRGB: N={len(self.obs_below_trgb)}",
        )
    
        # --------------------------------------------------
        # 3. Circular aperture
        # --------------------------------------------------
        if self.eff_rad_arcmin is None:
            self.eff_rad_arcmin = pc_to_arcmin(self.eff_rad_pc, self.distance)
    
        circ = plt.Circle(
            (0, 0),
            factor * self.eff_rad_arcmin,
            fill=False,
            ec="tab:blue",
            lw=2.0,
            alpha=0.95,
        )
    
        ax.add_patch(circ)
        aperture_label = rf"Circular aperture: $R \leq {factor} R_{{\rm eff}}$"
    
        # --------------------------------------------------
        # 4. Title/counts
        # --------------------------------------------------
        if hasattr(self, "N_target_stars") and hasattr(self, "expected_bg_in_target") and hasattr(self, "p_val"):
            title = (
                f"{self.name}: below-TRGB spatial counts\n"
                f"N_target={self.N_target_stars}, "
                f"N_exp={self.expected_bg_in_target:.2f}, "
                f"p={self.p_val:.2e}"
            )
        else:
            title = f"{self.name}: all masked stars and below-TRGB stars"
    
        ax.set_title(title)
    
        ax.text(
            0.05,
            0.95,
            aperture_label,
            transform=ax.transAxes,
            fontsize=11,
            va="top",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )
    
        ax.set_xlabel(r"$\Delta \alpha$ (arcmin)")
        ax.set_ylabel(r"$\Delta \delta$ (arcmin)")
        ax.set_aspect("equal")
        ax.invert_xaxis()
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
    
        fig.tight_layout()
    
        return fig, ax