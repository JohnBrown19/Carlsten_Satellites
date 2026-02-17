import numpy as np
import pandas as pd
from astropy.io import ascii

class Dwarf:
    # ---------- Defaults shared by ALL dwarfs ----------
    DEFAULT_DATA_MASK_KW = dict(
        snr606_min=4.0,
        snr814_min=4.0,
        err606_max=3.0,
        err814_max=3.0,
        sharp2_606_max=0.2,
        sharp2_814_max=0.2,
        crowd606_max=0.15,
        crowd814_max=0.15,
        objtype_max=2,
        vega606_max=99.0,
        vega814_max=99.0,
    )

    DEFAULT_AST_MASK_KW = dict(
        m606_in_max=99.0,
        m814_in_max=99.0,
    )

    ast_cols = ['ext','chip','X_in','Y_in','RA_in','Dec_in','X_out','Y_out','RA_out',
                'Dec_out', 'chi_fit','snr_det','shp_det','rnd_det', 'dir', 'crow_det',
                'type','pass','cts_606','sky_1', 'rate_606','rateerr_606','m606_in',
                'm606_out','NOUSE_1','err_1', 'chi_1','snr_1','shp_1','rnd_1','crow_1','ef_1',
                'cts_814', 'sky','rate_814', 'rateerr_814','m814_in','m814_out','NOUSE',
                'err', 'chi','snr', 'shp','rnd', 'crow', 'ef'
               ]

    def __init__(self, name, dist_mod, logmass, Host=None, base_dir="./17797"):
        self.name = str.upper(name)
        self.dist_mod = float(dist_mod)
        self.logmass = float(logmass)
        self.Host = Host
        self.base_dir = base_dir

        # raw
        self.data = None          # pandas DF
        self.asts = None          # pandas DF

        # cleaned + cached
        self.data_clean = None
        self.ast_clean = None

        # per-instance overrides (start empty; fall back to class defaults)
        self.data_mask_overrides = {}
        self.ast_mask_overrides = {}

    # ---------- I/O (pandas only) ----------
    def load_asts(self):
        ast_tbl = ascii.read(
            f"{self.base_dir}/merged-results/17797_{self.name}_fakestars.dat",
            names=self.ast_cols
        )
        self.asts = ast_tbl.to_pandas()
        print(f"AST data for {self.name} loaded successfully (pandas).")

    def load_data(self):
        self.data = pd.read_hdf(
            f"{self.base_dir}/fake-results2/17797_{self.name}/proc_default_deepCR/17797_{self.name}.phot_full.hdf5",
            key="data"
        )
        print(f"Data for {self.name} loaded successfully (pandas).")

    # ---------- Mask configuration ----------
    def set_mask_overrides(self, data=None, ast=None):
        """
        Permanently override default mask thresholds for THIS dwarf only.
        Example:
            dwarf.set_mask_overrides(data={"crowd606_max":0.2}, ast={"m606_in_max":50.0})
        """
        if data:
            self.data_mask_overrides.update(data)
        if ast:
            self.ast_mask_overrides.update(ast)

    # ---------- Mask builders ----------
    def build_data_mask(self, overrides=None):
        if self.data is None:
            raise ValueError("Load data first: dwarf.load_data()")

        kw = dict(self.DEFAULT_DATA_MASK_KW)
        kw.update(self.data_mask_overrides)
        if overrides:
            kw.update(overrides)

        df = self.data

        mask = (
            (df['acs_f606w_snr'] > kw["snr606_min"]) &
            (df['acs_f814w_snr'] > kw["snr814_min"]) &
            (df['acs_f606w_err'] <= kw["err606_max"]) &
            (df['acs_f814w_err'] <= kw["err814_max"]) &
            ((df['acs_f606w_sharp']**2) < kw["sharp2_606_max"]) &
            ((df['acs_f814w_sharp']**2) < kw["sharp2_814_max"]) &
            (df['acs_f606w_crowd'] < kw["crowd606_max"]) &
            (df['acs_f814w_crowd'] < kw["crowd814_max"]) &
            (df['objtype_gl'] <= kw["objtype_max"]) &
            (df['acs_f606w_vega'] < kw["vega606_max"]) &
            (df['acs_f814w_vega'] < kw["vega814_max"])
        )
        return mask

    def build_ast_mask(self, overrides=None):
        if self.asts is None:
            raise ValueError("Load ASTs first: dwarf.load_asts()")

        kw = dict(self.DEFAULT_AST_MASK_KW)
        kw.update(self.ast_mask_overrides)
        if overrides:
            kw.update(overrides)

        df = self.asts
        mask = (
            (df["m606_in"] < kw["m606_in_max"]) &
            (df["m814_in"] < kw["m814_in_max"])
        )
        return mask

    # ---------- Apply masks + cache cleaned tables ----------
    def apply_quality_cuts(self, data_overrides=None, ast_overrides=None, inplace=True):
        """
        Apply default masks + per-dwarf overrides + optional call-time overrides.
        Caches cleaned dataframes as self.data_clean and self.ast_clean.
        """
        data_mask = self.build_data_mask(overrides=data_overrides)
        ast_mask  = self.build_ast_mask(overrides=ast_overrides)

        data_clean = self.data.loc[data_mask].copy()
        ast_clean  = self.asts.loc[ast_mask].copy()

        if inplace:
            self.data_clean = data_clean
            self.ast_clean  = ast_clean

        print(f"{self.name}: data_clean={len(data_clean)} rows, ast_clean={len(ast_clean)} rows")
        return data_clean, ast_clean

    def build_ast_interpolators(
        self,
        mag_min=22.0,
        mag_max=29.0,
        bin_width=0.1,
        recovered_cuts=None
        ):
        """
        Build and cache interpolators from ASTs:
          - mag -> completeness (F814W reference band)
          - mag -> bias and scatter for F606W and F814W
        Uses self.ast_clean if available; else self.asts.

        Stores:
          self.ast_mag_grid
          self.ast_comp_interp_814
          self.ast_bias_interp_814, self.ast_scatter_interp_814
          self.ast_bias_interp_606, self.ast_scatter_interp_606
        """

        if self.ast_clean is not None:
            ast_df = self.ast_clean
        elif self.asts is not None:
            ast_df = self.asts
        else:
            raise ValueError("No AST data loaded. Run load_asts() first.")

        # Default recovered cuts similar to what you used before (tweak as needed)
        if recovered_cuts is None:
            def recovered_mask(df):
                return (
                    (df['snr_1'] > 4.0) & (df['snr'] > 4.0) &
                    (df['err_1'] <= 3.0) & (df['err'] <= 3.0) &
                    ((df['shp_1']**2) < 0.1) & ((df['shp']**2) < 0.1) &
                    ((df['crow'] + df['crow_1']) < 0.3) &
                    (df['type'] <= 2)
                )
        else:
            recovered_mask = recovered_cuts

        # Bin in input F814W magnitude (typical for completeness reference)
        bins = np.arange(mag_min, mag_max + bin_width, bin_width)
        bin_centers = bins[:-1] + bin_width/2

        comp = np.zeros_like(bin_centers, dtype=float)
        bias_814 = np.full_like(bin_centers, np.nan, dtype=float)
        sig_814  = np.full_like(bin_centers, np.nan, dtype=float)
        bias_606 = np.full_like(bin_centers, np.nan, dtype=float)
        sig_606  = np.full_like(bin_centers, np.nan, dtype=float)

        # Define recovered once (full table), then slice per bin
        rec_all = recovered_mask(ast_df)

        # Compute output-input residuals for recovered stars
        # (these are what you want for bias/scatter)
        # NOTE: columns: m814_in, m814_out, m606_in, m606_out
        dm814 = (ast_df['m814_out'] - ast_df['m814_in'])
        dm606 = (ast_df['m606_out'] - ast_df['m606_in'])

        for i in range(len(bin_centers)):
            in_bin = (ast_df['m814_in'] >= bins[i]) & (ast_df['m814_in'] < bins[i+1])
            N_inj = int(np.sum(in_bin))
            if N_inj == 0:
                comp[i] = 0.0
                continue

            rec_bin = in_bin & rec_all
            N_rec = int(np.sum(rec_bin))
            comp[i] = N_rec / N_inj

            if N_rec > 5:
                # Use median for bias (robust) and robust scatter
                # (keep mean if you want, but median is usually safer with AST tails)
                bias_814[i] = np.nanmedian(dm814[rec_bin])
                bias_606[i] = np.nanmedian(dm606[rec_bin])

                # robust scatter (MAD->sigma) or std; choose one
                def robust_sigma(x):
                    med = np.nanmedian(x)
                    mad = np.nanmedian(np.abs(x - med))
                    return 1.4826 * mad

                sig_814[i] = robust_sigma(dm814[rec_bin])
                sig_606[i] = robust_sigma(dm606[rec_bin])

        # Clean up monotonicity and interpolation behavior
        # For completeness we want mag -> comp. We will:
        # - keep valid points
        valid_c = np.isfinite(comp)
        mag_c = bin_centers[valid_c]
        comp_c = comp[valid_c]

        # Completeness should generally decrease with mag; smooth by sorting mag
        sort_m = np.argsort(mag_c)
        mag_c = mag_c[sort_m]
        comp_c = comp_c[sort_m]

        # For bias/scatter, just interpolate where finite
        def make_interp(x, y, fill_low, fill_high):
            ok = np.isfinite(y) & np.isfinite(x)
            if np.sum(ok) < 2:
                # fallback: constant
                return lambda z: np.full_like(np.asarray(z, dtype=float), np.nan, dtype=float)
            xs = x[ok]
            ys = y[ok]
            s = np.argsort(xs)
            xs, ys = xs[s], ys[s]
            return interp1d(xs, ys, bounds_error=False, fill_value=(fill_low, fill_high), assume_sorted=True)

        # Cache grids and interpolators
        self.ast_mag_grid = bin_centers

        # mag -> completeness (clip to [0,1] when using)
        self.ast_comp_interp_814 = make_interp(mag_c, comp_c, fill_low=comp_c[0], fill_high=comp_c[-1])

        self.ast_bias_interp_814 = make_interp(bin_centers, bias_814, fill_low=bias_814[np.isfinite(bias_814)][0],          
                                               fill_high=bias_814[np.isfinite(bias_814)][-1])
        
        self.ast_scatter_interp_814 = make_interp(bin_centers, sig_814,  fill_low=sig_814[np.isfinite(sig_814)][0],  
                                                  fill_high=sig_814[np.isfinite(sig_814)][-1])
        
        self.ast_bias_interp_606 = make_interp(bin_centers, bias_606, fill_low=bias_606[np.isfinite(bias_606)][0],  
                                                fill_high=bias_606[np.isfinite(bias_606)][-1])
        
        self.ast_scatter_interp_606 = make_interp(bin_centers, sig_606,  fill_low=sig_606[np.isfinite(sig_606)][0],  
                                                    fill_high=sig_606[np.isfinite(sig_606)][-1])

        print(f"{self.name}: built AST interpolators on {len(bin_centers)} mag bins.")


    def get_data_colors(
        self,
        use_clean=True,
        use_obs_cols=('acs_f606w_vega', 'acs_f814w_vega'),
        mag_range=None,
        color_range=None
        ):
        """
        Return data colors (F606W - F814W) as a 1D numpy array.
        Uses self.data_clean by default.
        Optional filtering on magnitude/color ranges.
        """
        df = self.data_clean if (use_clean and self.data_clean is not None) else self.data
        if df is None:
            raise ValueError("No data loaded. Run load_data() first (and apply_quality_cuts if desired).")

        c606, c814 = use_obs_cols
        color = (df[c606] - df[c814]).to_numpy()
        m814 = df[c814].to_numpy()

        mask = np.isfinite(color) & np.isfinite(m814)

        if mag_range is not None:
            mmin, mmax = mag_range
            mask &= (m814 >= mmin) & (m814 <= mmax)

        if color_range is not None:
            cmin, cmax = color_range
            mask &= (color >= cmin) & (color <= cmax)

        return color[mask]

    def apply_ast_to_mock(self, mock_df, seed=None):
        if not hasattr(self, "ast_comp_interp_814"):
            raise ValueError("AST interpolators not built. Run dwarf.build_ast_interpolators() first.")

        return apply_ast_probabilistic_interp_df(
            mock_df,
            self.ast_comp_interp_814,
            self.ast_bias_interp_814, self.ast_scatter_interp_814,
            self.ast_bias_interp_606, self.ast_scatter_interp_606,
            seed=seed
        )

