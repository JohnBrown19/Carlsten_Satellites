#started 2/19/26

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
        Use teh AST culls (same values, but the columsn have different names) if False

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