import numpy as np
import sep
def estimate_aperture_photometry_err(flux,aper_radius,bkg,x_list,y_list):
    bkg_rms = bkg.rms()
    bkg_mean = bkg.back()
    err_list = []
    for i,x,y in zip(range(len(x_list)),x_list,y_list):
        x = round(x)
        y = round(y)
        
        gain = bkg_mean[y,x]/bkg_rms[y,x]
        var_photon = np.abs(flux[i]/gain)
        var_background = bkg_rms[y,x]**2 * aper_radius**2 * np.pi
        var_total = var_photon + var_background
        # print(var_photon,var_background,var_total)
        err = np.sqrt(var_total)
        err_list.append(err)
    return np.array(err_list)