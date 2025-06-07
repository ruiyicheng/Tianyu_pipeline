import os
import sys
sys.path.append(os.path.abspath("/home/test/workspace/Tianyu_pipeline/algorithm/code"))
import numpy as np
import pandas as pd
import astropy.io.fits as fits
import numpy.ma as ma
import sep
from astropy.time import Time
from astropy import coordinates as coord, units as u
from astropy.wcs import WCS
import utils.dataloader as dataloader
import utils.estimate_aperture_photometry_err as estimate_aperture_photometry_err
from astropy.table import Table
from astropy.modeling.fitting import LevMarLSQFitter
from photutils.psf import PSFPhotometry, IntegratedGaussianPRF
import middleware.pipeline_component as pipeline_component
class absolute_flux_calibration(pipeline_component.pipeline_component):
    def __init__(self,free = True):
        super().__init__(free=free)
        #self.data_loader = dataloader.dataloader()
    def absolute_calibration_algo(self,lightcurve_df,star_df,stacked_image_header,inst_info):
        def design_matrix(stars):
            x = np.array(stars['normalized_x']).reshape(-1,1)
            y = np.array(stars['normalized_y']).reshape(-1,1)
            g_b = np.array(stars['phot_g_mean_mag']).reshape(-1,1)-np.array(stars['phot_bp_mean_mag']).reshape(-1,1)
            g_r = np.array(stars['phot_g_mean_mag']).reshape(-1,1)-np.array(stars['phot_rp_mean_mag']).reshape(-1,1)
            ones = np.ones_like(x)
            design_matrix = np.hstack([ones,x,y,x**2,y**2,x**3,y**3,x*y,x*y**2,x**2*y,g_b,g_b**2,g_r,g_r**2,g_b*g_r])
            return design_matrix

        def calculate_prediction(coefficients,stars):
            coefficients = np.array(coefficients).reshape(-1,1)
            design_matrix_this = design_matrix(stars)
            prediction = np.dot(design_matrix_this,coefficients)
            
            return prediction
        def fit_and_filter(used_star):
            # flux calibration using Gaia Data, return the mask of outliers
            # model: mg_gaia = -2.5log10(flux_raw)+C+a1 x+b1 y+ a2 x^2 + b2 y^2 + a3 x^3 + b3 y^3 + c11 xy + c12 xy^2 + c21 x^2y + cb1(mg-mb) + cb2(mg-mb)^2 +cr1(mg-mr) + cr2(mg-mr)^2 +cbr(mg-mr)(mg-mb)
            # chi2 = (mg_gaia-mg_gaia_model)^2/sigma^2; sigma = 1/phot_g_mean_flux_over_error*2.5/log(10)
            # mask the outliers that have abs((mg_gaia-mg_gaia_model)/sigma)>3
            # return the mask and coefficients
            base_mag = -2.5*np.log10(np.array(used_star['flux']).reshape(-1,1))
            base_mag_error = np.squeeze((np.array(used_star['flux_err'])/np.array(used_star['flux']))*2.5/np.log(10))
            sigmainv = np.diag(1/base_mag_error**2)
            X = design_matrix(used_star)
            Y = np.array(used_star['phot_g_mean_mag']).reshape(-1,1)-base_mag
            Y = Y.reshape(-1,1)
            #print(X.shape,Y.shape,sigmainv.shape)
            theta = np.linalg.inv(X.T.dot(sigmainv).dot(X)).dot(X.T).dot(sigmainv).dot(Y)
            prediction = calculate_prediction(theta,used_star)
            residual = np.abs((np.squeeze(prediction)-np.squeeze(Y))/base_mag_error)
            #print(theta)
            return np.squeeze(residual>3),theta
        fov_deg = 0.1+(((inst_info['x_scale_mum']/inst_info['focal_length_mm'])*inst_info['x_resolution'])**2+((inst_info['y_scale_mum']/inst_info['focal_length_mm'])*inst_info['y_resolution'])**2)**0.5/1000/2*180/np.pi
        gaia_stars = self.data_loader.search_GDR3(inst_info['ra'],inst_info['dec'],fov_deg)
        gaia_stars = gaia_stars[['SOURCE_ID',"phot_g_mean_mag","phot_bp_mean_mag","phot_rp_mean_mag","phot_g_mean_flux_over_error"]]
        image_id_list = lightcurve_df.groupby('image_id').size().index.tolist()
        
        wcs = WCS(stacked_image_header)
        arcsec_per_pixel = inst_info['x_scale_mum']/inst_info['focal_length_mm']/1000*206265
        x, y = wcs.all_world2pix(star_df['ra'].values, star_df['dec'].values, 0)
        print(x,y)
        star_df.loc[:,'x_template'] = x
        star_df.loc[:,'y_template'] = y
        star_df['normalized_x'] = (star_df['x_template']-np.min(star_df['x_template']))/(np.max(star_df['x_template'])-np.min(star_df['x_template']))*2-1
        star_df['normalized_y'] = (star_df['y_template']-np.min(star_df['y_template']))/(np.max(star_df['y_template'])-np.min(star_df['y_template']))*2-1
        star_df.rename(columns = {'flux':'template_flux','fluxerr':'template_flux_err'},inplace = True)
        result_list = []
        for image_id in image_id_list:
            lc_this_image = lightcurve_df[lightcurve_df['image_id'] == image_id]
            
            lc_this_image = pd.merge(lc_this_image,star_df,on = 'source_id',how = 'left')
            lc_this_image = pd.merge(lc_this_image,gaia_stars,left_on = 'gaia_id1',right_on = 'SOURCE_ID',how = 'left')
            mask = (lc_this_image['gaia_dist1']<2)&(lc_this_image['gaia_is_variable1']==False)&(lc_this_image['flux']>0)&(lc_this_image['gaia_dist2']/lc_this_image['gaia_dist1']>1.5) & (lc_this_image['phot_g_mean_mag']<18)& (lc_this_image['phot_bp_mean_mag']==lc_this_image['phot_bp_mean_mag'])& (lc_this_image['phot_rp_mean_mag']==lc_this_image['phot_rp_mean_mag'])
            used_star = lc_this_image[mask]
            print(np.sum(mask),'origin')
            while 1:
                mask,theta = fit_and_filter(used_star)
                if np.sum(mask)==0:
                    #print(theta)
                    break
                else:
                    print(f'{np.sum(mask)} outliers detected, refitting')
                used_star = used_star[~mask]
            print(np.sum(~mask),'left')
            prediction_all = np.squeeze(calculate_prediction(theta,lc_this_image))-2.5*np.log10(np.squeeze(np.array(lc_this_image['flux'])))
            lc_this_image['mag'] = prediction_all
            predice_used = np.squeeze(calculate_prediction(theta,used_star))-2.5*np.log10(np.squeeze(np.array(used_star['flux'])))
            used_star['mag'] = predice_used
            lc_this_image['mag_err'] = lc_this_image['flux_err']/lc_this_image['flux']*2.5/np.log(10)
            upload_result = lc_this_image.dropna(subset=['mag'])
            upload_result = upload_result[['image_id','source_id','time_bjd_tdb','mag','mag_err']]
            result_list.append(upload_result)
            #upload_result.rename(columns = {'mag':'flux','mag_err':'flux_err'},inplace = True)
        result_df = pd.concat(result_list,ignore_index=True)
        return result_df
    def absolute_calibration_batch(self,obs_id,mode = 'aper'):
        inst_info = self.data_loader.get_instrument_info(obs_id = obs_id)
        if mode == 'aper':
            lightcurve_df = self.data_loader.lightcurve_raw_aper_df
        elif mode == 'psf':
            lightcurve_df = self.data_loader.lightcurve_raw_psf_df
        # obs -> image -> lightcurve
        image_metadata = self.data_loader.query_image_metadata(obs_id = obs_id,image_type = "calibrated_science")
        
        stacked_image_metadata = self.data_loader.query_image_metadata(obs_id = obs_id,image_type = "stacked_science")
        stacked_path = stacked_image_metadata.iloc[0,:]['img_name']
        stacked_image_header = fits.getheader(stacked_path)
        # get the lightcurve_df which image_id appeared in image_metadata
        lightcurve_used = lightcurve_df[lightcurve_df['image_id'].isin(image_metadata['image_id'].values)]
        star_df = self.data_loader.source_df[self.data_loader.source_df['sky_id']==int(inst_info['sky_id'])]
        result_light_curve_df = self.absolute_calibration_algo(lightcurve_used,star_df,stacked_image_header,inst_info)
        if mode == 'aper':
            self.data_loader.lightcurve_mag_aper_df = pd.concat([self.data_loader.lightcurve_mag_aper_df,result_light_curve_df],ignore_index=True)
        elif mode == 'psf':
            self.data_loader.lightcurve_mag_psf_df = pd.concat([self.data_loader.lightcurve_mag_psf_df,result_light_curve_df],ignore_index=True)
        self.data_loader.save_lightcurve_info()


if __name__ == "__main__":
    lightcurve_absolute_calibration = absolute_flux_calibration()
    obs_id = 3
    lightcurve_absolute_calibration.absolute_calibration_batch(obs_id,mode = 'aper')
