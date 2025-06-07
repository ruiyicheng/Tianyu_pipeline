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
import middleware.pipeline_component as pipeline_component
from astropy.table import Table
from astropy.modeling.fitting import LevMarLSQFitter
from photutils.psf import PSFPhotometry, IntegratedGaussianPRF

class relative_flux_calibration(pipeline_component.pipeline_component):
    def __init__(self,free = True):
        #self.data_loader = dataloader.dataloader()
        super().__init__(free=free)
        

    def find_reference_star(self,lightcurve_df,star_df,stacked_image_header,inst_info,n_q):
        def assign_quantiles(reference_df, target_df, column, n_quantiles):
            # Step 1: Get bin edges from reference data
            _, bin_edges = pd.qcut(reference_df[column], q=n_quantiles, retbins=True, duplicates='drop')
            # Step 2: Extend bin edges slightly to include out-of-range target values
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            # Step 3: Assign quantiles to target data
            target_quantiles = pd.cut(target_df[column], bins=bin_edges, labels=False, include_lowest=True)
            return target_quantiles
        
        wcs = WCS(stacked_image_header)
        arcsec_per_pixel = inst_info['x_scale_mum']/inst_info['focal_length_mm']/1000*206265
        source_number = lightcurve_df.groupby('source_id').size()
        max_size = source_number.max()
        max_groups = source_number[source_number == max_size].index.tolist()
        result_df = lightcurve_df[lightcurve_df['source_id'].isin(max_groups)]
        star_df_used = star_df[star_df['source_id'].isin(max_groups)]
        x, y = wcs.all_world2pix(star_df_used['ra'].values, star_df_used['dec'].values, 0)
        #print(x,y)
        star_df_used.loc[:,'x'] = x
        star_df_used.loc[:,'y'] = y
        maximum_tracking_and_pointing_error = 200/arcsec_per_pixel
        mask = (star_df_used['x']>maximum_tracking_and_pointing_error)&(star_df_used['x']<inst_info['x_resolution']//inst_info['bin_size']-maximum_tracking_and_pointing_error)&(star_df_used['y']>maximum_tracking_and_pointing_error)&(star_df_used['y']<inst_info['y_resolution']//inst_info['bin_size']-maximum_tracking_and_pointing_error)
        mask = mask & (star_df_used['gaia_dist1']<0.5)&(star_df_used['gaia_dist2']>10) &(star_df_used['gaia_is_variable1']<0.5)
        star_df_used = star_df_used[mask]
        target_quantiles = assign_quantiles(star_df_used,star_df,'flux',n_q)
        star_df['group_quantile'] = target_quantiles
        star_df['is_reference'] = star_df['source_id'].isin(star_df_used['source_id'])
        return star_df
    def relative_calibration_algo(self,lightcurve_df,star_df):
        # n_q = 10 # number of quantile
        # star_df = self.find_reference_star(lightcurve_df,star_df,stacked_image_header,inst_info,n_q)
        quantile_part = star_df.groupby('group_quantile').size().index.tolist()
        image_id_list = lightcurve_df.groupby('image_id').size().index.tolist()
        new_flux_list = []
        for i in quantile_part:
            star_df_used = star_df[star_df['group_quantile'] == i][['source_id','is_reference']]
            indexed_star_df = star_df_used.set_index(['source_id'], inplace=False)
            #print(indexed_star_df)
            for image_id in image_id_list:
                print('processing image_id:',image_id,'quantile:',i)
                flux_df_this_image = lightcurve_df[(lightcurve_df['image_id'] == image_id)&(lightcurve_df['source_id'].isin(star_df_used['source_id']))]
                #flux_df_this_image.set_index(['source_id'], inplace=True)
                flux_df_this_image = pd.merge(flux_df_this_image, indexed_star_df, how='left', left_on='source_id', right_index=True)
                #print(flux_df_this_image)
                reference_star = flux_df_this_image[flux_df_this_image['is_reference'] == True]
                flux_df_this_image['sum_reference'] = np.sum(flux_df_this_image['flux'])-flux_df_this_image['is_reference']*flux_df_this_image['flux']
                flux_df_this_image['eff_reference'] = len(reference_star)-flux_df_this_image['is_reference']
                
                flux_df_this_image['relative_flux'] = flux_df_this_image['flux']*flux_df_this_image['eff_reference']/flux_df_this_image['sum_reference']
                flux_df_this_image['relative_flux_err'] = flux_df_this_image['flux_err']*flux_df_this_image['eff_reference']/flux_df_this_image['sum_reference']
                flux_df_this_image = flux_df_this_image.reset_index()
                new_flux_list.append(flux_df_this_image[['source_id','image_id','time_bjd_tdb','relative_flux','relative_flux_err']].rename(columns={'relative_flux': 'flux','relative_flux_err': 'flux_err'}))
    
        # select reference star and save in the 
        result = pd.concat(new_flux_list)
        return result
    def relative_calibration_batch(self,obs_id,previous_obs_id = None,mode = 'aper'):
        if previous_obs_id is None:
            previous_obs_id = obs_id # if no previous obs_id, use the current obs_id

        inst_info = self.data_loader.get_instrument_info(obs_id = obs_id)
        if mode == 'aper':
            lightcurve_df = self.data_loader.lightcurve_raw_aper_df
            reference_star_df = self.data_loader.reference_star_aper_df[self.data_loader.reference_star_aper_df['obs_id']== previous_obs_id]
        elif mode == 'psf':
            lightcurve_df = self.data_loader.lightcurve_raw_psf_df
            reference_star_df = self.data_loader.reference_star_psf_df[self.data_loader.reference_star_psf_df['obs_id']== previous_obs_id]
        # obs -> image -> lightcurve


        image_metadata = self.data_loader.query_image_metadata(obs_id = obs_id,image_type = "calibrated_science")
        
        # stacked_image_metadata = self.data_loader.query_image_metadata(obs_id = obs_id,image_type = "stacked_science")
        # stacked_path = stacked_image_metadata.iloc[0,:]['img_name']
        # stacked_image_header = fits.getheader(stacked_path)
        # get the lightcurve_df which image_id appeared in image_metadata
        lightcurve_used = lightcurve_df[lightcurve_df['image_id'].isin(image_metadata['image_id'].values)]
        star_df = self.data_loader.source_df[self.data_loader.source_df['sky_id']==int(inst_info['sky_id'])]
        star_df = pd.merge(star_df,reference_star_df[['source_id','is_reference','group_quantile']],on='source_id',how='left')

        result_light_curve_df = self.relative_calibration_algo(lightcurve_used,star_df)
        if mode == 'aper':
            self.data_loader.lightcurve_relative_aper_df = pd.concat([self.data_loader.lightcurve_relative_aper_df,result_light_curve_df],ignore_index=True)
        elif mode == 'psf':
            self.data_loader.lightcurve_relative_psf_df = pd.concat([self.data_loader.lightcurve_relative_psf_df,result_light_curve_df],ignore_index=True)
        self.data_loader.save_lightcurve_info()
    def select_reference_star(self,obs_id,n_q = 10,mode = 'aper'):
        # select reference star for the obs_id need raw flux and stacked image 

        inst_info = self.data_loader.get_instrument_info(obs_id = obs_id)
        if mode == 'aper':
            lightcurve_df = self.data_loader.lightcurve_raw_aper_df
        elif mode == 'psf':
            lightcurve_df = self.data_loader.lightcurve_raw_psf_df
        image_metadata = self.data_loader.query_image_metadata(obs_id = obs_id,image_type = "calibrated_science")
        
        stacked_image_metadata = self.data_loader.query_image_metadata(obs_id = obs_id,image_type = "stacked_science")
        stacked_path = stacked_image_metadata.iloc[0,:]['img_name']
        stacked_image_header = fits.getheader(stacked_path)
        # get the lightcurve_df which image_id appeared in image_metadata
        lightcurve_used = lightcurve_df[lightcurve_df['image_id'].isin(image_metadata['image_id'].values)]

        star_df = self.data_loader.source_df[self.data_loader.source_df['sky_id']==int(inst_info['sky_id'])]
        star_df = self.find_reference_star(lightcurve_used,star_df,stacked_image_header,inst_info,n_q)
        star_df['obs_id'] = int(obs_id)
        star_df = star_df[["obs_id","source_id","is_reference","group_quantile"]]
        self.data_loader.append_reference_star(star_df,mode = mode)



if __name__ == "__main__":
    lightcurve_relative_calibration = relative_flux_calibration()
    obs_id = 3
    lightcurve_relative_calibration.relative_calibration_batch(obs_id,mode = 'aper')
