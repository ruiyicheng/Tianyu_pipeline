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

class lightcurve_extractor(pipeline_component.pipeline_component):
    def __init__(self,free = True):
        super().__init__(free = free)
        # self.data_loader = dataloader.dataloader()
    def select_source(self,star_df,image_header,inst_info):
        x_limit = inst_info['x_resolution']//inst_info['bin_size']-0.5
        y_limit = inst_info['y_resolution']//inst_info['bin_size']-0.5
        
        ra_star = star_df['ra'].values
        dec_star = star_df['dec'].values
        wcs = WCS(image_header)
        # find the nearest star in the new image
        x_star, y_star = wcs.all_world2pix(ra_star, dec_star, 0)
        # find the stars that is within the image
        mask_stars = (x_star>-0.5)&(x_star<x_limit)&(y_star>-0.5)&(y_star<y_limit)
        star_df_used = star_df.iloc[mask_stars]
        x_star = x_star[mask_stars]
        y_star = y_star[mask_stars]
        return star_df_used,x_star,y_star
    def barycentric_correction(self,star_df_used,image_metadata,inst_info):
        obsloc = coord.EarthLocation(lat=inst_info['latitude_deg']*u.deg, lon=inst_info['longitude_deg']*u.deg, height=inst_info['altitude_m']*u.m)
        jd_utc_mid = image_metadata['jd_utc_mid']
        t = Time(jd_utc_mid, format='jd', scale='utc',location=obsloc)
        t_tdb_average = t.tdb
        
        coord_star = coord.SkyCoord(ra = star_df_used['ra'].values, dec = star_df_used['dec'].values,
                        unit=(u.deg, u.deg), frame='icrs')
        light_travel_time = t_tdb_average.light_travel_time(coord_star)
        #print(light_travel_time)
        real_jd_tdb = t_tdb_average + light_travel_time
        return real_jd_tdb.jd
    
    def extract_aper_algo(self,image,image_header,image_metadata,star_df,inst_info = None):
        # input: image, image_header(including WCS)
        # output: source info
        # extract the source using sep, and do photometry on it
        aper_radius = 15
        

        star_df_used,x_star,y_star = self.select_source(star_df,image_header,inst_info)
        
        bkg = sep.Background(image.byteswap().newbyteorder())
        data = image - bkg.back()
        flux,_,flag = sep.sum_circle(data, x_star, y_star, aper_radius, err=bkg.rms())
        eflux = estimate_aperture_photometry_err.estimate_aperture_photometry_err(flux,aper_radius,bkg,x_star,y_star)
        # check if the flux is valid
        
        valid_flux = (flux>0)&(eflux>0)
        flux = flux[valid_flux]
        eflux = eflux[valid_flux]
        star_df_used = star_df_used.iloc[valid_flux]
        star_df_used['flux'] = flux
        star_df_used['flux_err'] = eflux
        # calculate jd_tdb_bjd for the stars using astropy
       
        star_df_used['time_bjd_tdb'] = self.barycentric_correction(star_df_used,image_metadata,inst_info)
        
        
        #print(star_df_used)
        return star_df_used
    def extract_psf_algo(self,image,image_header,image_metadata,star_df,inst_info = None):
        star_df_used,x_star,y_star = self.select_source(star_df,image_header,inst_info)

        sources = Table()
        sources['x_0'] = x_star
        sources['y_0'] = y_star

        bkg = sep.Background(image.byteswap().newbyteorder())
        data = image - bkg.back()
        # do pdf photometry using photutils
        psf_model = IntegratedGaussianPRF()
        psf_model.sigma.fixed = False

        # Step 5: PSF photometry
        photometry = PSFPhotometry(
            psf_model=psf_model,
            fitter=LevMarLSQFitter(),
            fit_shape=(11, 11),
            aperture_radius=15
        )

        # Run PSF fitting
        result = photometry(data, init_params=sources)
        # Step 6: Output flux and uncertainty
        output = result['flux_fit', 'flux_err']
       
        star_df_used['time_bjd_tdb'] = self.barycentric_correction(star_df_used,image_metadata,inst_info)
        star_df_used['flux'] = output['flux_fit']
        star_df_used['flux_err'] = output['flux_err']
        star_df_used = star_df_used[star_df_used['flux_err']==star_df_used['flux_err']]    
        star_df_used = star_df_used[star_df_used['flux']>0]                                               
        return star_df_used

    def extract_batch(self,obs_id,mode = 'aper'):
        inst_info = self.data_loader.get_instrument_info(obs_id = obs_id)
        print(inst_info)
        image_type = 'calibrated_science'
        image_df = self.data_loader.query_image_metadata(obs_id = obs_id,image_type = image_type)
        #print(inst_info['sky_id'])
        source_df = self.data_loader.source_df[self.data_loader.source_df['sky_id']==int(inst_info['sky_id'])]
        #print(source_df)
        for i,r in image_df.iterrows():
            print(f'processing {i}th image')
            if r['nstar_resolved']<0:
                print(f"image {r['img_name']} has no star resolved")
                continue
            image_path = r['img_name']
            image_header = fits.getheader(image_path)
            image = fits.getdata(image_path)
            if mode == 'aper':
                result_df = self.extract_aper_algo(image,image_header,r,source_df,inst_info)
            elif mode == 'psf':
                result_df = self.extract_psf_algo(image,image_header,r,source_df,inst_info)

            # update the lightcurve df in data_loader
            result_df['sky_id'] = int(inst_info['sky_id'])
            result_df['image_id'] = int(r['image_id'])
            append_result = result_df[['source_id','image_id','time_bjd_tdb','flux','flux_err']]
            # Append this result after the last row of the lightcurve_raw_df of data_loader
            if mode == 'aper':
                self.data_loader.lightcurve_raw_aper_df = pd.concat([self.data_loader.lightcurve_raw_aper_df,append_result],ignore_index=True)
            elif mode == 'psf':
                self.data_loader.lightcurve_raw_psf_df = pd.concat([self.data_loader.lightcurve_raw_psf_df,append_result],ignore_index=True)
        self.data_loader.save_lightcurve_info()

    def extract_psf_batch(self,obs_id):
        pass
if __name__ == "__main__":
    lightcurve_extractor_this = lightcurve_extractor()
    obs_id = 3
    lightcurve_extractor_this.extract_batch(obs_id,mode='psf')
