# This script resolve the sources in the stacked image, and record them.
# The source extraction is done by sep

import os
import sys
sys.path.append(os.path.abspath("/home/test/workspace/Tianyu_pipeline/algorithm/code"))
import numpy as np
import cv2
import pandas as pd
import astropy.io.fits as fits
import numpy.ma as ma
import sep
import utils.dataloader as dataloader
import utils.plotter as plotter
import middleware.pipeline_component as pipeline_component
from astropy.wcs import WCS
import utils.estimate_aperture_photometry_err as estimate_aperture_photometry_err


class source_extractor(pipeline_component.pipeline_component):
    def __init__(self,free = True):
        super().__init__(free = free)
        # self.data_loader = dataloader.dataloader()
        self.plotter = plotter.plotter()
        
    def extract_algo(self,image,image_header,old_source,inst_info = None):
        # input: image, image_header(including WCS)
        # output: source info
        # extract the source using sep, and do photometry on it
        aper_radius = 15
        arcsec_per_pixel = inst_info['x_scale_mum']/inst_info['focal_length_mm']/1000*206265
        dist_threshold = 2/arcsec_per_pixel
        fov_deg = 0.1+(((inst_info['x_scale_mum']/inst_info['focal_length_mm'])*inst_info['x_resolution'])**2+((inst_info['y_scale_mum']/inst_info['focal_length_mm'])*inst_info['y_resolution'])**2)**0.5/1000/2*180/np.pi
        print('fov_deg',fov_deg)
        def find_nearest_kdtree(x1, y1, x2, y2):
            # input: x1, y1, x2, y2
            # output: nearest_indices, distances
            # x1.shape = y1.shape
            # x2.shape = y2.shape
            # output.shape = x1.shape
            # function: find the nearest point in x2, y2 for each point in x1, y1
            # used for matching the sources in template image and target image
            from scipy.spatial import cKDTree
            tree = cKDTree(np.c_[x2, y2])
            distances, nearest_indices = tree.query(np.c_[x1, y1], k=3)
            
            return nearest_indices, distances

        def get_full_df_info(stars,flux,eflux,arcsec_per_pixel,fov,wcs):

            # get the full df info for each star, including ra, dec flux ,eflux and gaiadist for 1,2 stars and return whether they are variable or not
            gaia_stars = self.data_loader.search_GDR3(inst_info['ra'],inst_info['dec'],fov)
            print(gaia_stars)
            gaia_stars['in_vari_classification_result'] = gaia_stars['in_vari_classification_result'].fillna(False)
            x_gaia,y_gaia = wcs.all_world2pix(gaia_stars['ra'], gaia_stars['dec'], 0)
            # find the nearest star in the new image
            nearest_indices, distances = find_nearest_kdtree(stars['x'], stars['y'],x_gaia, y_gaia)
            gid_1 = gaia_stars.loc[np.squeeze(nearest_indices[:,0]),'SOURCE_ID']
            gid_2 = gaia_stars.loc[np.squeeze(nearest_indices[:,1]),'SOURCE_ID']
            gaia_distance1 = np.squeeze(np.array(distances[:,0]*arcsec_per_pixel))
            gaia_distance2 = np.squeeze(np.array(distances[:,1]*arcsec_per_pixel))
            gaia_is_variable1 = gaia_stars.loc[nearest_indices[:,0],'in_vari_classification_result']
            gaia_is_variable2 = gaia_stars.loc[nearest_indices[:,1],'in_vari_classification_result']
            print(gaia_is_variable1)
            star_ra,star_dec = wcs.all_pix2world(stars['x'], stars['y'], 0)
            # return the combined dataframe
            print(len(star_ra),len(star_dec),len(flux),len(eflux),len(gaia_distance1),len(gaia_distance2),len(gid_1),len(gid_2),len(gaia_is_variable1),len(gaia_is_variable2))
            full_df_info = pd.DataFrame({
                'ra': list(star_ra),
                'dec': list(star_dec),
                'flux': list(flux),
                'fluxerr': list(eflux),
                'gaia_dist1': list(gaia_distance1),
                'gaia_dist2': list(gaia_distance2),
                'gaia_id1': list(gid_1),
                'gaia_id2': list(gid_2),
                'gaia_is_variable1': [int(i) for i in list(gaia_is_variable1)],
                'gaia_is_variable2': [int(i) for i in list(gaia_is_variable2)]
            })
            print(full_df_info)
            return full_df_info


        bkg = sep.Background(image.byteswap().newbyteorder())
        data = image - bkg.back()
        
        stars = sep.extract(data, 5, err=bkg.rms(), minarea=5)
        flux,_,flag = sep.sum_circle(data, stars['x'], stars['y'], aper_radius, err=bkg.rms())
        stars = stars[flux>0]
        flux = flux[flux>0]
        eflux = estimate_aperture_photometry_err.estimate_aperture_photometry_err(flux,aper_radius,bkg,stars['x'],stars['y'])
        # now try to get x,y of old stars in new template
        wcs = WCS(image_header)
        # convert the x,y to ra,dec
        x_old,y_old = wcs.all_world2pix(old_source['ra'], old_source['dec'], 0)
        # find the nearest star in the new image
        x_new = stars['x']
        y_new = stars['y']
        index_x = np.arange(len(x_new))
        # find the nearest star in the new image
        nearest_indices, distances = find_nearest_kdtree(x_old, y_old, x_new, y_new)

        matched = distances[:,0]<dist_threshold
        print(f'matched {len(matched[matched])} stars')
        matched_star_index = np.squeeze(nearest_indices[matched][:,0])
        # unmatched_index is index_x without matched_star_index
        unmatched_index = np.setdiff1d(index_x,matched_star_index)
        print(unmatched_index)
        df_matched = get_full_df_info(stars[matched_star_index],flux[matched_star_index],eflux[matched_star_index],arcsec_per_pixel,fov_deg,wcs)
        df_unmatched = get_full_df_info(stars[unmatched_index],flux[unmatched_index],eflux[unmatched_index],arcsec_per_pixel,fov_deg,wcs)
        return df_matched,df_unmatched




        

    def extract_stacked_image(self,obs_id):
        inst_info = self.data_loader.get_instrument_info(obs_id = obs_id)
        image_type = "stacked_science"
        image_info = self.data_loader.query_image_metadata(obs_id = obs_id,image_type = image_type)
        # use the image with highest n_stack if multiple images are obtained
        image_info = image_info.sort_values(by = 'n_stack',ascending = False)
        image_info = image_info.iloc[0,:]
        stack_image_path = image_info['img_name']
        # read the image
        image = fits.getdata(stack_image_path)
        # read the image header
        image_header = fits.getheader(stack_image_path)
        # read the old source
        old_source = self.data_loader.source_df[self.data_loader.source_df['sky_id']==inst_info['sky_id']]
        # extract the source
        old_source_refreshed,new_source = self.extract_algo(image,image_header,old_source,inst_info)
        sky_id = int(inst_info['sky_id'])
        old_source_refreshed['sky_id'] = [sky_id]*len(old_source_refreshed)
        old_source_refreshed['source_id'] = old_source['source_id']
        new_source['sky_id'] = [sky_id]*len(new_source)
        new_source['source_id'] = self.data_loader.get_another_source_id_list(len(new_source))
        # update the old source in source_df using pandas
        self.data_loader.source_df.loc[self.data_loader.source_df['source_id'].isin(old_source_refreshed['source_id']), :] = old_source_refreshed
        # append the new source to source_df
        if len(new_source)>0:
            self.data_loader.append_source(new_source['sky_id'],new_source['ra'],new_source['dec'],new_source['flux'],new_source['fluxerr'],new_source['gaia_id1'],new_source['gaia_dist1'],new_source['gaia_is_variable1'],new_source['gaia_id2'],new_source['gaia_dist2'],new_source['gaia_is_variable2'])
        
if __name__ == "__main__":
    # test the source extractor
    source_extractor = source_extractor()
    obs_id = 3
    source_extractor.extract_stacked_image(obs_id)

    


    