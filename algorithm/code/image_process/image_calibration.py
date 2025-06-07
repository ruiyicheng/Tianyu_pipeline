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
import utils.Bertin as Bertin
import middleware.pipeline_component as pipeline_component
from skimage.morphology import binary_dilation, disk
from ccdproc import cosmicray_lacosmic
# image calibration include: for raw flat: subtract bias
# for raw science: subtract bias, divide flat, mask, resolve WCS
# for both of these type of input:
# The algorithm take the input image, return the calibrated image, header and pandas (database) entry
# The input image are fits image,header, metainfo

class image_calibrator(pipeline_component.pipeline_component):
    def __init__(self,free = True):
        # self.data_loader = dataloader.dataloader()
        super().__init__(free = free)
        #self.Bertin = Bertin.Bertin_tools()
    def get_mask(self,image,Nbit = 16,sat_level = 0.9,gain = 1.3,readnoise = 1,dilation_radius = 5,saturate = False,show = False):
        # Get the masks for saturated pixels, Cosmic ray, hot pixels and airplane/satellite tracks
        # saturate
        mask_saturate = image > (2 ** Nbit * sat_level)
        cr_params = {
            'sigclip': 4.5,
            'objlim': 5.0,
            'gain': gain,      # e-/ADU - CHECK YOUR HEADER!
            'readnoise': readnoise, # e- - CHECK YOUR HEADER!
            'satlevel': 2 ** Nbit * sat_level # ADU - CHECK YOUR HEADER!
        }
        image_this = image.astype('float32')

        mask_hot_pixel = self.mask_cosmic_rays(image_this, **cr_params)

        bkg = sep.Background(image_this,mask = mask_hot_pixel | mask_saturate)
        bkg_image = bkg.back()
        bkg_rms = bkg.rms()
        image_sub = image - bkg_image
        mask_criteria = image_sub>1.5*bkg_rms
        mask_criteria = (mask_criteria*255).astype('uint8')
        kernel = np.ones((3,3),np.uint8)
        mask_criteria = cv2.erode(mask_criteria,kernel)
        kernel = np.ones((3,3),np.uint8)
        mask_criteria = cv2.dilate(mask_criteria,kernel)
        contours,hierarchy = cv2.findContours(mask_criteria,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


        
        airplane_contour = []
        for cnt in contours:
        
            rect = cv2.minAreaRect(cnt)
            long_axis = np.maximum(rect[1][0],rect[1][1])
            short_axis = np.minimum(rect[1][0],rect[1][1])

            # Used the criteria in LAHER 14 
            if long_axis>800 or long_axis>300 and short_axis<20 or long_axis>150 and short_axis<4:
                airplane_contour.append(cnt)

        print(f"find {len(airplane_contour)} airplanes")
        mask_track = np.zeros(mask_criteria.shape,dtype='uint8')
        mask_track = cv2.drawContours(mask_track,airplane_contour, -1, 255,thickness=cv2.FILLED)

        mask_track = mask_track>1
        struct = disk(dilation_radius) # Or square(2*dilation_radius + 1)
        mask_track = binary_dilation(mask_track, footprint=struct)
        mask =  mask_hot_pixel | mask_track
        if saturate:
            
            mask = mask | mask_saturate
        # compute the mask
        
        print(f'masked {np.sum(mask)} pixels')

        return mask,len(airplane_contour)
    def mask_cosmic_rays(self,data, sigclip=4.5, objlim=5.0, gain=1.0, readnoise=5.0, satlevel=60000.0, verbose=False):
        """
        Creates a mask for cosmic rays using the L.A.Cosmic algorithm.

        Args:
            data (np.ndarray): Input image data.
            sigclip (float): Sigma threshold for cosmic ray detection.
            objlim (float): Contrast limit between cosmic ray and underlying object.
            gain (float): Detector gain (e.g., e-/ADU). Get from header if possible.
            readnoise (float): Detector read noise (e.g., e-). Get from header if possible.
            satlevel (float): Saturation level (ADU). Get from header if possible.
            verbose (bool): Print progress information.

        Returns:
            np.ndarray: Boolean mask where True indicates a cosmic ray pixel.
        """
        print("Masking cosmic rays...")
        # L.A.Cosmic implementation in ccdproc
        # It returns the cleaned image and the mask
        # We need gain_apply=False if gain is in e-/ADU and data is in ADU
        cleaned_data, cr_mask = cosmicray_lacosmic(
            data,
            sigclip=sigclip,
            objlim=objlim,
            gain=gain,
            readnoise=readnoise,
            satlevel=satlevel,
            gain_apply=False, # Assume gain/readnoise are in e-, data in ADU
            verbose=verbose
        )
        print(f"Found {np.sum(cr_mask)} cosmic ray pixels.")
        # cr_mask is True where CRs were detected
        return cr_mask
    # Key algorithm
    def image_calibration_algo(self,raw,image_info_df,subtract = None, divide = None,inst_info = None):
        return_mask = None
        bkg_rms = -1
        n_stars = -1
        if subtract is not None: #for both flat and science
            # subtract bias
            return_image_type = "subtracted_flat"
            result_img = raw - subtract
            
        if divide is not None: # For science only. need to mask bad pixels, resolve stars, and WCS (for image without track)
            return_image_type = "calibrated_science"
            # divide flat. 
            result_img = result_img / divide
            # mask the image
            return_mask,airplane_contour = self.get_mask(result_img,Nbit = inst_info['n_bit'],gain = inst_info['gain'],readnoise = inst_info['readout_noise_e'])
            return_mask = ~return_mask
            bkg = sep.Background(result_img.astype("float32"))
            bkg_rms = bkg.globalrms
            calibrated_image_debkg = result_img.copy()-bkg.back()

            if airplane_contour == 0: # not contaminated image
                star_resolve = sep.extract(calibrated_image_debkg.astype("float32"),5,err=bkg.globalrms,minarea=5)
                n_stars = len(star_resolve)
                print(inst_info)
                ra = inst_info['ra']
                dec = inst_info['dec']
                arcsec_per_pixel = inst_info['x_scale_mum']/inst_info['focal_length_mm']/1000*206265
                # WCS resolve
                header_scamp = self.Bertin.SCAMP_image(image_info_df['img_name'],{"DETECT_MINAREA":5,"DETECT_THRESH":5},ra_deg = ra,dec_deg = dec, arcsec_per_pixel = arcsec_per_pixel)
                print('resolved stars:',n_stars)
                print('bkgrms:',bkg.globalrms)
                print('WCS:',header_scamp)



        return_header = fits.getheader(image_info_df['img_name'])
        return_image_entry = {
            'jd_utc_mid': image_info_df['jd_utc_mid'],
            'image_type': return_image_type,
            'n_stars_resolved': n_stars,
            'bkg_rms': bkg_rms
        }
        return return_image_entry,result_img,return_header,return_mask

            

        
    # IO part, would be substituted in the future
    def image_calibration_batch(self,raw_obs_id,subtract_obs_id = None,divide_obs_id = None,subset = None):
        instrument_info = self.data_loader.get_instrument_info(obs_id = raw_obs_id)
        stacked_bias_df = self.data_loader.query_image_metadata(obs_id = subtract_obs_id,image_type = "stacked_bias")
        print(stacked_bias_df.iloc[0,:]['img_name'])
        stacked_bias = fits.getdata(stacked_bias_df.iloc[0,:]['img_name'])
        jd_utc_mid = []
        n_stack = []
        image_type = []
        obs_id = []
        new_image_path_list = []
        n_stars_resolved = []
        bkg_rms = []
        stacked_flat = None
        if divide_obs_id is None:
            # Subtracting the bias from flat
            if subtract_obs_id is None:
                # error
                raise ValueError("Either subtract_obs_id must be provided")
            image_info_df = self.data_loader.query_image_metadata(obs_id = raw_obs_id,image_type = "flat")
            output_image_type = "subtracted_flat"
        else:
            # bias and flat for science
            stacked_flat_df = self.data_loader.query_image_metadata(obs_id = divide_obs_id,image_type = "stacked_flat")
            stacked_flat = fits.getdata(stacked_flat_df.iloc[0,:]['img_name'])
            image_info_df = self.data_loader.query_image_metadata(obs_id = raw_obs_id,image_type = "science")
            if subset is not None:
                image_info_df = image_info_df.iloc[subset,:]
            output_image_type = "calibrated_science"
        for i,r in image_info_df.iterrows():
            print('processing image:',r['img_name'])
            raw = fits.getdata(r['img_name'])
            return_image_entry,result_img,return_header,return_mask = self.image_calibration_algo(raw,r,subtract = stacked_bias,divide = stacked_flat,inst_info = instrument_info)

            
            # save the new image
            new_image_path = self.data_loader.output_image_dir(output_image_type)
            file_name = os.path.basename(r['img_name'])
            calibrated_file_name = file_name.replace(".fit",f"_{raw_obs_id}_calibrated.fit")
            new_image_path = os.path.join(new_image_path,calibrated_file_name)
            jd_utc_mid.append(return_image_entry['jd_utc_mid'])
            n_stack.append(1)
            image_type.append(return_image_entry['image_type'])
            n_stars_resolved.append(return_image_entry['n_stars_resolved'])
            bkg_rms.append(return_image_entry['bkg_rms'])
            obs_id.append(raw_obs_id)
            new_image_path_list.append(new_image_path)
            
            fits.writeto(new_image_path, result_img.astype('float32'), return_header, overwrite=True)
            if return_mask is not None:
                # save the mask
                mask_path = new_image_path.replace(".fit", "_mask.fits")
                fits.writeto(mask_path, return_mask.astype('uint8'), return_header, overwrite=True)
        new_image_id = self.data_loader.get_another_image_id_list(len(jd_utc_mid))
        self.data_loader.append_image_metadata(new_image_id, jd_utc_mid, n_stack, image_type, obs_id, new_image_path_list, n_stars_resolved, bkg_rms)



if __name__ == "__main__":
    # Test the image_calibrator class
    image_calibrator_this = image_calibrator()
    #image_calibrator_this.image_calibration_batch(2,subtract_obs_id = 1)
    image_calibrator_this.image_calibration_batch(3,subtract_obs_id = 1,divide_obs_id = 2)
    # image_calibrator.Bertin.SWARP_stack(["/media/test/nf/temp/TrES5_rotate/TrES5.fit"],{},target_prefix="/media/test/nf/temp/TrES5_rotate/out/rot_0_TrES5",rot_deg=0)
    # image_calibrator.Bertin.SCAMP_image("/media/test/nf/temp/TrES5_rotate/TrES5.fit",{'DETECT_THRESH':40},ra_deg = 305.2219478671864,dec_deg = 59.44890753010236, arcsec_per_pixel = 0.3025971127584861)



 