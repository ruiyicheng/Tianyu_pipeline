import os
import sys
sys.path.append(os.path.abspath("/home/test/workspace/Tianyu_pipeline/algorithm/code"))
import astropy.io.fits as fits
import numpy as np
import utils.dataloader as dataloader
import utils.Bertin as Bertin
from tqdm import tqdm
import sep
from scipy.ndimage import convolve
from astropy.stats import sigma_clip 
import subprocess
import middleware.pipeline_component as pipeline_component
# image stacking include: 
# for calibrated science: SWARP (only provide metainfo for using SWARP in Bertin)
# for calibrated flat: 3median-2mean (read image by myself)
# for bias: mean (read image by myself)

class image_stacking(pipeline_component.pipeline_component):
    def __init__(self,free = True):
        super().__init__(free = free)
        #self.Bertin = Bertin.Bertin_tools()
    def image_stacking_SWARP_algo(self,image_info_df,inst_info = None):
        
        #jd_utc_mid = [image_info_df.iloc[0,:]['jd_utc_mid']]
        mask_star = sigma_clip(np.array(image_info_df['nstar_resolved']),sigma=3).mask
        # Eliminate the images with aeroplane or other issues
        mask_bkg = sigma_clip(np.array(image_info_df['bkg_rms']),sigma_lower = 100,sigma_upper = 5).mask
        mask_result = ~(mask_star | mask_bkg)
        print(np.sum(mask_result))
        selected_image_df = image_info_df[mask_result]
        image_path_list = selected_image_df['img_name'].tolist()
        mask_image_path_list = [i.replace(".fit", "_mask.fits") for i in image_path_list]
        print(mask_image_path_list)
        output_image_file_path, output_weight_file_path = self.Bertin.SWARP_stack(image_path_list,{},mask_image_path_list)
        n_stack = int(np.sum(mask_result))
        jd_utc_mid = selected_image_df.iloc[0,:]['jd_utc_mid']
        return output_image_file_path, output_weight_file_path,jd_utc_mid,n_stack
    def image_stacking_general_algo(self,image_info_df,single_image_array,inst_info = None):
        def interpolate_nan_2d(array):
            #array = np.array(array, dtype=float)  # Ensure float for NaN support

            # Mask of valid (non-NaN) elements
            nan_mask = np.isnan(array)

            # Define a kernel for 8-connected neighbors
            kernel = np.array([[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]], dtype=float)/8

            # Replace NaNs with 0 for the purpose of summing
            array_zeroed = np.where(nan_mask, 0, array)

            # Convolve the array and the mask to count valid neighbors
            neighbor_sum = convolve(array_zeroed, kernel, mode='constant', cval=0.0)
            valid_neighbors = convolve(~nan_mask, kernel, mode='constant', cval=0.0)

            # Avoid division by zero
            with np.errstate(invalid='ignore', divide='ignore'):
                interpolated_values = neighbor_sum / valid_neighbors
                print(interpolated_values)

            # Fill NaNs only where we have valid neighbors
            filled_array = np.where(nan_mask & (valid_neighbors > 0), interpolated_values, array)

            return filled_array
        if inst_info['target_name']=="bias":
            return_image_type = "stacked_bias"
            return_data = np.mean(single_image_array, axis=0)

        elif inst_info['target_name']=="flat":
            return_image_type = "stacked_flat"
            averages_picture = np.mean(single_image_array, axis=(1, 2)).reshape(-1,1,1)
            single_image_array = single_image_array/averages_picture
            weights = averages_picture #1/sigma^2 proportional to flux due to Poisson noise
            weight_list = weights*np.ones_like(single_image_array)
            print('weights = ',weights)
            mean = 0
            for i in range(len(weights)):
                bkg = sep.Background(single_image_array[i])
                bkg_image = bkg.back()
                bkg_rms = bkg.rms()
                image_sub = single_image_array[i] - bkg_image
                mask_criteria = image_sub>3*bkg_rms
                print('masked pixels = ',np.sum(mask_criteria),' in',i)
                weight_list[i][mask_criteria] = 0
                mean += weight_list[i]*single_image_array[i]
            mean = mean/np.sum(weight_list, axis=0)
            return_data = interpolate_nan_2d(mean)
            #print('taking median')
            #median = np.median(single_image_array,axis = 0)
            # Do it line-by-line to save memory
            # for i in tqdm(range(len(median))): 
            #     median[i] = np.median(single_image_array[:,i,:],axis = 0)
            # np.median(res_dict,axis = 0)

            #return_data = 3*median-2*mean
            #return_data = mean
        return_header = fits.getheader(image_info_df.iloc[0,:]['img_name'])
        return_n_stack = len(single_image_array)
        return_image_entry = {
            'jd_utc_mid': image_info_df.iloc[0,:]['jd_utc_mid'],
            'n_stack': return_n_stack,
            'image_type': return_image_type,
        }
        return return_image_entry,return_data,return_header

    def image_stacking_use(self,obs_id):
        # stack the (calibrated) images in obs_id
        instrument_info = self.data_loader.get_instrument_info(obs_id = obs_id)
        if instrument_info['target_name']=="bias" or instrument_info['target_name']=="flat":
            if instrument_info['target_name']=="bias":
                query_type = "bias"
            elif instrument_info['target_name']=="flat":
                query_type = "subtracted_flat"

            image_df = self.data_loader.query_image_metadata(obs_id = obs_id,image_type = query_type)
            print(image_df)
            # read the images
            image_paths = image_df['img_name'].tolist()
            x_size = instrument_info['x_resolution']//instrument_info["bin_size"]
            y_size = instrument_info['y_resolution']//instrument_info["bin_size"]
            image_data = np.empty((len(image_paths),y_size ,x_size ))
            for i,image_path in enumerate(image_paths):
                image_data[i] = fits.getdata(image_path)

            new_image_entry,new_image,new_image_header = self.image_stacking_general_algo(image_df,image_data,inst_info = instrument_info)
            jd_utc_mid = new_image_entry['jd_utc_mid']
            n_stack = new_image_entry['n_stack']
            image_type = new_image_entry['image_type']
            
            
            # save the new image
            new_image_path = self.data_loader.output_image_dir(image_type)
            new_image_path = os.path.join(new_image_path,f"stacked_{instrument_info['target_name']}_{obs_id}.fit")
            print(new_image_path)
            fits.writeto(new_image_path, new_image.astype('float32'), new_image_header, overwrite=True)
            # save the new image metadata
            # append_image_metadata(self,image_id, jd_utc_mid, n_stack, image_type, obs_id, img_name, nstar_resolved, bkg_rms):
            
        else:
            query_type = "calibrated_science"
            image_df = self.data_loader.query_image_metadata(obs_id = obs_id,image_type = query_type)
            image_type = "stacked_science"
            new_image_path = self.data_loader.output_image_dir(image_type)
            new_image_path = os.path.join(new_image_path,f"stacked_{instrument_info['target_name']}_{obs_id}.fit")
            new_mask_image_path = new_image_path.replace(".fit", "_mask.fits")
            output_image_file_path, output_weight_file_path,jd_utc_mid,n_stack = self.image_stacking_SWARP_algo(image_df,inst_info = instrument_info)
            # mv the output image to the new path
            subprocess.run(f'mv {output_image_file_path} {new_image_path}', shell=True)
            subprocess.run(f'mv {output_weight_file_path} {new_mask_image_path}', shell=True)
            # save the new image metadata
            


        new_image_id = self.data_loader.get_another_image_id_list(1)
        self.data_loader.append_image_metadata(new_image_id, [jd_utc_mid], [n_stack], [image_type], [obs_id], [new_image_path], [-1], [-1])



if __name__ == "__main__":
    # Test the image_calibrator class
    image_stacker = image_stacking()
    # image_stacker.image_stacking_use(obs_id = 1)
    #image_stacker.image_stacking_use(obs_id = 2)
    image_stacker.image_stacking_use(obs_id = 3)
    # Test the Bertin tools
    # image_calibrator.Bertin.SWARP_stack(["/media/test/nf/temp/TrES5_rotate/TrES5.fit"],{},target_prefix="/media/test/nf/temp/TrES5_rotate/out/rot_0_TrES5",rot_deg=0)
    # image_calibrator.Bertin.SCAMP_image("/media/test/nf/temp/TrES5_rotate/TrES5.fit",{'DETECT_THRESH':40},ra_deg = 305.2219478671864,dec_deg = 59.44890753010236, arcsec_per_pixel = 0.3025971127584861)



