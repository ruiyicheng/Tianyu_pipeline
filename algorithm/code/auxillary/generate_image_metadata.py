# This script would register the metainfo of image into the result csv file
# In real applications, the metainfo is generated once observation is finished

import pandas as pd
import glob
import astropy.io.fits as fits
input_image_path = "/home/test/workspace/Tianyu_pipeline/algorithm/data/testinput/image"
output_image_catalog_path = "/home/test/workspace/Tianyu_pipeline/algorithm/data/testinput/metadata/image/demo_image.csv"

image_id_this = 1
image_id = []
jd_utc_mid = []
n_stack = []
image_type = []
obs_id = []
img_name = []
nstar_resolved = []
bkg_rms = []


def register_image(image_path_list,image_obs_id,image_type_this,image_id,jd_utc_mid,n_stack,image_type,obs_id,img_name,nstar_resolved,bkg_rms):
    global image_id_this
    for image_path in image_path_list:
        image_id.append(image_id_this)
        header = fits.getheader(image_path)
        jd_utc_mid.append(header['JD']+header['EXPOSURE']/3600/24/2)
        n_stack.append(1)
        image_type.append(image_type_this)
        obs_id.append(image_obs_id)
        img_name.append(image_path)
        nstar_resolved.append(-1)
        bkg_rms.append(-1)
        image_id_this += 1




bias_image_list = glob.glob(input_image_path + "/bias/*.fit")
#sort bias_image_list by name
bias_image_list.sort()
bias_obs_id = 1
type_name = "bias"
register_image(bias_image_list,bias_obs_id,type_name,image_id,jd_utc_mid,n_stack,image_type,obs_id,img_name,nstar_resolved,bkg_rms)
flat_image_list = glob.glob(input_image_path + "/flat/*.fit")
#sort flat_image_list by name
flat_image_list.sort()
flat_image_list = flat_image_list[:7]  # Limit to 7 images for testing
flat_obs_id = 2
type_name = "flat"
register_image(flat_image_list,flat_obs_id,type_name,image_id,jd_utc_mid,n_stack,image_type,obs_id,img_name,nstar_resolved,bkg_rms)
target_image_list = glob.glob(input_image_path + "/raw/KOI68/*.fit")
#sort target_image_list by name
target_image_list.sort()
target_image_list_before = target_image_list[:150]
target_obs_id = 3
type_name = "science"
register_image(target_image_list_before,target_obs_id,type_name,image_id,jd_utc_mid,n_stack,image_type,obs_id,img_name,nstar_resolved,bkg_rms)


target_image_list_after = target_image_list[150:]
target_obs_id = 4
type_name = "science"
register_image(target_image_list_after,target_obs_id,type_name,image_id,jd_utc_mid,n_stack,image_type,obs_id,img_name,nstar_resolved,bkg_rms)
# Register the image metadata into the csv file
image_metadata = pd.DataFrame({
    'image_id': image_id,
    'jd_utc_mid': jd_utc_mid,
    'n_stack': n_stack,
    'image_type': image_type,
    'obs_id': obs_id,
    'img_name': img_name,
    'nstar_resolved': nstar_resolved,
    'bkg_rms': bkg_rms
})
image_metadata.to_csv(output_image_catalog_path, index=False)
