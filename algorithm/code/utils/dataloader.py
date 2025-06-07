
import os
import sys
sys.path.append(os.path.abspath("/home/test/workspace/Tianyu_pipeline/algorithm/code"))
import pandas as pd
import numpy as np
from astroquery.gaia import Gaia

class dataloader:
    def __init__(self,homedir = "/home/test/workspace/Tianyu_pipeline/algorithm/data"):
        # Obtain the metadata required for the pipeline
        # In real applications, the metadata is obtained from the database through query
        # In this demo, only one observation is considered, and the metadata is avilable in csv files
        # Therefore, the provided input for the query is not used, but will be considered in the future
        self.homedir = homedir
        self.input_path = os.path.join(self.homedir,"testinput")
        self.output_path = os.path.join(self.homedir,"testoutput")
        self.output_cache_path = os.path.join(self.output_path,"temp")
        self.output_plot_path = os.path.join(self.output_path,"plot")
        self.output_lightcurve_raw_aper_path = os.path.join(self.output_path,"catalog/lightcurve/lightcurve_raw_aper.csv")
        self.lightcurve_raw_aper_df = pd.read_csv(self.output_lightcurve_raw_aper_path)
        self.output_lightcurve_raw_psf_path = os.path.join(self.output_path,"catalog/lightcurve/lightcurve_raw_psf.csv")
        self.lightcurve_raw_psf_df = pd.read_csv(self.output_lightcurve_raw_psf_path)

        self.output_lightcurve_relative_aper_path = os.path.join(self.output_path,"catalog/lightcurve/lightcurve_relative_aper.csv")
        self.lightcurve_relative_aper_df = pd.read_csv(self.output_lightcurve_relative_aper_path)
        self.output_lightcurve_relative_psf_path = os.path.join(self.output_path,"catalog/lightcurve/lightcurve_relative_psf.csv")
        self.lightcurve_relative_psf_df = pd.read_csv(self.output_lightcurve_raw_psf_path)

        self.output_lightcurve_mag_aper_path = os.path.join(self.output_path,"catalog/lightcurve/lightcurve_mag_aper.csv")
        self.lightcurve_mag_aper_df = pd.read_csv(self.output_lightcurve_mag_aper_path)
        self.output_lightcurve_mag_psf_path = os.path.join(self.output_path,"catalog/lightcurve/lightcurve_mag_psf.csv")
        self.lightcurve_mag_psf_df = pd.read_csv(self.output_lightcurve_mag_psf_path)

        self.output_source_path = os.path.join(self.output_path,"catalog/star/source.csv")
        self.source_df = pd.read_csv(self.output_source_path)
        self.output_reference_star_psf_path = os.path.join(self.output_path,"catalog/star/reference_star_psf.csv")
        self.reference_star_psf_df = pd.read_csv(self.output_reference_star_psf_path)
        self.output_reference_star_aper_path = os.path.join(self.output_path,"catalog/star/reference_star_aper.csv")
        self.reference_star_aper_df = pd.read_csv(self.output_reference_star_aper_path)
        self.output_image_path = os.path.join(self.output_path,"image")
        self.metadata_path = os.path.join(self.input_path,"metadata")
        self.image_metadata_path = os.path.join(self.metadata_path,"image/demo_image.csv")
        self.instrument_metadata_path = os.path.join(self.metadata_path,"instrument/demo_instrument.csv")
        self.observation_metadata_path = os.path.join(self.metadata_path,"observation/demo_observation.csv")
        self.sky_matadata_path = os.path.join(self.metadata_path,"sky/demo_sky.csv")
        # read csv files
        self.image_metadata = pd.read_csv(self.image_metadata_path)
        self.observation_metadata = pd.read_csv(self.observation_metadata_path)
        self.instrument_metadata = pd.read_csv(self.instrument_metadata_path)
        self.sky_metadata = pd.read_csv(self.sky_matadata_path)

        # observation_metadata left Join sky_metadata on target_name; left join instrument_metadata on instrument_id
        self.merged_obs_metadata = pd.merge(self.observation_metadata, self.sky_metadata, on='target_name', how='left')
        self.merged_obs_metadata = pd.merge(self.merged_obs_metadata, self.instrument_metadata, on='instrument_id', how='left')
        #print(self.merged_obs_metadata)
        
    def output_image_dir(self,image_type):
        return os.path.join(self.output_image_path,image_type)
    def get_another_source_id_list(self,n):
        id_max = np.max(self.source_df['source_id'])
        if np.isnan(id_max): 
            id_max=0 
        return [int(i) for i in list(range(id_max+1,id_max+n+1))]
    
    def get_another_image_id_list(self,n):
        id_max = np.max(self.image_metadata['image_id'])
        return list(range(id_max+1,id_max+n+1))
    def save_image_info(self):
        self.image_metadata.to_csv(self.image_metadata_path, index=False)
    def save_source_info(self):
        self.source_df.to_csv(self.output_source_path, index=False)
    def save_lightcurve_info(self):
        print('saving lc...')
        self.lightcurve_raw_aper_df.to_csv(self.output_lightcurve_raw_aper_path, index=False)
        self.lightcurve_raw_psf_df.to_csv(self.output_lightcurve_raw_psf_path, index=False)
        self.lightcurve_relative_aper_df.to_csv(self.output_lightcurve_relative_aper_path, index=False)
        self.lightcurve_relative_psf_df.to_csv(self.output_lightcurve_relative_psf_path, index=False)
        self.lightcurve_mag_aper_df.to_csv(self.output_lightcurve_mag_aper_path, index=False)
        self.lightcurve_mag_psf_df.to_csv(self.output_lightcurve_mag_psf_path, index=False)
        print('finished')
    def append_reference_star(self,star_df,mode = 'aper'):
        if mode == 'aper':
            self.reference_star_aper_df = pd.concat([self.reference_star_aper_df, star_df], ignore_index=True)
            self.reference_star_aper_df.to_csv(self.output_reference_star_aper_path, index=False)
        elif mode == 'psf':
            self.reference_star_psf_df = pd.concat([self.reference_star_psf_df, star_df], ignore_index=True)
            self.reference_star_psf_df.to_csv(self.output_reference_star_psf_path, index=False)
        else:
            raise ValueError("mode must be either aper or psf")
    def append_image_metadata(self,image_id, jd_utc_mid, n_stack, image_type, obs_id, img_name, nstar_resolved, bkg_rms):
        # The image metadata is obtained from the csv file
        # The csv file is provided, but in real applications, the information is obtained from the database
        new_image_metadata = pd.DataFrame({
            'image_id': image_id,
            'jd_utc_mid': jd_utc_mid,
            'n_stack': n_stack,
            'image_type': image_type,
            'obs_id': obs_id,
            'img_name': img_name,
            'nstar_resolved': nstar_resolved,
            'bkg_rms': bkg_rms
        })
        # Append the new image metadata to the existing metadata
        self.image_metadata = pd.concat([self.image_metadata, new_image_metadata], ignore_index=True)
        # Save the updated metadata to the csv file
        self.image_metadata.to_csv(self.image_metadata_path, index=False)
    def append_source(self,sky_id,ra,dec,flux,fluxerr,gaia_id1,gaia_dist1,gaia_is_variable1,gaia_id2,gaia_dist2,gaia_is_variable2):
        source_id = self.get_another_source_id_list(len(sky_id))
        new_source_df = pd.DataFrame({
            'source_id': source_id,
            'sky_id': sky_id,
            'ra': ra,
            'dec': dec,
            'flux': flux,
            'fluxerr': fluxerr,
            'gaia_id1': gaia_id1,
            'gaia_dist1': gaia_dist1,
            'gaia_is_variable1': gaia_is_variable1,
            'gaia_id2': gaia_id2,
            'gaia_dist2': gaia_dist2,
            'gaia_is_variable2': gaia_is_variable2
        })
        # Append the new source metadata to the existing metadata
        self.source_df = pd.concat([self.source_df, new_source_df], ignore_index=True)
        # Save the updated metadata to the csv file
        self.source_df.to_csv(self.output_source_path, index=False)
    def get_instrument_info(self, image_id = None,obs_id = None):
        # The instrument information is obtained from the csv file
        # The csv file is provided, but in real applications, the information is obtained from the database
        # if image id is provided, get obs_id
        # if obs id is provided, return the instrument information for the observation
        if obs_id is None and image_id is None:
            raise ValueError("Either image_id or obs_id must be provided")
        if image_id is not None:
            obs_id = self.image_metadata.loc[self.image_metadata['image_id'] == image_id, 'obs_id'].values[0]
        
        ret = self.merged_obs_metadata[self.merged_obs_metadata['obs_id'] == obs_id]
        # convert ret into a dictionary
        ret = ret.to_dict(orient='records')[0]
        return ret
    def query_image_metadata(self,image_id_list = None,obs_id = None,image_type = None,limit = 10000):
        # The image metadata is obtained from the csv file
        # The csv file is provided, but in real applications, the information is obtained from the database
        # if image id is provided, get obs_id
        # if obs id is provided, return the instrument information for the observation
        if (obs_id is None or image_type is None) and image_id_list is None:
            raise ValueError("Either obs_id+image_type or image_id_list must be provided")
        if image_id_list is not None:
            target_df = pd.DataFrame({'image_id':image_id_list})
            ret = pd.merge(target_df,self.image_metadata,on='image_id',how='left')
        
        if obs_id is not None and image_type is not None:
            ret = self.image_metadata[(self.image_metadata['obs_id'] == obs_id) & (self.image_metadata['image_type'] == image_type)]
        ret = ret.iloc[:limit,:]
        return ret
    def search_GDR3(self,ra,dec,fov,mag_limit = 21):
        # input:  stars, flux, eflux
        # output: full_df_info
        # function: get the full df info for each star, including ra, dec flux ,eflux and gaiadist
        cache_file_path = os.path.join(self.output_cache_path,f'{ra}_{dec}_{fov}_{mag_limit}.csv')
        if os.path.exists(cache_file_path):
            full_df_info = pd.read_csv(cache_file_path)
            return full_df_info
        else:
            sql = f'''
    SELECT g3.source_id,g3.ra,g3.dec,g3.phot_g_mean_mag,g3.phot_g_mean_flux_over_error,g3.parallax,g3.pmra,g3.pmdec,gv.in_vari_classification_result,g3.phot_bp_mean_mag,g3.phot_bp_mean_flux_over_error,g3.phot_rp_mean_mag,g3.phot_rp_mean_flux_over_error from gaiadr3.gaia_source as g3 LEFT JOIN gaiadr3.vari_summary as gv on gv.source_id=g3.source_id 
WHERE g3.phot_g_mean_mag<{mag_limit} AND
CONTAINS(
POINT('ICRS',g3.ra,g3.dec),
CIRCLE('ICRS',{ra},{dec},{fov})
)=1'''      
            job = Gaia.launch_job_async(sql)
            r = job.get_results()
            r.to_pandas().to_csv(cache_file_path,index = False)
            return r.to_pandas()
    def query_light_curve(self,gdr3_id,photometry_method = 'aper',lightcurve_type = 'relative'):
        source = self.source_df[self.source_df['gaia_id1'] == gdr3_id]
        # retrieve the source with smallest gaia_dist1
        source = source.loc[source['gaia_dist1'].idxmin()]
        source_id = int(source.iloc[0]['source_id'])
        if photometry_method == 'aper':
            if lightcurve_type == 'relative':
                lightcurve_df = self.lightcurve_relative_aper_df
            elif lightcurve_type == 'absolute':
                lightcurve_df = self.lightcurve_mag_aper_df
            elif lightcurve_type == 'raw':
                lightcurve_df = self.lightcurve_raw_aper_df
            else:
                raise ValueError("lightcurve_type must be either relative or absolute")
        elif photometry_method == 'psf':
            if lightcurve_type == 'relative':
                lightcurve_df = self.lightcurve_relative_psf_df
            elif lightcurve_type == 'absolute':
                lightcurve_df = self.lightcurve_mag_psf_df
            elif lightcurve_type == 'raw':
                lightcurve_df = self.lightcurve_raw_psf_df
            else:
                raise ValueError("lightcurve_type must be either relative or absolute")
        else:
            raise ValueError("photometry_method must be either aper or psf")
        lightcurve_df = lightcurve_df[lightcurve_df['source_id'] == source_id]
        return lightcurve_df
    def rollback(self):
        # first step: delete all light_curve
        self.lightcurve_raw_aper_df = pd.DataFrame(columns=self.lightcurve_raw_aper_df.columns)
        self.lightcurve_raw_psf_df = pd.DataFrame(columns=self.lightcurve_raw_psf_df.columns)
        self.lightcurve_relative_aper_df = pd.DataFrame(columns=self.lightcurve_relative_aper_df.columns)
        self.lightcurve_relative_psf_df = pd.DataFrame(columns=self.lightcurve_relative_psf_df.columns)
        self.lightcurve_mag_aper_df = pd.DataFrame(columns=self.lightcurve_mag_aper_df.columns)
        self.lightcurve_mag_psf_df = pd.DataFrame(columns=self.lightcurve_mag_psf_df.columns)
        self.save_lightcurve_info()
        # second step: delete all source
        self.source_df = pd.DataFrame(columns=self.source_df.columns)
        self.save_source_info()
        # third step: delete all image metadata
        # self.image_metadata = pd.DataFrame(columns=self.image_metadata.columns)


if __name__ == "__main__":
    dl = dataloader()
    dl.rollback()
    # print(dl.get_instrument_info(obs_id = 3))
    # print(dl.query_image_metadata(obs_id = 3,image_type = 'science'))
    # print(dl.query_image_metadata(image_id_list = [1,2,3]))
