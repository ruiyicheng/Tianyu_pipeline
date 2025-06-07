import os
import sys
sys.path.append(os.path.abspath("/home/test/workspace/Tianyu_pipeline/algorithm/code"))
import numpy as np 

import image_process.source_extraction as source_extraction
import image_process.lightcurve_extraction as lightcurve_extraction
import image_process.image_calibration as image_calibration
import image_process.image_stacking as image_stacking
import calibration.relative_photometry_calibration as relative_photometry_calibration
import calibration.absolute_photometry_calibration as absolute_photometry_calibration
import utils.dataloader as dataloader
import utils.Bertin as Bertin
class process_controller:
    def __init__(self):
        print('initializing data loader')
        self.data_loader = dataloader.dataloader()
        self.Bertin = Bertin.Bertin_tools()
        print('initializing pipeline components')
        self.source_extractor = source_extraction.source_extractor(free = False)
        self.source_extractor.connect(self)

        self.lightcurve_extractor = lightcurve_extraction.lightcurve_extractor(free = False)
        self.lightcurve_extractor.connect(self)

        # self.image_alignment = image_alignment.image_alignment(free = False)
        # self.image_alignment.connect(self)

        self.image_calibration = image_calibration.image_calibrator(free = False)
        self.image_calibration.connect(self)

        self.image_stacker = image_stacking.image_stacking(free = False)
        self.image_stacker.connect(self)

        self.relative_photometry_calibration = relative_photometry_calibration.relative_flux_calibration(free = False)
        self.relative_photometry_calibration.connect(self)

        self.absolute_photometry_calibration = absolute_photometry_calibration.absolute_flux_calibration(free = False)
        self.absolute_photometry_calibration.connect(self)
    def run_pipeline(self,obs_id_science,obs_id_flat,obs_id_bias):
        # obs_id_flat,obs_id_bias must be ready
        # start with image calibration
        # Query dataloader to see whether flat and bias images are ready
        bias_df = self.data_loader.query_image_metadata(obs_id = obs_id_bias,image_type = 'stacked_bias')
        flat_df = self.data_loader.query_image_metadata(obs_id = obs_id_flat,image_type = 'stacked_flat')
        instr_data = self.data_loader.get_instrument_info(obs_id = obs_id_science)
        obs_id_target = instr_data['target_name']
        obs2_same_sky = self.data_loader.observation_metadata[(self.data_loader.observation_metadata['target_name']==obs_id_target)&(self.data_loader.observation_metadata['obs_id']!= obs_id_science)]
        template_list = self.data_loader.image_metadata[self.data_loader.image_metadata['obs_id'].isin(obs2_same_sky['obs_id']) & (self.data_loader.image_metadata['image_type'] == 'stacked_science')]
        new_batch = len(template_list) == 0
        print(f"obs2_same_sky: {obs2_same_sky}")
        print(f"bias_df: {bias_df}")
        print(f"flat_df: {flat_df}")
        print(template_list)
        print(instr_data)


        if len(bias_df) == 0 or len(flat_df) == 0:    
            print("Flat or bias images not ready, running flat and bias calibration pipeline")
            self.batch_calibrateion_flatbias_pipeline(obs_id_flat,obs_id_bias)
        else:
            print("Flat and bias images ready")
            
        # # determine the type of pipeline to run according to whether there is historical events for obs_id_stacked
        # self.data_loader.image_metadata[]
        if new_batch:
            print("No historical events found, running batch pipeline")
            self.batch_pipeline(obs_id_science,obs_id_flat,obs_id_bias)
        else:
            print("Historical events found, running individual pipeline")
            ref_obs_id = int(np.max(template_list['obs_id'].tolist()))
            print('using latest reference observation id:',ref_obs_id)
            self.stream_pipeline(obs_id_science,obs_id_flat,obs_id_bias,ref_obs_id)



    def batch_calibrateion_flatbias_pipeline(self,obs_id_flat,obs_id_bias):
        self.image_stacker.image_stacking_use(obs_id = obs_id_bias)
        self.image_calibration.image_calibration_batch(obs_id_flat,subtract_obs_id = obs_id_bias)
        self.image_stacker.image_stacking_use(obs_id = obs_id_flat)


    def batch_pipeline(self,obs_id_science,obs_id_flat,obs_id_bias):
        # batch pipeline include: calibration(can be done by real-time stream pipeline), image stacking, source extraction, light curve extraction, relative photometry calibration (including reference star selection), absolute photometry calibration
        # 1. check whether all the images are already calibrated
        calibrated_df = self.data_loader.query_image_metadata(obs_id = obs_id_science,image_type = 'calibrated_science')
        raw_df = self.data_loader.query_image_metadata(obs_id = obs_id_science,image_type = 'science')
        if not len(calibrated_df) == len(raw_df):
            self.image_calibration.image_calibration_batch(obs_id_science,subtract_obs_id = obs_id_bias,divide_obs_id = obs_id_flat)
        # 2. stack the calibrated images
        self.image_stacker.image_stacking_use(obs_id = obs_id_science)
        # 3. source extraction
        self.source_extractor.extract_stacked_image(obs_id_science)
        # 4. light curve extraction
        self.lightcurve_extractor.extract_batch(obs_id_science,mode = 'aper')
        self.lightcurve_extractor.extract_batch(obs_id_science,mode = 'psf')
        # 5. relative photometry calibration
        self.relative_photometry_calibration.select_reference_star(obs_id_science,mode = 'aper')
        self.relative_photometry_calibration.select_reference_star(obs_id_science,mode = 'psf')
        self.relative_photometry_calibration.relative_calibration_batch(obs_id_science,mode = 'aper')
        self.relative_photometry_calibration.relative_calibration_batch(obs_id_science,mode = 'psf')
        # 6. absolute photometry calibration
        self.absolute_photometry_calibration.absolute_calibration_batch(obs_id_science,mode = 'aper')
        self.absolute_photometry_calibration.absolute_calibration_batch(obs_id_science,mode = 'psf')


        
    def stream_pipeline(self,obs_id_science,obs_id_flat,obs_id_bias,ref_obs_id):
        # stream pipeline include: calibration, light curve extraction(using old source position), relative photometry calibration (using old reference star), absolute photometry calibration, differential image, transient detection, transient classification (real-bogus? astroid? variable?)
        pass

    def individual_pipeline(self,obs_id_science,obs_id_flat,obs_id_bias):
        pass
    def debug(self):
        self.run_pipeline(3,2,1)
        self.run_pipeline(4,2,1)


if __name__ == "__main__":
    # Test the process_controller class
    process_control = process_controller()
    # process_control.run_pipeline(obs_id_science = 3,obs_id_flat = 2,obs_id_bias = 1)
    process_control.debug()