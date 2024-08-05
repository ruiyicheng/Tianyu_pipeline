import Tianyu_pipeline.pipeline.dev.file_system.file_system as file_sys

class file_transferer:
    def __init__(self):
        self.fs = file_sys.file_system()
    def transfer_file_to_file_system(self,input_path,observation_info):
        '''
        This function is used for debug. Observation is created manually. Input the Observation, and raw data file path so it will transfer it to the right file system. 
        The pictures will be registered.
        In real runtime. This step would be executed automatically because of real-time demand and large number of sky region.
        '''
        self.fs.init_file_system()
    def transfer_obs_site_to_site(self,obs_id,site_1,site_2):
        pass