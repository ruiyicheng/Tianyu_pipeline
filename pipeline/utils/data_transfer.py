import glob
import Tianyu_pipeline.pipeline.dev.file_system.file_system as file_sys
import Tianyu_pipeline.pipeline.utils.data_loader as data_loader 
import Tianyu_pipeline.pipeline.dev.process_management.process_publisher as process_pub 
import Tianyu_pipeline.pipeline.utils.sql_interface as sql_interface
import time
import os
class file_transferer:
    def __init__(self):
        self.sql_interface = sql_interface.sql_interface()
        self.fs = file_sys.file_system()
        self.dl = data_loader.data_loader()
        self.site_id = self.fs.site_info['process_site_id']
        self.pp_this_site = process_pub.process_publisher(site_id = self.site_id, group_id = self.site_id)
    def transfer_file_to_file_system(self,input_path,observation_id,image_info_dict={'image_type':"raw"}):
        '''
        This function is used for debug. Observation is created manually. Input the Observation, and raw data file path so it will transfer it to the right file system. 
        The pictures will be registered.
        In real runtime. This step would be executed automatically because of real-time demand and large number of sky region.
        '''
        file_paths = sorted(glob.glob(input_path+'/*'))
        self.fs.init_file_system()


        sql = '''SELECT obs.batch_size as batch_size FROM observation as obs WHERE obs.obs_id=%s;'''
        args = (observation_id,)
        result = self.sql_interface.query(sql,args)
        assert len(result)==1
        result = result.to_dict('records')[0]
        batch_size = result['batch_size']
        #3 steps: register; create dir; mv
        PID_list = []
        image_type_id = self.sql_interface.image_type_id[image_info_dict['image_type']]
        for i,fp in enumerate(file_paths):
            batch_number = i//batch_size+1   
            PID = self.pp_this_site.register_info({"cmd":"INSERT INTO img (store_site_id,batch,image_type_id,obs_id,img_name) VALUES (%s,%s,%s,%s,%s);","args":[self.site_id,batch_number,image_type_id,observation_id,fp.split('/')[-1]]})
            while True:
                print(f"waiting for img {i} registration")
                time.sleep(0.5)
                sql = '''SELECT * FROM img WHERE img.birth_process_id=%s;'''
                args = (PID,)
                result,_ = self.sql_interface.query(sql,args,return_df = False)
                if len(result==1):
                    break
                time.sleep(0.3)
        
            PID_list.append(PID)
            obs_folder_path,file_name = self.fs.create_dir_for_object('img',{'birth_pid':PID})
            os.system(f"mv {fp} {obs_folder_path}/{file_name}")
        self.pp_this_site.load_UTC(PID_list)


            
    def transfer_obs_site_to_site(self,obs_id,site_target):
        pass