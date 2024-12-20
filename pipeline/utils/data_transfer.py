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
        self.image_type_id = self.sql_interface.image_type_id
    def transfer_file_to_file_system(self,input_path,observation_id,image_info_dict={'image_type':"raw"}):
        '''
        This function is used for debug. Observation is created manually. Input the Observation, and raw data file path so it will transfer it to the right file system. 
        The pictures will be registered.
        In real runtime. This step would be executed automatically because of real-time demand and large number of sky region.
        '''
        file_paths = sorted(glob.glob(input_path+'*'))
        self.fs.init_file_system()


        sql = '''SELECT obs.batch_size as batch_size FROM observation as obs WHERE obs.obs_id=%s;'''
        args = (observation_id,)
        result = self.sql_interface.query(sql,args)
        assert len(result)==1
        result = result.to_dict('records')[0]
        batch_size = result['batch_size']
        #3 steps: register; create dir; mv
        PID_list = []
        image_type_id = self.image_type_id[image_info_dict['image_type']]
        for i,fp in enumerate(file_paths):
            batch_number = i//batch_size+1   
            PID = self.pp_this_site.register_info({"cmd":"INSERT INTO img (store_site_id,batch,image_type_id,obs_id,img_name) VALUES (%s,%s,%s,%s,%s);","args":[self.site_id,batch_number,image_type_id,observation_id,fp.split('/')[-1]]})
            print('PID=',PID)
            while True:
                print(f"waiting for img {i} registration")
                time.sleep(0.5)
                sql = '''SELECT * FROM img WHERE img.birth_process_id=%s;'''
                args = (PID,)
                result = self.sql_interface.query(sql,args)
                print(result)
                if len(result==1):
                    break
                time.sleep(0.3)
        
            PID_list.append(PID)
            obs_folder_path,file_name = self.fs.get_dir_for_object('img',{'birth_pid':PID})
            success = self.fs.create_dir_for_object('img',{'birth_pid':PID})

            os.system(f"mv {fp} {obs_folder_path}/{file_name}")
        self.pp_this_site.load_UTC(PID_list)
        mycursor = self.sql_interface.cnx.cursor()
        sql = 'UPDATE observation SET n_pic=%s where obs_id=%s;'
        args = (len(file_paths),observation_id)
        mycursor.execute(sql,args)
        self.sql_interface.cnx.commit()



            
    def transfer_obs_site_to_site(self,obs_id,site_target):
        #1. Create dir; 2. scp 3. mv
        pp_target_site = process_pub.process_publisher(site_id = site_target, group_id = site_target)
        sql = '''SELECT image_id FROM img WHERE obs_id=%s and store_site_id=%s;'''
        args = (obs_id,self.site_id)
        result = self.sql_interface.query(sql,args)
        print(result)
        remote_site_info = self.fs.psg.get_channel(channel_id = site_target)
        for i,r in result.iterrows():
            img_id = int(r['image_id'])
            PID = pp_target_site.create_dir({"obj_type":"img","param_dict":{"image_id":img_id}},consume_site_id=site_target,consume_group_id=site_target)
            while True:
                print(f"waiting for img {i} registration on remote")
                time.sleep(0.5)
                sql = '''SELECT * FROM process_list WHERE process_id=%s;'''
                args = (PID,)
                result = self.sql_interface.query(sql,args)
                assert len(result)==1
                result = result.to_dict('records')[0]
                if result['process_status_id']==5:
                    break
                time.sleep(0.3)
            path_remote,fn_remote = self.fs.get_dir_for_object('img',{'image_id':img_id},site_id = site_target)
            path,fn = self.fs.get_dir_for_object('img',{'image_id':img_id},site_id = -1)
            os.system(f"scp {path}/{fn} {remote_site_info['user_name']}@{remote_site_info['process_site_ip']}:{path_remote}/{fn_remote}")
            mycursor = self.sql_interface.cnx.cursor()
            sql = 'UPDATE img SET store_site_id=%s where image_id=%s;'
            args = (site_target,img_id)
            mycursor.execute(sql,args)
            self.sql_interface.cnx.commit()
            os.system(f"rm {path}/{fn}")
        return 1

