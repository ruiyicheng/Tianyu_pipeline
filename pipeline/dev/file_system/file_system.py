import os
import mysql.connector
from pathlib import Path
from Tianyu_pipeline.pipeline.middleware.consumer_component import consumer_component

import Tianyu_pipeline.pipeline.utils.process_site_getter as psg
import Tianyu_pipeline.pipeline.utils.sql_interface as sql_interface
class file_system(consumer_component):
    # create a file system if not created
    # detect if a folder exist
    # return the path of a file according to parameters passed
    # provide services that managing file system 
    def __init__(self,host_pika = '192.168.1.107',host_sql = '192.168.1.107',_path_root = None):
        super().__init__()
        #self.sql_interface = sql_interface.sql_interface()
        # self.consumer.psg = psg.process_site_getter()
        # self.consumer.site_info = self.psg.get_channel()
        # self.init_file_system()
        if not _path_root is  None:
            self._path_root = _path_root
    def init_file_system(self):
        path_root = self.path_root
        Path(path_root+"/image").mkdir( parents=True, exist_ok=True)
        Path(path_root+"/cache").mkdir( parents=True, exist_ok=True)
        Path(path_root+"/logs").mkdir( parents=True, exist_ok=True)
        Path(path_root+"/queue_result/temp").mkdir( parents=True, exist_ok=True)
        Path(path_root+"/queue_result/lc").mkdir( parents=True, exist_ok=True)
        self._path_root = path_root
    @property
    def path_root(self):
        if hasattr(self,"_path_root"):
            return self._path_root
        else: #Obtain the path from database
            self._path_root = self.consumer.site_info['file_path']
            return self._path_root
            
    
    def get_dir_for_object(self,obj_type,param_dict,site_id = -1):
        '''
        return the path of the object
        '''
        item_name = ''
        if site_id == -1:
            path_root = self.path_root
        else:
            path_root = self.consumer.psg.get_channel(channel_id = site_id)['file_path']
        if obj_type=='observation':
            if 'full_observation_info' in param_dict:
                result_dict = param_dict

                
            if 'observation_pid' in param_dict:
                sql = '''SELECT site.obs_site_name as site_name,ins.instrument_name as instrument_name, tg.target_name as target_name, filters.filter_name as filter_name, obs.process_id as pid FROM observation as obs 
LEFT JOIN target_n AS tg 
ON tg.target_id = obs.target_id
LEFT JOIN instrument AS ins
ON ins.instrument_id = obs.instrument_id
LEFT JOIN obs_site AS site
ON site.obs_site_id= obs.obs_site_id
LEFT JOIN filters
ON filters.filter_id = ins.filter_id
WHERE obs.process_id=%s;'''
                args = (param_dict['observation_pid'],)
                result = self.sql_interface.query(sql,args)
                assert len(result)==1
                result_dict = result.to_dict('records')[0]

            if 'observation_id' in param_dict:
                sql = '''SELECT site.obs_site_name as site_name,ins.instrument_name as instrument_name, tg.target_name as target_name, filters.filter_name as filter_name, obs.process_id as pid FROM observation as obs 
LEFT JOIN target_n AS tg 
ON tg.target_id = obs.target_id
LEFT JOIN instrument AS ins
ON ins.instrument_id = obs.instrument_id
LEFT JOIN obs_site AS site
ON site.obs_site_id= obs.obs_site_id
LEFT JOIN filters
ON filters.filter_id = ins.filter_id
WHERE obs.obs_id=%s;'''
                args = (param_dict['observation_id'],)
                result = self.sql_interface.query(sql,args)
                assert len(result)==1
                result_dict = result.to_dict('records')[0]
            site_name = result_dict['site_name']           
            instrument_name = result_dict['instrument_name']
            target_name = result_dict['target_name']
            filter_name = result_dict['filter_name']
            process_id = result_dict['pid']
            dir_path = path_root+f'/image/{site_name}/{instrument_name}/{target_name}/{filter_name}/{process_id}'

            return dir_path,item_name
        
        if obj_type=='img':#batch+imgtype
            if 'birth_pid' in param_dict:
                sql = '''SELECT img.batch AS batch, imgt.image_type as image_type, img.is_mask as is_mask, img.obs_id as obs_id,img.img_name as img_name FROM img
LEFT JOIN image_type AS imgt ON imgt.image_type_id = img.image_type_id
WHERE img.birth_process_id = %s; 
''' 
                args = (param_dict['birth_pid'],)
                result = self.sql_interface.query(sql,args)
                assert len(result)==1
                result_dict = result.to_dict('records')[0]
            if 'image_id' in param_dict:
                sql = '''SELECT img.batch AS batch, imgt.image_type as image_type, img.is_mask as is_mask, img.obs_id as obs_id,img.img_name as img_name FROM img
LEFT JOIN image_type AS imgt ON imgt.image_type_id = img.image_type_id
WHERE img.image_id = %s; 
''' 
                args = (param_dict['image_id'],)
                result = self.sql_interface.query(sql,args)
                assert len(result)==1
                result_dict = result.to_dict('records')[0]
            #print(result)
            obs_path,_ = self.get_dir_for_object('observation',{'observation_id':result_dict['obs_id']},site_id = site_id)
            batch_name = result_dict['batch']
            type_name = result_dict['image_type']
            
            
            if not result_dict['is_mask']:
                img_mask = 'frame'
            else:
                img_mask = 'mask'
            item_name = result_dict['img_name']

            return f'{obs_path}/{batch_name}/{type_name}/{img_mask}',item_name
 
    def create_dir_for_object(self,obj_type,param_dict):
        #try:
        dir_path,_ = self.get_dir_for_object(obj_type,param_dict)
        print('object created at',dir_path)
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return 1
        #except:
        #    print('Create dir failed!')
        #    return 0


