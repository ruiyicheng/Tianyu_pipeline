import os
import mysql.connector
from pathlib import Path

class file_system:
    # create a file system if not created
    # detect if a folder exist
    # return the path of a file according to parameters passed
    # provide services that managing file system 
    def __init__(self,host_pika = '192.168.1.107',host_sql = '192.168.1.107'):
        self.cnx = mysql.connector.connect(user='tianyu', password='tianyu',
                              host=host_sql,
                              database='tianyudev')
    
    def init_file_system(self,par):
        path_root = par['root_path']
        Path(path_root+"/image").mkdir( parents=True, exist_ok=True)
        Path(path_root+"/logs").mkdir( parents=True, exist_ok=True)
        Path(path_root+"/queue_result/temp").mkdir( parents=True, exist_ok=True)
        Path(path_root+"/queue_result/lc").mkdir( parents=True, exist_ok=True)
        self._path_root = path_root
    @property
    def path_root(self):
        if hasattr(self,"_path_root"):
            return self._path_root
        else: #Obtain the path from database
            pass
    
    def get_dir_for_object(self,obj_type,param_dict):
        '''
        return the path of the object
        '''
        item_name = ''
        dir_path = self.path_root

        if obj_type=='observation':

            if 'observation_name' in param_dict:
                return dir_path,item_name
            if 'observation_id' in param_dict:
                return dir_path,item_name
        if obj_type=='img':#batch+imgtype
            if 'birth_PID' in param_dict:
                return dir_path,item_name
            if 'observation_name' in param_dict:
                return dir_path,item_name
            if 'observation_id' in param_dict:
                return dir_path,item_name
    def create_dir_for_object(self,obj_type,param_dict):
            dir_path,_ = self.get_dir_for_object(obj_type,param_dict)
            Path(dir_path).mkdir( parents=True, exist_ok=True)


