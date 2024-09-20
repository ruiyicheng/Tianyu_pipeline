from Tianyu_pipeline.pipeline.utils import sql_interface 
from Tianyu_pipeline.pipeline.dev.file_system import file_system 
import socket

class deleter:
    def __init__(self):
        self.sql_interface = sql_interface.sql_interface()
        self.fs = file_system.file_system()
    def get_channel(self,channel_id = -1):
        if channel_id == -1:
            #hostname = socket.gethostname()
            #local_host_list = set(socket.gethostbyname_ex(hostname)[2])
            #print(local_host_list)
            #print(local_host_list)
            sql = "SELECT * FROM data_process_site;"
            args = tuple()
            res = self.sql_interface.query(sql,args)
            #ip_this = set(res['process_site_ip'])&set(local_host_list)
            #assert len(ip_this)==1
            ip_this = (([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")] or [[(s.connect(("8.8.8.8", 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) + ["no IP found"])[0]
            print(ip_this)
            ret = res[res['process_site_ip']==ip_this]
        else:
            sql = "SELECT * FROM data_process_site where process_site_id = %s;"
            args = (channel_id,)
            ret = self.sql_interface.query(sql,args)
            assert len(ret)==1
        # pika_host = res[res['is_pika_site']].loc[0]['process_site_ip']
        # this_site_index = res[res['is_pika_site']].loc[0]['process_site_id']
        # if pika_host in local_host_list:
        #     pike_host = 'localhost'
        return ret.to_dict('records')[0]
    def files(self):
        site_info = self.get_channel()
        import glob
        import tqdm
        fp = site_info['file_path']

        image_files = glob.glob(fp+'/image/*/*/*/*/*/*/*/*/*.fit')+glob.glob(fp+'/image/*/*/*/*/*/*/*/*/*.fits')
        print(len(image_files))
        sql = "SELECT * FROM img;"
        args = ()
        res = self.sql_interface.query(sql,args)
        path_db = []
        for img_id in tqdm.tqdm(res["image_id"]):
            #print(img_id)
            dir,fn = self.fs.get_dir_for_object("img",{"image_id":img_id}) 
            path_db.append(dir+'/'+fn)
        print(len(path_db))
        res_files = set(image_files)-set(path_db)
        print(res_files)
        print(len(res_files))
        import os
        for f in tqdm.tqdm(res_files):
            os.system(f"rm {f}")




dl = deleter()
dl.files()