from Tianyu_pipeline.pipeline.utils import data_loader as dl
import socket

class process_site_getter:
    def __init__(self):
        self.dl = dl.data_loader()
        pass
    def get_channel(self):
        hostname = socket.gethostname()
        local_host_list = set(socket.gethostbyname_ex(hostname)[2])
        print(local_host_list)
        #print(local_host_list)
        sql = "SELECT * FROM data_process_site;"
        args = tuple()
        res = self.dl.query(sql,args)
        ip_this = set(res['process_site_ip'])&set(local_host_list)
        assert len(ip_this)==1
        ret = res[res['process_site_ip']==list(ip_this)[0]]
        # pika_host = res[res['is_pika_site']].loc[0]['process_site_ip']
        # this_site_index = res[res['is_pika_site']].loc[0]['process_site_id']
        # if pika_host in local_host_list:
        #     pike_host = 'localhost'
        return ret.to_dict('records')[0]
        #return this_site_index,pike_host