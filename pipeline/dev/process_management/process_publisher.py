import mysql.connector
import time
import numpy as np

class process_publisher:
    def __init__(self,host_pika = '192.168.1.107',host_sql = '192.168.1.107',site_id=1,group_id = 1):
        print('connecting to db')
        self.cnx = mysql.connector.connect(user='tianyu', password='tianyu',host=host_sql,database='tianyudev')
        print('done')
        self._default_site_id = site_id
        self._default_group_id = group_id
    @property
    def default_site_id(self):
        if hasattr(self,"_default_site_id"):
            return self._default_site_id

    
    @property
    def default_group_id(self):
        if hasattr(self,"_default_group_id"):
            return self._default_group_id
    def generate_PID(self):
        return (time.time_ns()+np.random.randint(0,1000))*100000+np.random.randint(0,100000)
    
    def calibrate(self,PIDs):
        PID_this = self.publish_CMD('calibrate',stack_this)
        pass
    def align(self,PID):
        pass
    def transfer_img(self,param_dict,consume_site_id=-1,consume_group_id=-1):
        if consume_site_id==-1:
            consume_site_id=self.default_site_id
        if consume_group_id==-1:
            consume_group_id=self.default_group_id 
        PID_this = self.publish_CMD(consume_site_id,consume_group_id,f'transfer_img|{param_dict}',[])
        return PID_this
    def create_dir(self,param_dict,consume_site_id=-1,consume_group_id=-1):
        '''
        example: register_info({"cmd":"INSERT INTO tabname (.....) VALUES (%s,%s.....);","args":'[...]'})
        '''
        if consume_site_id==-1:
            consume_site_id=self.default_site_id
        if consume_group_id==-1:
            consume_group_id=self.default_group_id 
        PID_this = self.publish_CMD(consume_site_id,consume_group_id,f'create_dir|{param_dict}',[])
        return PID_this

    def load_UTC(self,PIDs,consume_site_id=-1,consume_group_id=-1):
        if consume_site_id==-1:
            consume_site_id=self.default_site_id
        if consume_group_id==-1:
            consume_group_id=self.default_group_id         
        PID_this = self.publish_CMD(consume_site_id,consume_group_id,f'load_UTC',PIDs)
        return PID_this

    def register_info(self,param_dict,consume_site_id=-1,consume_group_id=-1):
        '''
        example: register_info({"cmd":"INSERT INTO tabname (.....) VALUES (%s,%s.....);","args":'[...]'})
        '''
        if consume_site_id==-1:
            consume_site_id=self.default_site_id
        if consume_group_id==-1:
            consume_group_id=self.default_group_id         
        PID_this = self.publish_CMD(consume_site_id,consume_group_id,f'register|{param_dict}',[])
        return PID_this
    def stacking(self,PIDs,num_image_limit = 5,consume_site_id=-1,consume_group_id=-1):
        if consume_site_id==-1:
            consume_site_id=self.default_site_id
        if consume_group_id==-1:
            consume_group_id=self.default_group_id  
        Next_hierarchy_PID_list = []
        for i in range((len(PIDs)-1)//num_image_limit+1):
            stack_this = PIDs[i*num_image_limit:(i+1)*num_image_limit]
            if len(stack_this)!=1:
                PID_this = self.publish_CMD(consume_site_id,consume_group_id,'stack',stack_this)
                Next_hierarchy_PID_list.append(PID_this)
            else:
                Next_hierarchy_PID_list.append(stack_this[0])

        if len(Next_hierarchy_PID_list)>1:
            PID_ret = self.stacking(consume_site_id,consume_group_id,Next_hierarchy_PID_list,num_image_limit=num_image_limit)
        else:
            return PID_this
        return PID_ret


    def publish_CMD(self,process_site,process_group,CMD,dep_PID_list):
        PID_this = self.generate_PID()

        mycursor = self.cnx.cursor()
        sql = "INSERT INTO process_list (process_id,process_cmd,process_status_id,process_site_id,process_group_id) VALUES (%s, %s,1,%s,%s)"
        argsql = (PID_this,CMD,process_site,process_group)
        mycursor.execute(sql, argsql)
        self.cnx.commit()
        for PID_dep in dep_PID_list:

            sql = "INSERT INTO process_dependence (master_process_id, dependence_process_id) VALUES (%s, %s)"
            argsql = (PID_this,PID_dep)
            mycursor.execute(sql, argsql)
        self.cnx.commit()
        return PID_this
    

    def test(self,item = 'stacking'):
        if item =="stacking":
            print(self.stacking(1,[self.publish_CMD(1,1,1,"capture",[]) for i in range(35)]))
if __name__=="__main__":
    pp = process_publisher()
    pp.test()
