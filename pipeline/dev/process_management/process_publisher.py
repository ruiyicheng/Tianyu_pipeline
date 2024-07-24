import mysql.connector
import time
import numpy as np

class process_publisher:
    def __init__(self,host_pika = '192.168.1.107',host_sql = '192.169.1.107'):

        self.cnx = mysql.connector.connect(user='tianyu', password='tianyu',
                              host=host_sql,
                              database='tianyudev')
    def generate_PID(self):
        return (time.time_ns()+np.random.randint(0,1000))*100000+np.random.randint(0,100000)
    
    def calibrate(self,PIDs):
        PID_this = self.publish_CMD('calibrate',stack_this)
        pass
    def align(self,PID):
        pass

    def stacking(self,consume_site_id,consume_group_id,PIDs,num_image_limit = 5):
        Next_hierarchy_PID_list = []
        for i in range((len(PIDs)-1)//num_image_limit+1):
            stack_this = PIDs[i*num_image_limit:(i+1)*num_image_limit]
            if len(stack_this)!=1:
                PID_this = self.publish_CMD(consume_site_id,consume_group_id,'stack',stack_this)
                Next_hierarchy_PID_list.append(PID_this)
            else:
                Next_hierarchy_PID_list.append(stack_this[0])

        if len(Next_hierarchy_PID_list)>1:
            PID_ret = self.stacking(consume_site_id,consume_group_id,Next_hierarchy_PID_list,num_image_limit)
        else:
            return PID_this
        return PID_ret


    def publish_CMD(self,process_site,process_group,CMD,dep_PID_list):
        PID_this = self.generate_PID()

        mycursor = self.cnx.cursor()
        sql = "INSERT INTO process (process_id,process_cmd,process_status_id,process_site_id,process_group_id) VALUES (%s, %s,1,%s,%s)"
        argsql = (PID_this,CMD,process_site)
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
