import mysql.connector
import time
import numpy as np

class process_publisher:
    def __init__(self,host_pika = 'localhost',host_sql = 'localhost'):

        self.cnx = mysql.connector.connect(user='root', password='root',
                              host=host_sql,
                              database='tianyudev',port = 8889)
    def generate_PID(self):
        return (time.time_ns()+np.random.randint(0,1000))*100000+np.random.randint(0,100000)
    
    def calibrate(self,PIDs):
        PID_this = self.publish_CMD('calibrate',stack_this)
        pass
    def align(self,PID):
        pass

    def stacking(self,PIDs,num_image_limit = 5):
        Next_hierarchy_PID_list = []
        for i in range((len(PIDs)-1)//num_image_limit+1):
            stack_this = PIDs[i*num_image_limit:(i+1)*num_image_limit]
            if len(stack_this)!=1:
                PID_this = self.publish_CMD('stack',stack_this)
                Next_hierarchy_PID_list.append(PID_this)
            else:
                Next_hierarchy_PID_list.append(stack_this[0])

        if len(Next_hierarchy_PID_list)>1:
            PID_ret = self.stacking(Next_hierarchy_PID_list,num_image_limit)
        else:
            return PID_this
        return PID_ret


    def publish_CMD(self,process_site,CMD,dep_PID_list):
        PID_this = self.generate_PID()

        mycursor = self.cnx.cursor()
        sql = "INSERT INTO process (process_id,process_cmd,process_status_id) VALUES (%s, %s,1)"
        argsql = (PID_this,CMD)
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
            print(self.stacking([self.publish_CMD("capture",[]) for i in range(33)]))
if __name__=="__main__":
    pp = process_publisher()
    pp.test()
