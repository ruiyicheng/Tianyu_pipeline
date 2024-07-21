import mysql.connector
import time
import numpy as np
import pika
import pandas as pd

class process_manager:
    def __init__(self,host_pika = 'localhost',host_sql = 'localhost'):

        self.cnx = mysql.connector.connect(user='root', password='root',
                              host=host_sql,
                              database='tianyudev',port = 8889)
        self.host_pika = host_pika



        
    def __del__(self):
        print("Shut down")
        self.connection.close()

    def send(self,site_id,message):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.host_pika))
        self.channel = self.connection.channel()

        self.channel.queue_declare(queue='command_queue_'+str(site_id), durable=True)
        #message = ' '.join(sys.argv[1:]) or "Hello World!"
        self.channel.basic_publish(
            exchange='',
            routing_key='command_queue_'+str(site_id),
            body=message,
            properties=pika.BasicProperties(
                delivery_mode=pika.DeliveryMode.Persistent
            ))
        print(f" [x] Sent {message}")
        time.sleep(0.0001)
    def generate_PID(self):
        return (time.time_ns()+np.random.randint(0,1000))*100000+np.random.randint(0,1000000)
    def queue_db(self,sql,argsql):
        mycursor = self.cnx.cursor()
        mycursor.execute(sql,argsql)
        myresult = mycursor.fetchall()
        headers = [i[0] for i in mycursor.description]
        res = pd.DataFrame(myresult,columns = headers,dtype=object)
        return res
        # df_processes.columns = headers
    def scan_waiting_process(self,look_for_day = 1):
        d2ns = 24*3600*10**9
        sql = "SELECT * FROM process WHERE process_id > %s and (process_status_id=1 or process_status_id=4);"
        argsql = ((time.time_ns()-d2ns*look_for_day)*100000,)
        df_processes = self.queue_db(sql,argsql)
        # sql = "SELECT * FROM process as p WHERE p.process_id > %s JOIN process_dependence AS pd ON pd.master_process_id = p.process_id or pd.dependence_process_id = p.process_id; "
        # argsql = (time.time_ns()-d2ns*look_for_day,)
        # df_processes = self.queue_db(sql,argsql)
        return df_processes
        # def delete(self,process_id):
        #     d2ns = 24*3600*10**9
        #     sql = "SELECT * FROM process WHERE process_id > %s and process_status_id=1;"
        #     argsql = (time.time_ns()-d2ns*look_for_day,)
        #     df_processes = self.queue_db(sql,argsql)
            # sql = "SELECT * FROM process as p WHERE p.process_id > %s JOIN process_dependence AS pd ON pd.master_process_id = p.process_id or pd.dependence_process_id = p.process_id; "
            # argsql = (time.time_ns()-d2ns*look_for_day,)
            # df_processes = self.queue_db(sql,argsql)
            # return df_processes
    def submit_mission(self,site_id,process_id,process_cmd): #change mission status and submit mission to queue
        message = str(process_id)+"|"+process_cmd   # Format of cmd should be cmdname|{par.json}
        mycursor = self.cnx.cursor()

        sql = "UPDATE process SET process_status_id = 2 WHERE process_id = %s;"
        argsql = (process_id,)
        mycursor.execute(sql,argsql)
        self.send(site_id,message)
        self.cnx.commit()

    def spin(self):
        while 1:
            df_p = self.scan_waiting_process()
            df_p['process_id'] = df_p['process_id'].apply(int)
            print(len(df_p),'unfinished process detected')

            for i,r in df_p.iterrows():
                sql = "SELECT * FROM process where process_id in (SELECT dependence_process_id from process as p JOIN process_dependence AS pd ON pd.master_process_id = p.process_id where pd.master_process_id = %s) and process_status_id!=5;"
                argsql = (r['process_id'],)
                res = self.queue_db(sql,argsql)
                #print(res)
                if len(res)==0:# all dependences are satisfied
                    self.submit_mission(r['process_site_id'],r['process_id'],r['process_cmd'])
            time.sleep(0.5)
            self.cnx.commit()





if __name__=='__main__':

    pm = process_manager()
    print(len(pm.spin()))