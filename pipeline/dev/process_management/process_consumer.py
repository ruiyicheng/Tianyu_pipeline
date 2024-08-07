import numpy as np
import time
import pika
import pandas as pd
import mysql.connector
import json
import socket

#from Tianyu_pipeline.pipeline.template_generating import image_alignment as il
#from Tianyu_pipeline.pipeline.dev.file_system import file_system as fs



class process_consumer:
    def __init__(self,mode = 'test',host_sql = 'localhost',user_sql = 'root', password_sql = 'root'):
        self.cnx = mysql.connector.connect(user=user_sql, password=password_sql,
                              host=host_sql,
                              database='tianyudev',port = 8889)
    
        self.pika_channel_id, self.pika_host = self.get_channel()
        self.connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=self.pika_host))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='command_queue_'+str(self.pika_channel_id), durable=True)
    def queue_db(self,sql,argsql):
        mycursor = self.cnx.cursor()
        mycursor.execute(sql,argsql)
        myresult = mycursor.fetchall()
        headers = [i[0] for i in mycursor.description]
        res = pd.DataFrame(myresult,columns = headers,dtype=object)
        return res
    def get_channel(self):
        hostname = socket.gethostname()
        local_host_list = set(socket.gethostbyname_ex(hostname)[2])
        print(local_host_list)
        sql = "SELECT * FROM data_process_site;"
        args = tuple()
        res = self.queue_db(sql,args)
        # pika_host = res[res['is_pika_site']].loc[0]['process_site_ip']
        # this_site_index = res[res['is_pika_site']].loc[0]['process_site_id']
        # if pika_host in local_host_list:
        #     pike_host = 'localhost'
        this_site_index = 1
        pike_host = '127.0.0.1'
        return this_site_index,pike_host

    def resolve_msg(self,msg):
        res = msg.split("|")
        PID = int(res[0])
        cmd = res[1]
        if len(res)==2:    
            par = "{}"
        if len(res)==3:
            par = res[2]
        print(par)
        par = json.loads(par)
        return PID,cmd,par
    def work(self,PID,cmd,par):
        print(f'Executing process {PID}, command {cmd} with parameter {par}')
        if cmd == 'stack':
            pass
        if cmd == 'init_dir':
            pass
        if cmd == 'capture':
            pass
        if cmd == 'data_deliver':
            pass
        if cmd == 'data_receive':
            pass
        if cmd == 'calibrate':
            pass
        if cmd == 'image_assess':
            pass
        if cmd == 'alignment':
            pass
        if cmd == 'alignment':
            pass
        time.sleep(0.5)
        #return 0
        return 1
    def callback(self,ch, method, properties, body):
        print(f" [x] Received {body.decode()}, changing db...")
        PID,cmd,par = self.resolve_msg(body.decode())
        mycursor = self.cnx.cursor()
        sql = "UPDATE process SET process_status_id = 3 WHERE process_id = %s;"
        argsql = (PID,)
        mycursor.execute(sql,argsql)
        self.cnx.commit()

        success = self.work(PID,cmd,par)
        
        mycursor = self.cnx.cursor()
        if success:
            suc = 5
        else:
            suc = 4
        sql = "UPDATE process SET process_status_id = %s WHERE process_id = %s;"
        argsql = (suc, PID)
        mycursor.execute(sql,argsql)
        self.cnx.commit()
        print(" [x] Done")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def run(self):
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='command_queue_'+str(self.pika_channel_id) , on_message_callback=self.callback)
        self.channel.start_consuming()

pc = process_consumer()
pc.run()