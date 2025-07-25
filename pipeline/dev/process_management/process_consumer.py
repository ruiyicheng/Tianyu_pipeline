import numpy as np
import time
import pika
import pandas as pd
import mysql.connector
import json
import socket
import os
import datetime
import logging

from Tianyu_pipeline.pipeline.utils import data_loader as dl
from Tianyu_pipeline.pipeline.utils import sql_interface
from Tianyu_pipeline.pipeline.utils.cache_manager import cache_manager
from Tianyu_pipeline.pipeline.dev.file_system import file_system as fs
from Tianyu_pipeline.pipeline.utils import data_transfer as dt
from Tianyu_pipeline.pipeline.image_process import image_processor as image_processor
from Tianyu_pipeline.pipeline.dev.calibration import calibrator as calibrator
from Tianyu_pipeline.pipeline.utils.Bertin import Bertin_tools
import Tianyu_pipeline.pipeline.utils.process_site_getter as psg

class process_consumer(sql_interface.sql_caller):
    def __init__(self,mode = 'test',pika_host = "192.168.1.107",site_id=1,group_id = 1,host_sql = '192.168.1.107',user_sql = 'tianyu', password_sql = 'tianyu'):
        # initialize the sql_interface
        super().__init__(host = host_sql,user = user_sql,password = password_sql)
        # initialize the cache
        self.cache = cache_manager()
        self.cache.connect(self)

        # initialize rabbitmq client
        self.pika_host = pika_host
        self.site_id, self.group_id = site_id,group_id
        self.connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=self.pika_host,heartbeat = 3000,blocked_connection_timeout=10000))
        self.channel = self.connection.channel()

        # initialize other consumer components, establish the bi-directional reference
        # process site getter
        self.psg = psg.process_site_getter()
        self.psg.connect(self)
        self.site_info = self.psg.get_channel()
        # file system
        self.fs = fs.file_system()
        self.fs.connect(self)
        self.fs.init_file_system()
        # data loader
        self.dl = dl.data_loader()
        self.dl.connect(self)

        # file transferer not used in the latest version
        # self.ft = dt.file_transferer()
        # self.ft.connect(self)
        # Bertin softwares (stateless services)
        self.Bertin = Bertin_tools()
        self.Bertin.connect(self)
        # image processor
        self.image_processor = image_processor.image_processor()
        self.image_processor.connect(self)

        # calibrator
        self.calibrator = calibrator.calibrator()
        self.calibrator.connect(self)

        self.channel.queue_declare(queue=f'command_queue_{self.site_id}_{self.group_id}', durable=True)

        # Setup logging for performance metrics
        self.logger = logging.getLogger('process_consumer_metrics')
        self.logger.setLevel(logging.INFO)
        # file_handler = logging.FileHandler(f"{os.getpid()}.log")
        # #set the format of the log, the time and the message
        # formatter = logging.Formatter('%(asctime)s,%(message)s')
        # file_handler.setFormatter(formatter)
        # self.logger.addHandler(file_handler)
        # self.logger.info("linux_pid,process_id,command,start_time,end_time,duration_seconds")
        
        # Record this process's Linux PID
        self.linux_pid = os.getpid()
        print(f"Process Consumer started with Linux PID: {self.linux_pid}")       

    def resolve_msg(self,msg):
        res = msg.split("|")
        PID = int(res[0])
        cmd = res[1]
        if len(res)==2:    
            par = "{}"
        if len(res)==3:
            par = res[2]
        par = par.replace("'",'"')
        print(f'par={par}')
        par = json.loads(par)

        return PID,cmd,par
    def work(self,PID,cmd,par):
        print(f'Executing process {PID}, command {cmd} with parameter {par}')
        start_time = datetime.datetime.now()
        if cmd == 'stack':
            if not "PID_type" in par:
                PID_type = "birth"
            else:
                PID_type = par['PID_type']
            if not "method" in par:
                method = "mean"
            else:
                method = par['method']
            if not "consider_goodness" in par:
                consider_goodness = 0
            else:
                consider_goodness = par['consider_goodness']
            success = self.image_processor.stacking(PID,self.site_id,method=method,PID_type = PID_type,par = par,consider_goodness = consider_goodness)
        if cmd == 'init_dir':
            pass
        if cmd == 'register':
            if type(par['args'])==str:
                argsend = eval(par['args'])
            else:
                argsend = par['args']
            success  = self.dl.register(PID,par['cmd'],argsend)
        if cmd== 'create_dir':
            success = self.fs.create_dir_for_object(par['obj_type'],par['param_dict'])
        if cmd== 'load_UTC':
            success = self.dl.load_UTC(PID)
        # if cmd== 'transfer_img': # In the latest version, RPC call of obseration and scp is used to transfer the image
        #     success = self.ft.transfer_obs_site_to_site(par['obs_id'],par['site_target'])
        if cmd == 'capture':
            pass
        if cmd == 'data_deliver':
            pass
        if cmd == 'data_receive':
            pass
        if cmd == 'calibrate':
            if not "PID_sub" in par:
                par['PID_sub'] = -1
            if not "PID_div" in par:
                par['PID_div'] = -1
            if not "subtract_bkg" in par:
                par['subtract_bkg'] = 0
            if not "obs_id" in par:
                par['obs_id'] = -1
            success = self.image_processor.calibration(PID,self.site_id,par['PID_cal'], sub_img_pid = par['PID_sub'], div_img_pid = par['PID_div'],subtract_bkg = par['subtract_bkg'])
        if cmd == 'select_good_img':
            success = self.image_processor.select_good_img(PID)

        if cmd == 'align':
            success = self.image_processor.alignment(PID,par["template_birth_PID"],par["cal_birth_PID"])
        if cmd=='detect_source':
            success = self.image_processor.detect_source_in_template(PID,par["sky_id"],as_new_template = par["as_new_template"])
        if cmd == "crossmatch":
            success = self.calibrator.crossmatch_external(par['sky_id'])
        if cmd == "select_reference_star":
            success = self.calibrator.select_reference_star(PID,par['PID_template_generating'],par['PID_crossmatch'])
        if cmd == "select_reference_star_and_calibrate":
            success = self.calibrator.select_reference_star_and_calibrate(PID,par['PID_template_generating'])
        if cmd == "extract_flux":
            success = self.image_processor.extract_flux(PID,par['PID_img'],par['PID_detect_source'])
        if cmd == "relative_photometry":
            success = self.calibrator.relative_photometric_calibration(PID,par['PID_reference_star'],par['PID_extract_flux'])
        if cmd == "absolute_photometry":
            success = self.calibrator.absolute_photometric_calibration_single_frame(PID,par['PID_extract_flux'])

        #if cmd == "extract_flux":
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Log performance metrics
        self.logger.info(f"{self.linux_pid},{PID},{cmd},{start_time.isoformat()},{end_time.isoformat()},{duration}")
        #time.sleep(0.5)
        #return 0
        if success:
            print("Success!")
        else:
            print('Failed')
        return success
    def callback(self,ch, method, properties, body):
        print(f" [x] Received {body.decode()}, changing db...")
        PID,cmd,par = self.resolve_msg(body.decode())
        #mycursor = self.sql_interface.cnx.cursor()
        sql = "UPDATE process_list SET process_status_id = 3 WHERE process_id = %s;"
        argsql = (PID,)
        #mycursor.execute(sql,argsql)
        #self.sql_interface.cnx.commit()
        self.sql_interface.execute(sql,argsql)
        success = self.work(PID,cmd,par)
        
        #mycursor = self.sql_interface.cnx.cursor()
        if success:
            suc = 5
        else:
            suc = 4
        sql = "UPDATE process_list SET process_status_id = %s WHERE process_id = %s;"
        argsql = (suc, PID)
        #mycursor.execute(sql,argsql)
        #self.sql_interface.cnx.commit()
        self.sql_interface.execute(sql,argsql)
        print(" [x] Done")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def run(self):
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=f'command_queue_{self.site_id}_{self.group_id}' , on_message_callback=self.callback)
        print(f"[*] Process consumer with Linux PID {self.linux_pid} waiting for messages.")
        self.channel.start_consuming()
#
#pc = process_consumer()
#pc.run()