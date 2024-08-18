import pathlib
import mysql.connector
import numpy as np
from astropy.io import fits
import glob
from tqdm import tqdm
import time
import Tianyu_pipeline.pipeline.utils.sql_interface as sql_interface
import Tianyu_pipeline.pipeline.dev.file_system.file_system as fs

class image_processor:
    def __init__(self):
        self.fs = fs.file_system()
        self.sql_interface = sql_interface.sql_interface()

    def get_dep_img(self,PID):
        res_query = []
        PID_stacker = self.sql_interface.get_process_dependence(PID)
        for PID_stacke_this in PID_stacker:
            # if PID_type=="birth":
            sql = "SELECT * FROM img where img.birth_process_id = %s;"
            args = (PID_stacke_this,)
            result = self.sql_interface.query(sql,args).to_dict("record")
            res_query.extend(result)
        return res_query
    def stacking(self,PID,site_id,method = "mean",PID_type = "birth"):
        def nanaverage(A,weights,axis):
            return np.nansum(A*weights,axis=axis)/((~np.isnan(A))*weights).sum(axis=axis)
        
        res_query = self.get_dep_img(PID)
        path_first_file,name_first_file = self.fs.get_dir_for_object("img",{"image_id":res_query[0]['image_id']})
        header0 = fits.getheader(f"{path_first_file}/{name_first_file}")
        res_dict = np.empty((len(myresult),*(fits.getdata(f"{path_first_file}/{name_first_file}").shape)))#,dtype = np.uint16)
        res_dict[:] = np.nan
        print('importing data...')

        id_list = []
        jd_start_list = []
        jd_mid_list = []
        jd_end_list = []
        n_stack_list = []
        for i,data_line in tqdm(enumerate(res_query)):
            #image_id,jd_utc_start,jd_utc_mid,jd_utc_end,bjd_tdb_start_approximation,bjd_tdb_mid_approximation,bjd_tdb_end_approximation,n_stack,processed,image_type_id,flat_image_id,bias_image_id,x_to_template,y_to_template,obs_id,img_path,deleted_this,hierarchy_this = data_line
            id_list.append(data_line['image_id'])
            jd_start_list.append(data_line["jd_utc_start"])
            jd_mid_list.append(data_line["jd_utc_mid"])
            jd_end_list.append(data_line["jd_utc_end"])
            n_stack_list.append(data_line["n_stack"])
            if type(x_to_template)==type(None) and type(y_to_template)==type(None):
                x_to_template=0
                y_to_template=0
            img_folder,img_name = self.fs.get_dir_for_object("img",{"image_id":data_line['image_id']})
            img_path = f"{img_folder}/{img_name}"
            img_data = fits.getdata(img_path)
            y_to_template = -y_to_template
            if x_to_template>0:
                if y_to_template>0:
                    res_dict[i,:-y_to_template,x_to_template:] = img_data[y_to_template:,:-x_to_template]
                if y_to_template==0:
                    res_dict[i,:,x_to_template:] = img_data[:,:-x_to_template]
                if y_to_template<0:
                    res_dict[i,-y_to_template:,x_to_template:] = img_data[:y_to_template,:-x_to_template]
            if x_to_template==0:
                if y_to_template>0:
                    res_dict[i,:-y_to_template,:] = img_data[y_to_template:,:]
                if y_to_template==0:
                    res_dict[i,:,:] = img_data[:,:]
                if y_to_template<0:
                    res_dict[i,-y_to_template:,:] = img_data[:y_to_template,:]
            if x_to_template<0:
                if y_to_template>0:
                    res_dict[i,:-y_to_template,:x_to_template] = img_data[y_to_template:,-x_to_template:]
                if y_to_template==0:
                    res_dict[i,:,:x_to_template] = img_data[:,-x_to_template:]
                if y_to_template<0:
                    res_dict[i,-y_to_template:,:x_to_template] = img_data[:y_to_template,-x_to_template:]   
        if method=="mean":

            res = nanaverage(res_dict,np.array(n_stack_list),axis = 0)
        if method == "median":
            res = np.nanmedian(res_dict,axis = 0)
        if method == "ZOGY":
            pass

        res = res.astype(np.float32)
        new_name = f"n_{int(np.sum(n_stack_list))}_PID_{PID}_from_{name_first_file.split('from_')[-1]}"
        
        if res_query[0]['image_type_id']==2:
            img_type_id = 3
        else:
            img_type_id = res_query[0]['image_type_id']

        args = (np.min(jd_start_list),np.mean(jd_mid_list),np.max(jd_end_list),int(np.sum(n_stack_list)),1,img_type_id,res_query[0]['flat_image_id'],res_query[0]['bias_image_id'],res_query[0]['dark_image_id'],0,0,res_query[0]['obs_id'],new_name,0,res_query[0]['align_target_image_id'],res_query[0]['batch'],site_id,PID)
        #print(self.obs_id)
        mycursor = self.sql_interface.cnx.cursor()
        sql = "INSERT INTO img (jd_utc_start,jd_utc_mid,jd_utc_end,n_stack,processed,image_type_id,flat_image_id,bias_image_id,dark_image_id,x_to_template,y_to_template,obs_id,img_name,deleted,align_target_image_id,batch,store_site_id,birth_process_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        mycursor.execute(sql,args)
        self.cnx.commit()

        mycursor = self.cnx.cursor()
        mycursor.execute("SELECT LAST_INSERT_ID();")
        myresult = mycursor.fetchall()
        new_img_id = myresult[0][0] #auto_increment
        path_first_file,name_first_file = self.fs.get_dir_for_object("img",{"image_id":new_img_id})
        fits.writeto(f"{path_first_file}/{name_first_file}",res,header = header0,overwrite=True)
        print("recording stacking")
        for i in id_list:
            args = (new_img_id,i)
            sql = "INSERT INTO img_stacking (image_id,stacked_id) VALUES (%s,%s)"
            mycursor.execute(sql,args)
            self.cnx.commit()
        return 1
    
    def calibration(self,process_PID,site_id, sub_img_id = -1, div_img_id = -1,debug = True):#,obs_id=1,hierarchy=2,img_type = "flat_raw",consider_flat = True,consider_bias = True,bias_id = 2,bias_hierarchy = 2,flat_id = 1,flat_hierarchy = 2,keep_origin = True,outpath="samedir",debug = False):
        img_2_cal = self.get_dep_img(process_PID)
        # mycursor = self.cnx.cursor()
        # sql = "SELECT * from img where img.obs_id = %s AND img.image_type_id = %s AND !deleted AND hierarchy = %s;"
        # args = (obs_id,self.image_type_id[img_type],hierarchy)
        # mycursor.execute(sql,args)
        # myresult_corr = mycursor.fetchall()
        if len(img_2_cal)<1:
            print("cannot find target")
            return 0
        print(img_2_cal)

        if sub_img_id>0:
            sub_file_path,sub_file_name = fs.get_dir_for_object("img",{"image_id":sub_img_id})
            sub_img = fits.getdata(f"{sub_file_path}/{sub_file_name}")

        if div_img_id>0:
            div_file_path,div_file_name = fs.get_dir_for_object("img",{"image_id":div_img_id})
            div_img = fits.getdata(f"{div_file_path}/{div_file_name}")          


        for img_target in img_2_cal:
            #image_id,jd_utc_start,jd_utc_mid,jd_utc_end,bjd_tdb_start_approximation,bjd_tdb_mid_approximation,bjd_tdb_end_approximation,n_stack,processed,image_type_id,flat_image_id,bias_image_id,x_to_template,y_to_template,obs_id_this,img_path_this,deleted_this,hierarchy_this = pic
            target_image_path,target_image_name = fs.get_dir_for_object("img",{"image_id":img_target["image_id"]})
            calibrated_image = fits.getdata(f"{target_image_path}/{target_image_name}")
            calibrated_image_header = fits.getheader(f"{target_image_path}/{target_image_name}")
            new_target_image_name = f"cal_{process_PID}_{target_image_name}"
            if self.sql_interface.image_type_id['raw']==img_target["image_type_id"]: 
                img_type_id_this = self.sql_interface.image_type_id['calibrated_single']
            if self.sql_interface.image_type_id['flat_raw']==img_target["image_type_id"]: 
                img_type_id_this = self.sql_interface.image_type_id['flat_debiased']           

            #this_outpath = process_PID
            #print(calibrated_image)
            # if outpath=="samedir":
            #     this_outpath = img_path_this.strip(".fits").strip(".fit")+"_corrected_at_"+str(hash(time.time()))+".fits"
            # else:
            #     this_outpath = outpath+"_corrected_at_"+str(hash(time.time()))+".fits"
            if sub_img_id>0:
                calibrated_image = calibrated_image-sub_img              
                # if img_type=="flat_raw":
                #     sp = calibrated_image.shape
                #     #print(calibrated_image)
                #     calibrated_image = calibrated_image/np.mean(calibrated_image[int(sp[0]/3):int(sp[0]/3*2),int(sp[1]/3):int(sp[1]/3*2)])
                    
            if div_img_id>0:
                calibrated_image = calibrated_image/div_img
            args = (img_target['jd_utc_start'],img_target['jd_utc_mid'],img_target['jd_utc_end'],img_target['n_stack'],1,img_type_id_this,sub_img_id,-1,div_img_id,0,0,img_target['obs_id'],new_target_image_name,0,-1,img_target['batch'],site_id,process_PID)
            mycursor = self.sql_interface.cnx.cursor()
            sql = "INSERT INTO img (jd_utc_start,jd_utc_mid,jd_utc_end,n_stack,processed,image_type_id,flat_image_id,bias_image_id,dark_image_id,x_to_template,y_to_template,obs_id,img_name,deleted,align_target_image_id,batch,store_site_id,birth_process_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            mycursor.execute(sql,args)
            self.cnx.commit()



            mycursor = self.cnx.cursor()
            mycursor.execute("SELECT LAST_INSERT_ID();")
            myresult = mycursor.fetchall()
            new_img_id = myresult[0][0] #auto_increment
            save_image_path,save_image_name = fs.get_dir_for_object("img",{"image_id":new_img_id})

            fits.writeto(f"{save_image_path}/{save_image_name}",calibrated_image.astype('float32'),header = calibrated_image_header,overwrite=True)

            #args = (jd_utc_start,jd_utc_mid,jd_utc_end,bjd_tdb_start_approximation,bjd_tdb_mid_approximation,bjd_tdb_end_approximation,n_stack,processed,self.image_type_id[res_img_type],flat_image_id_used,bias_image_id_used,x_to_template,y_to_template,obs_id_this,this_outpath,0,hierarchy_this)
            #print(len(args))
            #print(self.obs_id)
            # if not debug:
            #     mycursor = self.cnx.cursor()
            #     sql = "INSERT INTO img (jd_utc_start,jd_utc_mid,jd_utc_end,bjd_tdb_start_approximation,bjd_tdb_mid_approximation,bjd_tdb_end_approximation,n_stack,processed,image_type_id,flat_image_id,bias_image_id,x_to_template,y_to_template,obs_id,img_path,deleted,hierarchy) VALUES (%s"+",%s"*(len(args)-1)+")"
            #     mycursor.execute(sql,args)
            #     self.cnx.commit()
            #     print("this_outpath saved at",this_outpath)

            #     return new_img_id
            return 1 
if __name__=="__main__":
    clb = calibrator()
    #print(clb.stacking('/home/yichengrui/workspace/TianYu/pipeline/image_process/out/caliborate/superbias_mgo.fit',2,method = "median"))
    #clb.stacking('/home/yichengrui/workspace/TianYu/pipeline/image_process/out/caliborate/superflat_mgo_uncorred.fit',1,method = "median",img_type="flat_raw")
    #clb.calibration(obs_id=1,hierarchy=2,consider_flat = False,img_type = "flat_raw",bias_id = 2,bias_hierarchy = 2,debug = False)
    #clb.calibration(obs_id=3,hierarchy=1,consider_flat = True,img_type = "science_raw",bias_id = 2,bias_hierarchy = 2,debug = False)
    clb.calibration(obs_id=8,hierarchy=1,consider_flat = True,img_type = "deep_raw",bias_id = 2,bias_hierarchy = 2,debug = False)
    clb.stacking('/home/yichengrui/workspace/TianYu/pipeline/image_process/out/reduced_res/stack_img/M81_l.fit',obs_id = 8,method = "mean",img_type="deep_processed")