import mysql.connector
import numpy as np
from astropy.io import fits
import glob
from tqdm import tqdm
import time



class calibrator:
    def __init__(self):
        self.cnx = mysql.connector.connect(user='root', password='',
                            host='127.0.0.1',
                            database='mgo')
        self.observation_type_id = self.get_table_dict("observation_type")
        self.target_id = self.get_table_dict("target_n")
        self.image_type_id = self.get_table_dict("image_type")
        self.instrument_id = self.get_table_dict("instrument")
        self.site_id = self.get_table_dict("obs_site")
        self.observer_id = self.get_table_dict("observer")
        print(self.observation_type_id,self.target_id,self.image_type_id,self.instrument_id,self.site_id,self.observer_id)
    def get_table_dict(self,table,index_key=1,index_value=0):
        mycursor = self.cnx.cursor()
        mycursor.execute("SELECT * from "+table+";")
        myresult = mycursor.fetchall()
        # print(myresult)
        res_dict = {}
        for row in myresult:
            res_dict[row[index_key]] = row[index_value]
        return res_dict
    def stacking(self,outpath,stack_img_id_list = [],mode = "query",obs_id = 2,img_type = "bias",hierarchy = 1,keep_origin = True,img_use = "all",use_GPU = False,method = "median",batch_stack = False, batch_size = 5):

        mycursor = self.cnx.cursor()
        if mode=="query":
            sql = "SELECT * from img where img.obs_id = %s AND img.image_type_id = %s AND !deleted AND hierarchy = %s;"
            args = (obs_id,self.image_type_id[img_type],hierarchy)
            mycursor.execute(sql,args)
            myresult = mycursor.fetchall()

            if img_use != "all":
                myresult = myresult[img_use[0]:img_use[1]]
            if len(myresult)==0:
                return -1
        
        if mode == "fixed_id":
            myresult = []
            for stack_img_id_this in stack_img_id_list:
                sql = "SELECT * from img where img.image_id = %s;"
                args = (stack_img_id_this,)
                mycursor.execute(sql,args)
                myresult_this = mycursor.fetchall()
                myresult.append(myresult_this[0])



        res_dict = np.empty((len(myresult),*(fits.getdata(myresult[0][15]).shape)))#,dtype = np.uint16)
        res_dict[:] = np.nan
        print('importing data...')

        id_list = []
        jd_start_list = []
        jd_mid_list = []
        jd_end_list = []
        n_stack_list = []
        for i,data_line in tqdm(enumerate(myresult)):
            image_id,jd_utc_start,jd_utc_mid,jd_utc_end,bjd_tdb_start_approximation,bjd_tdb_mid_approximation,bjd_tdb_end_approximation,n_stack,processed,image_type_id,flat_image_id,bias_image_id,x_to_template,y_to_template,obs_id,img_path,deleted_this,hierarchy_this = data_line
            id_list.append(image_id)
            jd_start_list.append(jd_utc_start)
            jd_mid_list.append(jd_utc_mid)
            jd_end_list.append(jd_utc_end)
            n_stack_list.append(n_stack)
            if type(x_to_template)==type(None) and type(y_to_template)==type(None):
                x_to_template=0
                y_to_template=0
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

            res = np.nanmean(res_dict,axis = 0)
        if method == "median":
            res = np.nanmedian(res_dict,axis = 0)
        res = res.astype(np.float32)
        fits.writeto(outpath,res,overwrite=True)

        args = (np.min(jd_start_list),np.mean(jd_mid_list),np.max(jd_end_list),int(np.sum(n_stack_list)),1,image_type_id,flat_image_id,bias_image_id,0,0,obs_id,outpath,0,hierarchy_this+1)
        #print(self.obs_id)
        mycursor = self.cnx.cursor()
        sql = "INSERT INTO img (jd_utc_start,jd_utc_mid,jd_utc_end,n_stack,processed,image_type_id,flat_image_id,bias_image_id,x_to_template,y_to_template,obs_id,img_path,deleted,hierarchy) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        mycursor.execute(sql,args)
        self.cnx.commit()

        mycursor = self.cnx.cursor()
        mycursor.execute("SELECT LAST_INSERT_ID();")
        myresult = mycursor.fetchall()
        new_img_id = myresult[0][0] #auto_increment
        print("recording stacking")
        for i in id_list:
            args = (new_img_id,i)
            sql = "INSERT INTO img_stacking (image_id,stacked_id) VALUES (%s,%s)"
            mycursor.execute(sql,args)
            self.cnx.commit()
        return new_img_id
    
    def calibration(self,obs_id=1,hierarchy=2,img_type = "flat_raw",consider_flat = True,consider_bias = True,bias_id = 2,bias_hierarchy = 2,flat_id = 1,flat_hierarchy = 2,keep_origin = True,outpath="samedir",debug = False):
        mycursor = self.cnx.cursor()
        sql = "SELECT * from img where img.obs_id = %s AND img.image_type_id = %s AND !deleted AND hierarchy = %s;"
        args = (obs_id,self.image_type_id[img_type],hierarchy)
        mycursor.execute(sql,args)
        myresult_corr = mycursor.fetchall()


        if len(myresult_corr)<1:
            print("cannot find target")
            return -1
        print(myresult_corr)
        if consider_flat:
            mycursor = self.cnx.cursor()
            sql = "SELECT * from img where img.obs_id = %s AND img.image_type_id = %s AND !deleted AND hierarchy = %s;"
            args = (flat_id,self.image_type_id['flat_debiased'],flat_hierarchy)
            mycursor.execute(sql,args)
            myresult = mycursor.fetchall()
            if len(myresult)!=1:
                print('cannot find flat')
                return -1
            
            flat = fits.getdata(myresult[0][15])
            flat_image_id_used = myresult[0][0]
        else:
            flat_image_id_used = None


        if consider_bias:
            mycursor = self.cnx.cursor()
            sql = "SELECT * from img where img.obs_id = %s AND img.image_type_id = %s AND !deleted AND hierarchy = %s;"
            args = (bias_id,self.image_type_id['bias'],bias_hierarchy)
            mycursor.execute(sql,args)
            myresult = mycursor.fetchall()
            if len(myresult)!=1:
                print('cannot find bias')
                return -1
            bias = fits.getdata(myresult[0][15])
            #print(bias)
            bias_image_id_used = myresult[0][0]
        else:
            bias_image_id_used = None


        if img_type=="flat_raw":
            res_img_type = "flat_debiased"
        if img_type=="science_raw":
            res_img_type = "science_processed"
        if img_type=="deep_raw":
            res_img_type = "deep_processed"
        if img_type=="planet_raw":
            res_img_type = "planet_processed"
        for pic in myresult_corr:
            image_id,jd_utc_start,jd_utc_mid,jd_utc_end,bjd_tdb_start_approximation,bjd_tdb_mid_approximation,bjd_tdb_end_approximation,n_stack,processed,image_type_id,flat_image_id,bias_image_id,x_to_template,y_to_template,obs_id_this,img_path_this,deleted_this,hierarchy_this = pic
            calibrated_image = fits.getdata(img_path_this)
            calibrated_image_header = fits.getheader(img_path_this)
            #print(calibrated_image)
            if outpath=="samedir":
                this_outpath = img_path_this.strip(".fits").strip(".fit")+"_corrected_at_"+str(hash(time.time()))+".fits"
            else:
                this_outpath = outpath+"_corrected_at_"+str(hash(time.time()))+".fits"
            if consider_bias:
                calibrated_image = calibrated_image-bias

                
                if img_type=="flat_raw":
                    sp = calibrated_image.shape
                    #print(calibrated_image)
                    calibrated_image = calibrated_image/np.mean(calibrated_image[int(sp[0]/3):int(sp[0]/3*2),int(sp[1]/3):int(sp[1]/3*2)])
                    
            if consider_flat:
                calibrated_image = calibrated_image/flat

            fits.writeto(this_outpath,calibrated_image.astype('float32'),header = calibrated_image_header,overwrite=True)

            args = (jd_utc_start,jd_utc_mid,jd_utc_end,bjd_tdb_start_approximation,bjd_tdb_mid_approximation,bjd_tdb_end_approximation,n_stack,processed,self.image_type_id[res_img_type],flat_image_id_used,bias_image_id_used,x_to_template,y_to_template,obs_id_this,this_outpath,0,hierarchy_this)
            #print(len(args))
            #print(self.obs_id)
            if not debug:
                mycursor = self.cnx.cursor()
                sql = "INSERT INTO img (jd_utc_start,jd_utc_mid,jd_utc_end,bjd_tdb_start_approximation,bjd_tdb_mid_approximation,bjd_tdb_end_approximation,n_stack,processed,image_type_id,flat_image_id,bias_image_id,x_to_template,y_to_template,obs_id,img_path,deleted,hierarchy) VALUES (%s"+",%s"*(len(args)-1)+")"
                mycursor.execute(sql,args)
                self.cnx.commit()
                print("this_outpath saved at",this_outpath)
if __name__=="__main__":
    clb = calibrator()
    #print(clb.stacking('/home/yichengrui/workspace/TianYu/pipeline/image_process/out/caliborate/superbias_mgo.fit',2,method = "median"))
    #clb.stacking('/home/yichengrui/workspace/TianYu/pipeline/image_process/out/caliborate/superflat_mgo_uncorred.fit',1,method = "median",img_type="flat_raw")
    #clb.calibration(obs_id=1,hierarchy=2,consider_flat = False,img_type = "flat_raw",bias_id = 2,bias_hierarchy = 2,debug = False)
    #clb.calibration(obs_id=3,hierarchy=1,consider_flat = True,img_type = "science_raw",bias_id = 2,bias_hierarchy = 2,debug = False)
    clb.calibration(obs_id=8,hierarchy=1,consider_flat = True,img_type = "deep_raw",bias_id = 2,bias_hierarchy = 2,debug = False)
    clb.stacking('/home/yichengrui/workspace/TianYu/pipeline/image_process/out/reduced_res/stack_img/M81_l.fit',obs_id = 8,method = "mean",img_type="deep_processed")