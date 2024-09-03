import pathlib
import mysql.connector
import numpy as np
from astropy.io import fits
import glob
from tqdm import tqdm
import time
import sep
import Tianyu_pipeline.pipeline.utils.sql_interface as sql_interface
import Tianyu_pipeline.pipeline.dev.file_system.file_system as fs
import Tianyu_pipeline.pipeline.dev.process_management.process_publisher as process_pub 
import Tianyu_pipeline.pipeline.utils.process_site_getter as psg
class image_processor:
    def __init__(self,site_id=-1,group_id=-1):
        self.psg = psg.process_site_getter()
        self.site_info = self.psg.get_channel()

        # if site_id==-1:
        #     site_id = self.site_info['site_id']
        # if group_id==-1:
        #     group_id = site_id

        self.fs = fs.file_system()
        self.sql_interface = sql_interface.sql_interface()
        #self.pp_this_site = process_pub.process_publisher(site_id = site_id, group_id = group_id)

    def detect_source(self,img_pid,sky_pid,as_new_template = True):
        pass
    # def generate_template_image(self,site_id,sky_id,obs_id,n_stack=1):
    #     #align stack map to sky
    #     
    #     outpath = "/home/yichengrui/workspace/TianYu/pipeline/image_process/out/reduced_res/stack_img/template_obs_"+str(alignment_img_obs_id)+"_hierarchy_"+str(alignment_img_hierarchy)+"_imgtype_"+str(alignment_img_type)+"_"+str(hash(time.time()))+".fit"
    #     mycursor = self.dl.cnx.cursor()
    #     arg = (alignment_img_obs_id,alignment_img_hierarchy,self.dl.image_type_id[alignment_img_type])
    #     sql = "SELECT i.obs_id,i.img_path,i.image_id from img as i where i.obs_id = %s and hierarchy = %s and i.image_type_id = %s;"
    #     mycursor.execute(sql,arg)
    #     myresult = mycursor.fetchall()
    #     if len(myresult)<1:
    #         print("Cannot fing image")
    #         return -1
        
    #     base_id = myresult[0][2]
    #     alignment_result = self.al.get_deviation(base_id,alignment_img_obs_id, alignment_img_hierarchy, alignment_img_type,good_star_threshold = good_star_threshold)

    #     used_img_list = []
    #     for alires in alignment_result:
    #         if alires[3]<good_img_threshold:
    #             continue
    #         else:
    #             used_img_list.append(alires[0])
    #     new_img_id = self.cl.stacking(outpath,stack_img_id_list = used_img_list,mode = "fixed_id",method = "mean")

    #     mycursor = self.dl.cnx.cursor()
    #     arg = (new_img_id,)
    #     sql = "INSERT INTO sky (template_image_id) VALUES (%s);"
    #     mycursor.execute(sql,arg)
    #     self.dl.cnx.commit()

    #     mycursor = self.dl.cnx.cursor()
    #     mycursor.execute("SELECT LAST_INSERT_ID();")
    #     myresult = mycursor.fetchall()
    #     new_sky_id = myresult[0][0] #auto_increment
        
    #     return new_sky_id,outpath
    
    def get_dep_img(self,PID,process_type = "birth"):
        res_query = []
        PID_stacker = self.sql_interface.get_process_dependence(PID)
        for PID_stacke_this in PID_stacker:
            # if PID_type=="birth":
            if process_type == "birth":
                sql = "SELECT * FROM img where img.birth_process_id = %s;"
            if process_type == "align":
                sql = "SELECT * FROM img where img.align_process_id = %s;"
            args = (PID_stacke_this,)
            result = self.sql_interface.query(sql,args).to_dict("records")
            res_query.extend(result)
        return res_query
    


    def alignment(self,PID,template_img_pid,target_img_pid,max_deviation = 400):
        # Align 2 image! 2 possibility
        # 1 template_img not resolved, need to resolve the target in template_img
        # 2 template_img resolved, need to get star position from db
        

        sql = "SELECT sim.image_id as image_id,ts.source_id as source_id, ts.x_template as x_template,ts.y_template as y_template FROM sky_image_map AS sim INNER JOIN sky on sky.sky_id=sim.sky_id INNER JOIN tianyu_source as ts on ts.sky_id=sky.sky_id WHERE template_in_use=1 AND birth_process_id=%s;"
        args = (template_img_pid,)
        result = self.sql_interface.query(sql,args)
        if len(result)>=1:# resolved template image
            x_stars_template = np.array(result['x_template'])
            y_stars_template = np.array(result['y_template'])
        if result==0: # not resolved image
            print("Resolving stars in template image using sextractor...")
            img_folder,img_name = self.fs.get_dir_for_object("img",{"birth_pid":template_img_pid})
            img_path = f"{img_folder}/{img_name}"
            img_data = fits.getdata(img_path).byteswap().newbyteorder()
            objects = sep.extract(img_data,50,minarea=30)
            if len(objects)<3:
                print('Too few star resolved in image')
                return 0
            x_stars_template = objects['x']
            y_stars_template = objects['y']
        sql = "SELECT * FROM img WHERE birth_process_id = %s"
        args = (template_img_pid,)
        result_template = self.sql_interface.query(sql,args)
        assert len(result_template)==1
        template_img_id = result_template.to_dict('records')[0]['image_id']
        #mycursor = self.dl.cnx.cursor()
        #arg = (alignment_img_obs_id,alignment_img_hierarchy,self.dl.image_type_id[alignment_img_type])
        #sql = "SELECT i.obs_id,i.img_path,i.image_id from img as i where i.obs_id = %s and hierarchy = %s and i.image_type_id = %s;"
        #mycursor.execute(sql,arg)
        #myresult = mycursor.fetchall()
        # res_list = []
        #for target_img_pid in target_img_pid_list:
        img_folder,img_name = self.fs.get_dir_for_object("img",{"birth_pid":target_img_pid})
        img_data = fits.getdata(img_path).byteswap().newbyteorder()
        objects = sep.extract(img_data,50,minarea=30)
        if len(objects)<3:
            print('Too few star resolved in image')
            return 0

        # fits_data = self.se.use(res[1],keep_out = False,use_sep = True)
        #fits_res = fits.open(res_output)
        #os.system("rm "+res_output)
        x_stars_this = np.squeeze(objects['x'])
        y_stars_this = np.squeeze(objects['y'])
        # xx_stars = np.squeeze(fits_data['x2']).reshape(-1,1)
        # yy_stars = np.squeeze(fits_data['y2']).reshape(-1,1)
        # xy_stars = np.squeeze(fits_data['xy']).reshape(-1,1)
        # lambda1 = ((xx_stars+yy_stars)/2+np.sqrt(((xx_stars-yy_stars)/2)**2+xy_stars**2)).reshape(-1,1)
        # lambda2 = ((xx_stars+yy_stars)/2-np.sqrt(((xx_stars-yy_stars)/2)**2+xy_stars**2)).reshape(-1,1)
        # print(np.sum((lambda1/lambda2)<1.2)/len(lambda1),np.mean(lambda1/lambda2),np.median(lambda1/lambda2),np.min(lambda1/lambda2))

        dx = (x_stars_template.reshape(-1,1)-x_stars_this.reshape(1,-1)).reshape(-1,1)
        dy = (y_stars_template.reshape(-1,1)-y_stars_this.reshape(1,-1)).reshape(-1,1)

        xhist,xbins = np.histogram(dx,range=[-max_deviation,max_deviation],bins=2*max_deviation+1)
        yhist,ybins = np.histogram(dy,range=[-max_deviation,max_deviation],bins=2*max_deviation+1)

        idx = np.argmax(xhist)
        xshift = int((xbins[idx]+xbins[idx+1])/2.0)
        idx = np.argmax(yhist)
        yshift = int((ybins[idx]+ybins[idx+1])/2.0)
        #res_list.append([res[2],xshift,yshift,np.sum((lambda1/lambda2)<good_star_threshold)/len(lambda1),len(lambda2)])
        print(xshift,yshift)
        
        arg = (xshift,yshift,PID,template_img_id,len(objects),target_img_pid)
        mycursor = self.sql_interface.cnx.cursor()
        sql = "UPDATE img SET img.x_to_template = %s, img.y_to_template = %s, img.align_process_id = %s, img.align_target_image_id = %s, img.n_star_resolved = %s where img.birth_process_id = %s;"
        mycursor.execute(sql,arg)
        self.dl.cnx.commit()
        return 1
    
    def stacking(self,PID,site_id,method = "mean",PID_type = "birth",ret='success',par = {}):

        def nanaverage(A,weights,axis):
            w = weights.reshape(A.shape[0],1,1)
            return np.nansum(A*w,axis=axis)/((~np.isnan(A))*w).sum(axis=axis)
        
        res_query = self.get_dep_img(PID,process_type=PID_type)
        path_first_file,name_first_file = self.fs.get_dir_for_object("img",{"image_id":res_query[0]['image_id']})
        header0 = fits.getheader(f"{path_first_file}/{name_first_file}")
        res_dict = np.empty((len(res_query),*(fits.getdata(f"{path_first_file}/{name_first_file}").shape)))#,dtype = np.uint16)
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

            x_to_template = data_line['x_to_template']
            y_to_template = data_line['y_to_template']
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

        args = (np.min(jd_start_list),np.mean(jd_mid_list),np.max(jd_end_list),int(np.sum(n_stack_list)),1,img_type_id,res_query[0]['flat_image_id'],res_query[0]['dark_image_id'],0,0,res_query[0]['obs_id'],new_name,0,res_query[0]['align_target_image_id'],res_query[0]['batch'],site_id,PID)
        #print(self.obs_id)
        mycursor = self.sql_interface.cnx.cursor()
        sql = "INSERT INTO img (jd_utc_start,jd_utc_mid,jd_utc_end,n_stack,processed,image_type_id,flat_image_id,dark_image_id,x_to_template,y_to_template,obs_id,img_name,deleted,align_target_image_id,batch,store_site_id,birth_process_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        mycursor.execute(sql,args)
        self.sql_interface.cnx.commit()

        mycursor = self.sql_interface.cnx.cursor()
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
            self.sql_interface.cnx.commit()
        if ret=="success":
            return 1
        if ret=="new_id":
            return new_img_id
    
    def calibration(self,process_PID,site_id,cal_img_pid, sub_img_pid = -1, div_img_pid = -1,subtract_bkg = True,debug = True):#,obs_id=1,hierarchy=2,img_type = "flat_raw",consider_flat = True,consider_bias = True,bias_id = 2,bias_hierarchy = 2,flat_id = 1,flat_hierarchy = 2,keep_origin = True,outpath="samedir",debug = False):
        # img_2_cal = self.get_dep_img(process_PID)
        # # mycursor = self.sql_interface.cnx.cursor()
        # # sql = "SELECT * from img where img.obs_id = %s AND img.image_type_id = %s AND !deleted AND hierarchy = %s;"
        # # args = (obs_id,self.image_type_id[img_type],hierarchy)
        # # mycursor.execute(sql,args)
        # # myresult_corr = mycursor.fetchall()
        # if len(img_2_cal)<1:
        #     print("cannot find target")
        #     return 0
        # print(img_2_cal)
        sql = "SELECT * FROM img WHERE birth_process_id=%s;"
        args = (cal_img_pid,)
        result = self.sql_interface.query(sql,args)
        assert len(result)==1
        img_target = result.to_dict("records")[0]
        sql = "SELECT * FROM img WHERE birth_process_id=%s;"
        sub_img_id = -1
        div_img_id = -1
        if sub_img_pid>0:
            sub_file_path,sub_file_name = self.fs.get_dir_for_object("img",{"birth_pid":sub_img_pid})
            sub_img = fits.getdata(f"{sub_file_path}/{sub_file_name}")
            args = (sub_img_pid,)
            result = self.sql_interface.query(sql,args)
            assert len(result)==1
            sub_img_id = result.to_dict("records")[0]['image_id']

        if div_img_pid>0:
            div_file_path,div_file_name = self.fs.get_dir_for_object("img",{"birth_pid":div_img_pid})
            div_img = fits.getdata(f"{div_file_path}/{div_file_name}")         
            args = (div_img_pid,)
            result = self.sql_interface.query(sql,args)
            assert len(result)==1
            div_img_id = result.to_dict("records")[0]['image_id'] 


        # for img_target in img_2_cal:
        #image_id,jd_utc_start,jd_utc_mid,jd_utc_end,bjd_tdb_start_approximation,bjd_tdb_mid_approximation,bjd_tdb_end_approximation,n_stack,processed,image_type_id,flat_image_id,bias_image_id,x_to_template,y_to_template,obs_id_this,img_path_this,deleted_this,hierarchy_this = pic
        target_image_path,target_image_name = self.fs.get_dir_for_object("img",{"image_id":img_target["image_id"]})
        calibrated_image = fits.getdata(f"{target_image_path}/{target_image_name}")
        calibrated_image_header = fits.getheader(f"{target_image_path}/{target_image_name}")
        new_target_image_name = f"cal_{process_PID}_{target_image_name}"
        if 1==img_target["image_type_id"]: 
            img_type_id_this = 2
        if 5==img_target["image_type_id"]: 
            img_type_id_this = 6         

        #this_outpath = process_PID
        #print(calibrated_image)
        # if outpath=="samedir":
        #     this_outpath = img_path_this.strip(".fits").strip(".fit")+"_corrected_at_"+str(hash(time.time()))+".fits"
        # else:
        #     this_outpath = outpath+"_corrected_at_"+str(hash(time.time()))+".fits"
        if sub_img_pid>0:
            calibrated_image = calibrated_image-sub_img              
            # if img_type=="flat_raw":
            #     sp = calibrated_image.shape
            #     #print(calibrated_image)
            #     calibrated_image = calibrated_image/np.mean(calibrated_image[int(sp[0]/3):int(sp[0]/3*2),int(sp[1]/3):int(sp[1]/3*2)])
                
        if div_img_pid>0:
            calibrated_image = calibrated_image/div_img
        args = (img_target['jd_utc_start'],img_target['jd_utc_mid'],img_target['jd_utc_end'],img_target['n_stack'],1,img_type_id_this,div_img_id,sub_img_id,0,0,img_target['obs_id'],new_target_image_name,0,-1,img_target['batch'],site_id,process_PID)
        mycursor = self.sql_interface.cnx.cursor()
        sql = "INSERT INTO img (jd_utc_start,jd_utc_mid,jd_utc_end,n_stack,processed,image_type_id,flat_image_id,dark_image_id,x_to_template,y_to_template,obs_id,img_name,deleted,align_target_image_id,batch,store_site_id,birth_process_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        mycursor.execute(sql,args)
        self.sql_interface.cnx.commit()



        mycursor = self.sql_interface.cnx.cursor()
        mycursor.execute("SELECT LAST_INSERT_ID();")
        myresult = mycursor.fetchall()
        new_img_id = myresult[0][0] #auto_increment
        save_image_path,save_image_name = self.fs.get_dir_for_object("img",{"image_id":new_img_id})
        if subtract_bkg:
            #import sep
            calibrated_image = calibrated_image.byteswap().newbyteorder()
            bkg = sep.Background(calibrated_image)
            calibrated_image = calibrated_image-bkg
        fits.writeto(f"{save_image_path}/{save_image_name}",calibrated_image.astype('float32'),header = calibrated_image_header,overwrite=True)

            #args = (jd_utc_start,jd_utc_mid,jd_utc_end,bjd_tdb_start_approximation,bjd_tdb_mid_approximation,bjd_tdb_end_approximation,n_stack,processed,self.image_type_id[res_img_type],flat_image_id_used,bias_image_id_used,x_to_template,y_to_template,obs_id_this,this_outpath,0,hierarchy_this)
            #print(len(args))
            #print(self.obs_id)
            # if not debug:
            #     mycursor = self.sql_interface.cnx.cursor()
            #     sql = "INSERT INTO img (jd_utc_start,jd_utc_mid,jd_utc_end,bjd_tdb_start_approximation,bjd_tdb_mid_approximation,bjd_tdb_end_approximation,n_stack,processed,image_type_id,flat_image_id,bias_image_id,x_to_template,y_to_template,obs_id,img_path,deleted,hierarchy) VALUES (%s"+",%s"*(len(args)-1)+")"
            #     mycursor.execute(sql,args)
            #     self.sql_interface.cnx.commit()
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