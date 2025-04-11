
import mysql.connector
from astropy.io import fits
import glob
import tqdm
import numpy as np
import pandas as pd
from Tianyu_pipeline.pipeline.middleware.consumer_component import consumer_component
from Tianyu_pipeline.pipeline.utils import sql_interface 
import Tianyu_pipeline.pipeline.dev.file_system.file_system  as fs
from astroquery.gaia import Gaia
from astropy.table import Table
import os

class data_loader(consumer_component):
    def __init__(self):
        super().__init__()
        #self.sql_interface = sql_interface.sql_interface()
        #self.consumer.fs = fs.file_system()


    def bind_sky_image(self,PID,is_template = False):
        sky_img_PIDs = self.sql_interface.get_process_dependence(PID)
        for pids in sky_img_PIDs:
            sql = 'SELECT * FROM img WHERE img.birth_process_id = %s;'
            args = (pids,)
            result = self.sql_interface.query(sql,args)
            if len(result!=1):
                continue
            else:
                img_info_dict = result.to_dict('records')[0]
                break

        for pids in sky_img_PIDs:
            sql = 'SELECT * FROM sky WHERE img.process_id = %s;'   
            args = (pids,) 
            if len(result!=1):
                continue
            else:
                sky_info_dict = result.to_dict('records')[0]
        mycursor = self.sql_interface.cnx.cursor()
        if is_template:
            args = (sky_info_dict['sky_id'],)
            sql = 'UPDATE sky_image_map SET template_in_use=0 WHERE sky_id=%s;'
            mycursor.execute(sql,args)
            args = (sky_info_dict['sky_id'],img_info_dict['image_id'])
            sql = 'INSERT INTO sky_image_map (sky_id,image_id,template_in_use) VALUES (%s,%s,1);'
            mycursor.execute(sql,args)
        else:
            args = (sky_info_dict['sky_id'],img_info_dict['image_id'])
            sql = 'INSERT INTO sky_image_map (sky_id,image_id,template_in_use) VALUES (%s,%s,0);'
            mycursor.execute(sql,args)

        self.sql_interface.cnx.commit()
        return 1

    def register(self,PID,cmd,par):

        mycursor = self.sql_interface.cnx.cursor()
        sql = cmd
        mycursor.execute(sql,par)
        self.sql_interface.cnx.commit()
        if sql.split(' ')[2]=='img' or sql.split(' ')[2]=='observation' or sql.split(' ')[2]=='sky':#insert PID
            mycursor = self.sql_interface.cnx.cursor()
            mycursor.execute("SELECT LAST_INSERT_ID();")
            myresult = mycursor.fetchall()
            img_id = myresult[0][0] #auto_increment
            if sql.split(' ')[2]=='img':
                sql = 'UPDATE img SET birth_process_id=%s where image_id=%s;'
            if sql.split(' ')[2]=='observation':
                sql = 'UPDATE observation SET process_id=%s where obs_id=%s;'
            if sql.split(' ')[2]=='sky':
                sql = 'UPDATE sky SET process_id=%s where sky_id=%s;'                
            args = (PID,img_id)
            mycursor.execute(sql,args)
            self.sql_interface.cnx.commit()
        return 1

    
    def read_file(self,filename):
        with open(filename, 'rb') as f:
            file_data = f.read()
            return file_data

    def get_table_dict(self,table,index_key=1,index_value=0):
        mycursor = self.sql_interface.cnx.cursor()
        mycursor.execute("SELECT * from "+table+";")
        myresult = mycursor.fetchall()
        # print(myresult)
        res_dict = {}
        for row in myresult:
            res_dict[row[index_key]] = row[index_value]
        return res_dict
    
    def query(self,sql,args,return_df = True):
        mycursor = self.sql_interface.cnx.cursor()
        mycursor.execute(sql,args)
        myresult = mycursor.fetchall()
        headers = [i[0] for i in mycursor.description]
        if return_df:
            df = pd.DataFrame(myresult)
            df.columns = headers
            return df
        return myresult,headers

    def new_observation(self,img_dir,info = {"observation_type":"science","target":"flat","instrument":"L350+QHY600m","obs_site":"TDLI_MGO","observer":"Yicheng Rui"}):
        
        file_path = sorted(glob.glob(img_dir))
        #print(file_path)

        n_pic = len(file_path)


        mycursor = self.sql_interface.cnx.cursor()
        sql = "INSERT INTO observation (observation_type_id,target_id,n_pic,instrument_id,obs_site_id,observer_id) "+"values ("+str(self.observation_type_id[info["observation_type"]])+","+str(self.target_id[info["target"]])+","+str(n_pic)+","+str(self.instrument_id[info["instrument"]])+","+str(self.site_id[info["obs_site"]])+","+str(self.observer_id[info["observer"]])+")"
        #print(sql)
        mycursor.execute(sql)
        self.sql_interface.cnx.commit()

        mycursor = self.sql_interface.cnx.cursor()
        mycursor.execute("SELECT LAST_INSERT_ID();")
        myresult = mycursor.fetchall()
        self.obs_id = myresult[0][0] #auto_increment
        #print(mycursor.rowcount, "record inserted")
        #self.obs_id = 1

        # mycursor = self.cnx.cursor()
        # mycursor.execute("SELECT * from gdr3 where source_id="+gid_str+';')
        # myresult = mycursor.fetchall()
        print('Created observation id=',self.obs_id)

    def load_img_from_fit(self,img_dir,hierarchy = 1,info = {'image_type':'flat_raw'}):
        file_path = sorted(glob.glob(img_dir))
        mycursor = self.sql_interface.cnx.cursor()
        print('Loading data...')
        for fp in tqdm.tqdm(file_path):

            header = fits.getheader(fp)
            jd_utc_start = header['JD']
            jd_utc_mid = header['JD']+header['EXPOSURE']/3600/24/2
            jd_utc_end = header['JD']+header['EXPOSURE']/3600/24  
            args = (jd_utc_start,jd_utc_mid,jd_utc_end,self.image_type_id[info['image_type']],fp,self.obs_id,hierarchy)
            #print(self.obs_id)
            mycursor = self.sql_interface.cnx.cursor()
            sql = "INSERT INTO img (jd_utc_start,jd_utc_mid,jd_utc_end,image_type_id,img_path,obs_id,hierarchy) VALUES (%s,%s,%s,%s,%s,%s,%s)"
            mycursor.execute(sql,args)
            self.sql_interface.cnx.commit()

    def search_GDR3_by_square(self,ra=180,dec=0,fov=1,Gmag_limit = 17,method = "online",cache = True):
        def coord_region(ra,dec,scan_angle,fov):
            deg2rad = np.pi/180
            fov = fov/2*deg2rad #deg
            phi = ra*deg2rad-np.pi #-180 180
            dec_rad = dec * deg2rad
            raw_coord = np.array([[np.cos(fov)**2,np.cos(fov)**2,np.cos(fov)**2,np.cos(fov)**2,np.cos(fov)**2],[np.cos(fov)*np.sin(fov),-np.cos(fov)*np.sin(fov),-np.cos(fov)*np.sin(fov),np.cos(fov)*np.sin(fov),np.cos(fov)*np.sin(fov)],[np.sin(fov),np.sin(fov),-np.sin(fov),-np.sin(fov),np.sin(fov)]])
            scan_angle_rad = scan_angle*deg2rad
            csa = np.cos(scan_angle_rad)
            ssa = np.sin(scan_angle_rad)
            
            cde = np.cos(dec_rad)
            sde = np.sin(dec_rad)
            cal = np.cos(phi)
            sal = np.sin(phi)
            R__x = np.array([[1,0,0],[0,csa,ssa],[0,-ssa,csa]]) 
            R__y = np.array([[cde,0,-sde],[0,1,0],[sde,0,cde]]) 
            R_z = np.array([[cal,-sal,0],[sal,cal,0],[0,0,1]]) 
            after_cart = R_z@R__y@R__x@raw_coord
            after_sphere_theta = np.pi/2-np.arccos(after_cart[2])
            after_sphere_phi = np.sign(after_cart[1])*np.arccos(after_cart[0]/np.sqrt(after_cart[0]**2+after_cart[1]**2))
            return np.array([after_sphere_phi,after_sphere_theta])*180/np.pi

        def generate_sql_command(ra,dec,fov):
            res_coord = coord_region(ra,dec,0,fov)
            #'POLYGON((-136.0 0.0,-135.0 0.0,-135.0 0.35,-136.0 0.35,-136.0 0.0))'
            res = 'POLYGON(('+str(res_coord[0][0])+' '+str(res_coord[1][0])+','+str(res_coord[0][1])+' '+str(res_coord[1][1])+','+str(res_coord[0][2])+' '+str(res_coord[1][2])+','+str(res_coord[0][3])+' '+str(res_coord[1][3])+','+str(res_coord[0][4])+' '+str(res_coord[1][4])+'))'
            return res
        if method == "database":
            polycmd = generate_sql_command(ra,dec,fov)
            sql = "select g3.source_id,g3.ra,g3.`dec`,g3.phot_g_mean_mag,g3.parallax,g3.pmra,g3.pmdec,gv.best_class_name_id,gv.best_score from gaia_gost.gdr3 as g3 LEFT JOIN gdr3_variable as gv on gv.gdr3_id=g3.source_id where ST_Within(pos,ST_SRID(ST_PolyFromText('"+polycmd+"'),4326)) and g3.phot_g_mean_mag<"+str(Gmag_limit)+";"
            mycursor = self.sql_interface.cnx.cursor()
            mycursor.execute(sql)
            myresult = mycursor.fetchall()
            return myresult
        elif method == "online":
            file_name = f"{self.consumer.fs.path_root}/cache/{ra}_{dec}_{fov}_{Gmag_limit}.csv"
            # if exists read from file
            if os.path.exists(file_name):
                return Table.from_pandas(pd.read_csv(file_name))
            
            sql = f'''
            SELECT g3.source_id,g3.ra,g3.dec,g3.phot_g_mean_mag,g3.phot_g_mean_flux_over_error,g3.parallax,g3.pmra,g3.pmdec,gv.in_vari_classification_result,g3.phot_bp_mean_mag,g3.phot_bp_mean_flux_over_error,g3.phot_rp_mean_mag,g3.phot_rp_mean_flux_over_error from gaiadr3.gaia_source as g3 LEFT JOIN gaiadr3.vari_summary as gv on gv.source_id=g3.source_id 
WHERE g3.phot_g_mean_mag<{Gmag_limit} AND
CONTAINS(
    POINT('ICRS',g3.ra,g3.dec),
    CIRCLE('ICRS',{ra},{dec},{fov})
)=1'''      
            job = Gaia.launch_job_async(sql)
            r = job.get_results()
            if cache:
                r.to_pandas().to_csv(file_name,index = False)
            return r
    
    def load_UTC(self,PID):
        picture_birth_PID = self.sql_interface.get_process_dependence(PID)
        for birth_PID in tqdm.tqdm(picture_birth_PID):
            file_path,file_name = self.consumer.fs.get_dir_for_object('img',{'birth_pid':birth_PID})
            header = fits.getheader(f"{file_path}/{file_name}")
            jd_utc_start = header['JD']
            jd_utc_mid = header['JD']+header['EXPOSURE']/3600/24/2
            jd_utc_end = header['JD']+header['EXPOSURE']/3600/24
            sql = "UPDATE img SET jd_utc_start=%s,jd_utc_mid=%s,jd_utc_end=%s WHERE birth_process_id=%s;"
            args = (jd_utc_start,jd_utc_mid,jd_utc_end,birth_PID)
            mycursor = self.sql_interface.cnx.cursor()
            mycursor.execute(sql,args)
            self.sql_interface.cnx.commit()

            



        return 1
if __name__=="__main__":
    dl = data_loader()

    # print(len(dl.search_GDR3_by_square(100,30,0.75,Gmag_limit = 15)))
    # dl.new_observation(img_dir = "/home/share/muguang/image/flat/2024-02-16/*")
    # dl.load_img_from_fit(img_dir = "/home/share/muguang/image/flat/2024-02-16/*")

    # dl.new_observation(img_dir = "/home/share/muguang/image/bias/2024-02-23/*",info = {"observation_type":"science","target":"bias","instrument":"L350+QHY600m","obs_site":"TDLI_MGO","observer":"Yicheng Rui"})
    # dl.load_img_from_fit(img_dir = "/home/share/muguang/image/bias/2024-02-23/*",info = {'image_type':'bias'})

    # dl.new_observation(img_dir = "/home/share/muguang/image/frame/2024-02-16/*",info = {"observation_type":"science","target":"HAT-P-20","instrument":"L350+QHY600m","obs_site":"TDLI_MGO","observer":"Yicheng Rui"})
    # dl.load_img_from_fit(img_dir = "/home/share/muguang/image/frame/2024-02-16/*",info = {'image_type':'science_raw'})
    #dl.new_observation(img_dir = "/home/share/muguang/image/frame/M66/2024-03-02/*",info = {"observation_type":"outreach","target":"M66","instrument":"L350+QHY600m","obs_site":"TDLI_MGO","observer":"Yicheng Rui"})
    #dl.load_img_from_fit(img_dir = "/home/share/muguang/image/frame/M66/2024-03-02/*",info = {'image_type':'deep_raw'})
    dl.new_observation(img_dir = "/home/share/muguang/image/frame/M81/2024-03-11l/*",info = {"observation_type":"outreach","target":"M81","instrument":"L350+QHY600m","obs_site":"TDLI_MGO","observer":"Yicheng Rui"})
    dl.load_img_from_fit(img_dir = "/home/share/muguang/image/frame/M81/2024-03-11l/*",info = {'image_type':'deep_raw'})
    #res = np.array(dl.search_GDR3_by_square(ra=66.75,dec=15.866666666666667,fov=5.5),dtype = np.float64)
    #print(res[:,4])
    #print(len(res[res[:,4]>15]))
    #print(res[res[:,4]>15])