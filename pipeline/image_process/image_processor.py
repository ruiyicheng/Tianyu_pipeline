import pathlib
import mysql.connector
import numpy as np
from astropy.io import fits
import glob
from tqdm import tqdm
import time
import sep
from astropy.stats import sigma_clip 
from astropy.wcs import WCS
import cv2
import numpy.ma as ma
from skimage.morphology import binary_dilation, disk
from ccdproc import cosmicray_lacosmic
import subprocess
from Tianyu_pipeline.pipeline.middleware.consumer_component import consumer_component
import Tianyu_pipeline.pipeline.utils.sql_interface as sql_interface
import Tianyu_pipeline.pipeline.utils.data_loader as data_loader
import Tianyu_pipeline.pipeline.dev.file_system.file_system as fs
import Tianyu_pipeline.pipeline.dev.process_management.process_publisher as process_pub 
import Tianyu_pipeline.pipeline.utils.process_site_getter as psg

class image_processor(consumer_component):
    def __init__(self):#,site_id=-1,group_id=-1):
        super().__init__()
        # self.psg = psg.process_site_getter()
        # self.site_info = self.psg.get_channel()

        # if site_id==-1:
        #     site_id = self.site_info['site_id']
        # if group_id==-1:
        #     group_id = site_id
        #self.dl = data_loader.data_loader()
        # self.consumer.fs = fs.file_system()
        #self.sql_interface = sql_interface.sql_interface()
        #self.pp_this_site = process_pub.process_publisher(site_id = site_id, group_id = group_id)
    def get_dep_img(self,PID,process_type = "birth"):
        res_query = []
        PID_stacker = self.sql_interface.get_process_dependence(PID)
        print(PID_stacker)
        for PID_stacke_this in PID_stacker:
            # if PID_type=="birth":
            if process_type == "birth":
                sql = "SELECT * FROM img where img.birth_process_id = %s and deleted = 0;"
            if process_type == "align":
                sql = "SELECT * FROM img where img.align_process_id = %s and deleted = 0;"
            args = (PID_stacke_this,)
            result = self.sql_interface.query(sql,args)
            if len(result)>0:
                #print(result)
                res_query.extend(result.to_dict("records"))
        return res_query

    def get_image_WCS(self,image_id,target_id = None,ra = None, dec = None):
        # if there is WCS in the header, use it
        # if not, get the WCS from the database
        
        # Get the path of image

        image_path = self.consumer.fs.get_dir_for_object("img",{"image_id":image_id})
        img_path = f"{image_path[0]}/{image_path[1]}"
        header = fits.getheader(img_path)
        if "CTYPE1" in header and "CTYPE2" in header:
            has_wcs = True
        else:
            has_wcs = False
        if has_wcs:
            return WCS(header)
        else:
            if (target_id is None) and ((ra is None) or (dec is None)):
                raise ValueError("Please provide sky_PID or ra and dec")
            if target_id is not None:
                # get ra, dec from database
                sql = "SELECT ra, `dec` FROM sky WHERE target_id = %s;"
                args = (target_id,)
                result = self.sql_interface.query(sql,args)
                if len(result) == 0:
                    raise ValueError("No sky PID found")
                ra = result.loc[0,'ra']
                dec = result.loc[0,'dec']
            # Use Bertin tools to get WCS
            param = self.consumer.dl.get_img_instrument_param(img_id = image_id)
            arcsec_per_pixel = float((param['x_scale_mum']+param['y_scale_mum'])/2/1000/param['focal_length_mm']*206265)
            header_scamp = self.consumer.Bertin.SCAMP_image(img_path,{"DETECT_MINAREA":5,"DETECT_THRESH":5},ra_deg = ra,dec_deg = dec, arcsec_per_pixel = arcsec_per_pixel)
            return WCS(header_scamp)


    def extract_flux(self, PID,image_PID, template_source_PID, n_margin_pixels = 30, show = False):
        # 获取图像数据
        sql = "SELECT * FROM img WHERE birth_process_id = %s;"
        args = (image_PID,)
        img_this = self.sql_interface.query(sql, args)
        
        sql = "SELECT tsp.source_id as source_id, tsp.x_template as x_template,tsp.y_template as y_template, sim.image_id as image_id, sim.absolute_deviation_x as deviation_x, sim.absolute_deviation_y as deviation_y FROM tianyu_source_position as tsp INNER JOIN sky_image_map as sim ON sim.image_id =tsp.template_img_id WHERE sim.process_id = %s"
        args = (template_source_PID,)
        star_template = self.sql_interface.query(sql,args) 
        print(star_template)
        
        assert len(img_this) == 1
        img_id_this = int(img_this.loc[0,'image_id'])
        path_first_file, name_first_file = self.consumer.fs.get_dir_for_object("img", {"image_id": img_id_this})
        img_path = f"{path_first_file}/{name_first_file}"
        img_data = fits.getdata(img_path)
        max_y_img = img_data.shape[0]
        max_x_img = img_data.shape[1]
        # 获取背景
        print(img_data.shape)
        bkg = sep.Background(img_data.astype(np.float32))
        bkgrms = bkg.rms()
        print(img_this)
        x_in_template = np.array(star_template['x_template'].values)-np.array(star_template['deviation_x'])-int(img_this.loc[0,'x_to_template'])
        y_in_template = np.array(star_template['y_template'].values)-np.array(star_template['deviation_y'])-int(img_this.loc[0,'y_to_template'])

        # 10 20 30 can be decided by ml model
        img_data = img_data.byteswap().newbyteorder()
        flux, fluxerr, flag = sep.sum_circle(img_data, x_in_template,y_in_template,10.0, err=bkg,bkgann = (20,30))


        if show:
            from matplotlib.patches import Ellipse
            import matplotlib.pyplot as plt
            # plot background-subtracted image
            fig, ax = plt.subplots()
            m, s = np.mean(img_data), np.std(img_data)
            im = ax.imshow(img_data, interpolation='nearest', cmap='gray',
                        vmin=m-3*s, vmax=m+3*s, origin='lower')

            # plot an ellipse for each object
            # for i in range(len(x_in_template)):
            #     e = Ellipse(xy=(x_in_template[i], y_in_template[i]),
            #                 width=10+1.5*flux[i],
            #                 height=10+1.5*flux[i],
            #                 angle=0)
            #     e.set_facecolor('none')
            #     e.set_edgecolor('red')
            #     ax.add_artist(e)
            i= 1922
            print(f'position={np.array(star_template['x_template'].values)[i]}  {np.array(star_template['y_template'].values)[i]}')
            e = Ellipse(xy=(x_in_template[i], y_in_template[i]),
                        width=50*flux[i],
                        height=50*flux[i],
                        angle=0)
            e.set_facecolor('none')
            e.set_edgecolor('green')
            ax.add_artist(e)
            plt.show()
            print(star_template)
            print(img_this)
        
        
        # writing the flux into db not consider marginal results
        # sql = "SELECT * FROM star_pixel_img where image_id = %s;"
        # args = (int(star_template.loc[0,'image_id']),) 
        # results = self.sql_interface.query(sql,args)
        # print(results)
        # results_source = set([])
        # if len(results)>0:
        #     results_source = results['source_id']
        sql = "INSERT INTO star_pixel_img (image_id, source_id, flux_raw, flux_raw_error, birth_process_id) VALUES (%s,%s,%s,%s,%s);"
        args = []

        for i in tqdm(range(len(x_in_template))):
            if x_in_template[i]>max_x_img-n_margin_pixels or x_in_template[i]<n_margin_pixels or y_in_template[i]>max_y_img-n_margin_pixels or y_in_template[i]<n_margin_pixels:
                continue

            args.append( (int(img_id_this),int(star_template.loc[i,'source_id']), flux[i],fluxerr[i],PID))

        print(len(sql.split('%s')))

        self.sql_interface.executemany(sql,args)
        #print(args)

        return 1


    def detect_source_in_template(self,PID,sky_id,resolve_sigma = 3,minarea = 20,max_distance = 5,max_ratio = 2,debug = False,as_new_template = True):
        # 1. get the template image
        # 2. get the sources in template image
        # 3. get the sources in target image
        # 4. match the sources in template image and target image
        # 5. update the database
        # Optional: do similar procedure on differential image
        def find_nearest_kdtree(x1, y1, x2, y2):
            # input: x1, y1, x2, y2
            # output: nearest_indices, distances
            # x1.shape = y1.shape
            # x2.shape = y2.shape
            # output.shape = x1.shape
            # function: find the nearest point in x2, y2 for each point in x1, y1
            # used for matching the sources in template image and target image
            # 构建KDTree
            from scipy.spatial import cKDTree
            tree = cKDTree(np.c_[x2, y2])
            
            # 查找最近点的索引和距离
            distances, nearest_indices = tree.query(np.c_[x1, y1], k=1)
            
            return nearest_indices, distances
        img_this = self.get_dep_img(PID)
        assert len(img_this)==1
        img_id_this = img_this[0]['image_id']
        path_first_file,name_first_file = self.consumer.fs.get_dir_for_object("img",{"image_id":img_id_this})
        img_path = f"{path_first_file}/{name_first_file}"
        img_data = fits.getdata(img_path).byteswap().newbyteorder()
        bkg = sep.Background(img_data)
        data_sub = img_data-bkg
        objects = sep.extract(data_sub,resolve_sigma,err=bkg.rms(),minarea=minarea)
        # 过滤掉长宽比大于max_ratio的目标
        objects = objects[objects['a']/objects['b']<max_ratio]
        print(f"detected {len(objects)} objects")
        # photometry
        flux, fluxerr, flag = sep.sum_circle(data_sub, objects['x'], objects['y'],
                                     10.0, err=bkg.globalrms, gain=1.0)
        # 查询数据库记录, 现版本没管差分图像
        sql = """
SELECT 
    ts.source_id as source_id, 
    tsp.template_img_id as image_id, 
    tsp.x_template as x_template, 
    tsp.y_template as y_template, 
    sim.absolute_deviation_x as absolute_deviation_x, 
    sim.absolute_deviation_y as absolute_deviation_y 
FROM 
    tianyu_source AS ts 
INNER JOIN 
    (SELECT source_id, MAX(template_img_id) as max_template_img_id
     FROM tianyu_source_position
     GROUP BY source_id) as max_tsp ON ts.source_id = max_tsp.source_id
INNER JOIN 
    tianyu_source_position AS tsp ON ts.source_id = tsp.source_id 
    AND tsp.template_img_id = max_tsp.max_template_img_id
INNER JOIN 
    sky_image_map AS sim ON tsp.template_img_id = sim.image_id 
WHERE 
    sim.sky_id = %s 
ORDER BY 
    source_id, image_id;
"""
        args = (sky_id,)
        archive_star_result = self.sql_interface.query(sql, args)
        
        if len(archive_star_result) == 0:
            print("no template yet")
            absolute_deviation_x = 0
            absolute_deviation_y = 0
            matched_indices = []
            x_star_this = objects['x']
            y_star_this = objects['y']
            x_star_this_absolute = x_star_this + absolute_deviation_x
            y_star_this_absolute = y_star_this + absolute_deviation_y
        else:
            # 从数据库结果中提取坐标
            x_template = archive_star_result['x_template'].values
            y_template = archive_star_result['y_template'].values
            
            x_star_this = objects['x']
            y_star_this = objects['y']

            absolute_deviation_x, absolute_deviation_y = self.get_shift(x_template,y_template,x_star_this,y_star_this)

            x_star_this_absolute = x_star_this + absolute_deviation_x
            y_star_this_absolute = y_star_this + absolute_deviation_y
            # 交叉匹配
            max_distance = max_distance  # 最大匹配距离(像素)
            matched_indices = []
            nearest_indices, distances = find_nearest_kdtree(x_star_this_absolute,y_star_this_absolute,x_template,y_template)

            for ind_obj,ind_archive, dis in zip(range(len(nearest_indices)),nearest_indices,distances):
                if dis < max_distance:
                    matched_indices.append((ind_obj,ind_archive))
        if not debug:
                # 更新数据库
            if as_new_template:
                sql = "UPDATE sky_image_map SET template_in_use=%s;"
                args = (0,)
                self.sql_interface.execute(sql, args)
                
            sql = "INSERT INTO sky_image_map (sky_id,image_id,template_in_use,absolute_deviation_x,absolute_deviation_y,process_id) VALUES (%s,%s,%s,%s,%s,%s)"
            args = (sky_id,img_id_this,int(as_new_template),absolute_deviation_x,absolute_deviation_y,PID)
            self.sql_interface.execute(sql, args)
            for ind_obj,ind_archive in matched_indices:
                sql = "SELECT * FROM tianyu_source_position WHERE source_id = %s AND template_img_id = %s;"
                args = (int(archive_star_result['source_id'][ind_archive]), int(img_id_this))
                result = self.sql_interface.query(sql,args)
                if len(result)==0:
                    sql = """
                    INSERT INTO tianyu_source_position (source_id, template_img_id, x_template, y_template,flux_template,e_flux_template) 
                    VALUES (%s,%s,%s,%s,%s,%s);
                    """
                    args = (int(archive_star_result['source_id'][ind_archive]), int(img_id_this),float(x_star_this_absolute[ind_obj]), float(y_star_this_absolute[ind_obj]),float(flux[ind_obj]),float(fluxerr[ind_obj]))
                    self.sql_interface.execute(sql, args)
            
            # 插入新检测到的源
            new_sources = set(range(len(objects))) - set([m[0] for m in matched_indices])
            for i in new_sources:
                
                # 首先插入新源到 tianyu_source 表
                sql_source = "INSERT INTO tianyu_source (sky_id) VALUES (%s)"
                args_source = (sky_id,)
                new_source_id = self.sql_interface.execute(sql_source, args_source, return_last_id=True)
                
                # 然后插入新源的位置信息到 tianyu_source_position 表
                sql_position = "INSERT INTO tianyu_source_position (source_id, template_img_id, x_template, y_template, flux_template, e_flux_template) VALUES (%s, %s, %s, %s, %s, %s)"
                args_position = (new_source_id, img_id_this, x_star_this_absolute[i], y_star_this_absolute[i], flux[i], fluxerr[i])
                self.sql_interface.execute(sql_position, args_position)
                #self.sql_interface.execute(sql, args)
            
            print(f"crossmatched {len(matched_indices)} sources,  {len(new_sources)} new sources")
        else:
            self.show_source_img(data_sub,objects)
        return 1

    def get_shift(self,x_stars_template,y_stars_template,x_stars_this,y_stars_this,max_deviation = 400):
        dx = (x_stars_template.reshape(-1,1)-x_stars_this.reshape(1,-1)).reshape(-1,1)
        dy = (y_stars_template.reshape(-1,1)-y_stars_this.reshape(1,-1)).reshape(-1,1)
        xhist,xbins = np.histogram(dx,range=[-max_deviation,max_deviation],bins=2*max_deviation+1)
        yhist,ybins = np.histogram(dy,range=[-max_deviation,max_deviation],bins=2*max_deviation+1)

        idx = np.argmax(xhist)
        xshift = int((xbins[idx]+xbins[idx+1])/2.0)
        idx = np.argmax(yhist)
        yshift = int((ybins[idx]+ybins[idx+1])/2.0)
        return xshift,yshift
    def show_source_img(self,img_data,objects):
        from matplotlib.patches import Ellipse
        import matplotlib.pyplot as plt
        # plot background-subtracted image
        fig, ax = plt.subplots()
        m, s = np.mean(img_data), np.std(img_data)
        im = ax.imshow(img_data, interpolation='nearest', cmap='gray',
                    vmin=m-0.3*s, vmax=m+0.5*s, origin='lower')

        # plot an ellipse for each object
        for i in range(len(objects)):
            e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                        width=6*objects['a'][i],
                        height=6*objects['b'][i],
                        angle=objects['theta'][i] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax.add_artist(e)
        plt.show()
    def alignment(self,PID,template_img_pid,target_img_pid,max_deviation = 400,resolve_sigma = 5,minarea = 25,test = False):
        # Align 2 image! 2 possibility
        # 1 template_img not resolved, need to resolve the target in template_img
        # 2 template_img resolved, need to get star position from db
        
        sql = "SELECT tsp.x_template as x_template, tsp.y_template as y_template FROM img INNER JOIN tianyu_source_position as tsp on img.image_id = tsp.template_img_id WHERE img.birth_process_id = %s"
        args = (template_img_pid,)
        result = self.sql_interface.query(sql,args)
        if len(result)>=10:# resolved template image
            print("Image resolved")
            x_stars_template = np.array(result['x_template'])
            y_stars_template = np.array(result['y_template'])
        else: # not resolved image
            print("Resolving stars in template image using sextractor...")
            img_folder,img_name = self.consumer.fs.get_dir_for_object("img",{"birth_pid":template_img_pid})
            img_path = f"{img_folder}/{img_name}"
            img_data = fits.getdata(img_path).byteswap().newbyteorder()
            #img_bkg_rms = fits.getdata(img_path).byteswap().newbyteorder()
            bkg = sep.Background(img_data)
            objects = sep.extract(img_data-bkg,resolve_sigma,err=bkg.rms(),minarea=minarea)
            if False:#Used for debug, show the sep resolve results
                self.show_source_img(img_data,objects)

            if len(objects['x'])<3:
                print('Too few star resolved in template image')
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
        img_folder,img_name = self.consumer.fs.get_dir_for_object("img",{"birth_pid":target_img_pid})
        img_path = f'{img_folder}/{img_name}'
        img_data = fits.getdata(img_path).byteswap().newbyteorder()
        bkg = sep.Background(img_data)
        #bkg_rms_this = bkg.globalrms
        try:
            objects = sep.extract(img_data-bkg,resolve_sigma,err=bkg.rms(),minarea=minarea)
            x_stars_this = np.squeeze(objects['x'])
            y_stars_this = np.squeeze(objects['y'])

            xshift,yshift = self.get_shift(x_stars_template,y_stars_template,x_stars_this,y_stars_this,max_deviation = max_deviation)
            n_star_this = len(objects)
        except:
            xshift,yshift = 0,0
            n_star_this = 0
            bkg_rms_this = 10000000
        if len(objects['x'])<3:
            print("Failed to extract stars in target image, marking img")
            arg = (template_img_pid,)
            #mycursor = self.sql_interface.cnx.cursor()
            sql = "UPDATE img SET img.deleted = 1 WHERE img.birth_process_id = %s;"
            # 删除文件系统中的图像
            # try:
            #     os.remove(img_path)
            #     print(f"success to delete: {img_path}")
            # except OSError as e:
            #     print(f"failed to delete: {e}")

            #mycursor.execute(sql,arg)
            #self.sql_interface.cnx.commit()
            self.sql_interface.execute(sql,arg)
            return 1

        print("Alignment result:",xshift,yshift)
        arg = (xshift,yshift,PID,template_img_id,n_star_this,target_img_pid)
        sql = "UPDATE img SET img.x_to_template = %s, img.y_to_template = %s, img.align_process_id = %s, img.align_target_image_id = %s, img.n_star_resolved = %s where img.birth_process_id = %s;"
        self.sql_interface.execute(sql,arg)
        return 1
    #method = mean, median, flat_stacking (3median-2mean)
    def stacking(self,PID,site_id,method = "mean",PID_type = "birth",ret='success',par = {},consider_goodness = 0):
        
        def nanaverage(A,weights,axis):
            w = weights.reshape(A.shape[0],1,1)
            nume = ((~np.isnan(A))*w).sum(axis=axis)
            #nume[nume<10**-8]= np.nan
            return np.nansum(A*w,axis=axis)/nume
        try:
            #print(self.sql_interface.get_process_dependence(PID))
            #print(PID_type)
            res_query = self.get_dep_img(PID,process_type=PID_type)
            #print(res_query)
            path_first_file,name_first_file = self.consumer.fs.get_dir_for_object("img",{"image_id":res_query[0]['image_id']})
            header0 = fits.getheader(f"{path_first_file}/{name_first_file}")
            inst_info = self.consumer.dl.get_img_instrument_param(img_id = res_query[0]['image_id'])
            id_list = []
            jd_start_list = []
            jd_mid_list = []
            jd_end_list = []
            n_stack_list = []
            if method == "SWARP":
                sky_bkg_list = []
                nstar_list = []
                image_path_list = []
                mask_image_path_list = []
                n_stack_list = []
                for i,data_line in tqdm(enumerate(res_query)):
                    if int(data_line["n_star_resolved"])==0:
                        print("No star resolved in this image")
                        continue
                    id_list.append(int(data_line['image_id']))
                    jd_start_list.append(float(data_line["jd_utc_start"]))
                    jd_mid_list.append(float(data_line["jd_utc_mid"]))
                    jd_end_list.append(float(data_line["jd_utc_end"]))
                    sky_bkg_list.append(float(data_line["bkg_rms"]))
                    n_stack_list.append(float(data_line["n_stack"]))
                    nstar_list.append(int(data_line["n_star_resolved"]))
                    # to ensure the image have WCS. otherwise, exit
                    #self.get_image_WCS(int(data_line['image_id']),target_id = int(inst_info['target_id']))
                    path_first_file,name_first_file = self.consumer.fs.get_dir_for_object("img",{"image_id":int(data_line['image_id'])})
                    image_path = f"{path_first_file}/{name_first_file}"
                    mask_image_id = int(data_line['mask_image_id'])
                    path_first_file_mask,name_first_file_mask = self.consumer.fs.get_dir_for_object("img",{"image_id":mask_image_id})
                    mask_path = f"{path_first_file_mask}/{name_first_file_mask}"
                    image_path_list.append(image_path)
                    mask_image_path_list.append(mask_path)
                # filter the image
                mask_star = sigma_clip(np.array(nstar_list),sigma=3).mask
                # Eliminate the images with aeroplane or other issues
                mask_bkg = sigma_clip(np.array(sky_bkg_list),sigma_lower = 100,sigma_upper = 5).mask
                mask_result = ~(mask_star | mask_bkg)
                jd_start_list = np.array(jd_start_list)[mask_result]
                jd_mid_list = np.array(jd_mid_list)[mask_result]
                jd_end_list = np.array(jd_end_list)[mask_result]
                image_path_list = np.array(image_path_list)[mask_result]
                mask_image_path_list = np.array(mask_image_path_list)[mask_result]
                id_list = np.array(id_list)[mask_result]
                n_stack_list = np.array(n_stack_list)[mask_result]
                img_type_id = res_query[0]['image_type_id']
                new_name = f"n_{int(np.sum(n_stack_list))}_PID_{PID}_from_{name_first_file.split('from_')[-1]}"#.strip('.fits').strip('.fit')
                mask_name = "mask_" + new_name
                # Insert mask
                args = (np.min(jd_start_list),np.mean(jd_mid_list),np.max(jd_end_list),int(np.sum(n_stack_list)),1,img_type_id,res_query[0]['flat_image_id'],res_query[0]['dark_image_id'],0,0,res_query[0]['obs_id'],mask_name,0,res_query[0]['batch'],site_id)
                npar = ("%s,"*len(args))[:-1]
                sql = f"INSERT INTO img (jd_utc_start,jd_utc_mid,jd_utc_end,n_stack,processed,image_type_id,flat_image_id,dark_image_id,x_to_template,y_to_template,obs_id,img_name,deleted,batch,store_site_id) VALUES ({npar})"
                mask_image_id = self.sql_interface.execute(sql,args,return_last_id=True)
                args = (np.min(jd_start_list),np.mean(jd_mid_list),np.max(jd_end_list),int(np.sum(n_stack_list)),1,img_type_id,res_query[0]['flat_image_id'],res_query[0]['dark_image_id'],0,0,res_query[0]['obs_id'],new_name,0,res_query[0]['batch'],site_id,PID,int(mask_image_id))
                npar = ("%s,"*len(args))[:-1]
                sql = f"INSERT INTO img (jd_utc_start,jd_utc_mid,jd_utc_end,n_stack,processed,image_type_id,flat_image_id,dark_image_id,x_to_template,y_to_template,obs_id,img_name,deleted,batch,store_site_id,birth_process_id,mask_image_id) VALUES ({npar})"
                new_img_id = self.sql_interface.execute(sql,args,return_last_id=True)
                new_image_folder,new_image_path = self.consumer.fs.get_dir_for_object("img",{"image_id":new_img_id})

                new_image_path = f"{new_image_folder}/{new_name}"
                mask_folder,mask_image_path = self.consumer.fs.get_dir_for_object("img",{"image_id":mask_image_id})
                success = self.consumer.fs.create_dir_for_object("img",{"image_id":new_img_id})
                mask_image_path = f"{mask_folder}/{mask_name}"
                output_image_file_path, output_weight_file_path = self.consumer.Bertin.SWARP_stack(image_path_list,{},mask_image_path_list)
                success = self.consumer.fs.create_dir_for_object("img",{"image_id":mask_image_id})

                # move output files
                subprocess.run(f'mv {output_image_file_path} {new_image_path}', shell=True)
                subprocess.run(f'mv {output_weight_file_path} {mask_image_path}', shell=True)



                






            else:
                res_dict = np.empty((len(res_query),*(fits.getdata(f"{path_first_file}/{name_first_file}").shape)))#,dtype = np.uint16)
                res_dict[:] = np.nan
                print('importing data...')

                
                goodness_list = []
                for i,data_line in tqdm(enumerate(res_query)):
                    #image_id,jd_utc_start,jd_utc_mid,jd_utc_end,bjd_tdb_start_approximation,bjd_tdb_mid_approximation,bjd_tdb_end_approximation,n_stack,processed,image_type_id,flat_image_id,bias_image_id,x_to_template,y_to_template,obs_id,img_path,deleted_this,hierarchy_this = data_line
                    id_list.append(data_line['image_id'])
                    jd_start_list.append(data_line["jd_utc_start"])
                    jd_mid_list.append(data_line["jd_utc_mid"])
                    jd_end_list.append(data_line["jd_utc_end"])
                    n_stack_list.append(data_line["n_stack"])
                    goodness_list.append(data_line["coadd_weight"])
                    x_to_template = data_line['x_to_template']
                    y_to_template = data_line['y_to_template']
                    if type(x_to_template)==type(None) or type(y_to_template)==type(None):
                        x_to_template=0
                        y_to_template=0
                    img_folder,img_name = self.consumer.fs.get_dir_for_object("img",{"image_id":data_line['image_id']})
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
                if not consider_goodness:
                    weights = np.array(n_stack_list)
                else:
                    weights = np.array(goodness_list)

                if method=="mean":
                    weights_revised = weights.copy()
                    if np.sum(weights)<10**-2:
                        weights_revised = weights+1
                    #print(weights_revised,res_dict)
                    res = nanaverage(res_dict,weights_revised,axis = 0)
                if method == "median":
                    res = np.nanmedian(res_dict,axis = 0)
                if method == "flat_stacking":
                    print('flat stacking!!!')
                    averages_picture = np.mean(res_dict, axis=(1, 2)).reshape(-1,1,1)
                    for i in range(len(res_dict)):
                        res_dict[i] = res_dict[i]/np.mean(res_dict[i])
                    
                    #normalized_picture = res_dict/averages_picture
                    
                    weights = averages_picture #1/sigma^2 proportional to flux due to Poisson noise
                    print('weights = ',weights)
                    mean = 0
                    for i in range(len(weights)):
                        mean += weights[i]*res_dict[i]
                    mean = mean/np.sum(weights)
                    print('taking median')
                    median = np.zeros_like(mean)
                    # Do it line-by-line to save memory
                    for i in tqdm(range(len(median))): 
                        median[i] = np.median(res_dict[:,i,:],axis = 0)
                    # np.median(res_dict,axis = 0)
                    res = 3*median-2*mean
                #print(np.sum(n_stack_list))
                res = res.astype(np.float32)
                new_name = f"n_{int(np.sum(n_stack_list))}_PID_{PID}_from_{name_first_file.split('from_')[-1]}"
                
                if res_query[0]['image_type_id']==2:
                    img_type_id = 3
                else:
                    img_type_id = res_query[0]['image_type_id']

                if not consider_goodness:
                    args = (np.min(jd_start_list),np.mean(jd_mid_list),np.max(jd_end_list),int(np.sum(n_stack_list)),1,img_type_id,res_query[0]['flat_image_id'],res_query[0]['dark_image_id'],0,0,res_query[0]['obs_id'],new_name,0,res_query[0]['align_target_image_id'],res_query[0]['batch'],site_id,PID)
                    sql = "INSERT INTO img (jd_utc_start,jd_utc_mid,jd_utc_end,n_stack,processed,image_type_id,flat_image_id,dark_image_id,x_to_template,y_to_template,obs_id,img_name,deleted,align_target_image_id,batch,store_site_id,birth_process_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                else:
                    args = (np.min(jd_start_list),np.mean(jd_mid_list),np.max(jd_end_list),int(np.sum(n_stack_list)),1,img_type_id,res_query[0]['flat_image_id'],res_query[0]['dark_image_id'],0,0,res_query[0]['obs_id'],new_name,0,res_query[0]['align_target_image_id'],res_query[0]['batch'],site_id,PID,np.sum(goodness_list))
                    sql = "INSERT INTO img (jd_utc_start,jd_utc_mid,jd_utc_end,n_stack,processed,image_type_id,flat_image_id,dark_image_id,x_to_template,y_to_template,obs_id,img_name,deleted,align_target_image_id,batch,store_site_id,birth_process_id,coadd_weight) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                #mycursor = self.sql_interface.cnx.cursor()
                #mycursor.execute(sql,args)
                #self.sql_interface.cnx.commit()
                new_img_id = self.sql_interface.execute(sql,args,return_last_id=True)
                #mycursor = self.sql_interface.cnx.cursor()
                #mycursor.execute("SELECT LAST_INSERT_ID();")
                # myresult = mycursor.fetchall()
                # new_img_id = myresult[0][0] #auto_increment
                path_first_file,name_first_file = self.consumer.fs.get_dir_for_object("img",{"image_id":new_img_id})
                _ = self.consumer.fs.create_dir_for_object("img",{"image_id":new_img_id})
                fits.writeto(f"{path_first_file}/{name_first_file}",res,header = header0,overwrite=True)
            print("recording stacking")
            for i in id_list:
                args = (new_img_id,i)
                sql = "INSERT INTO img_stacking (image_id,stacked_id) VALUES (%s,%s)"
                #mycursor.execute(sql,args)
                self.sql_interface.execute(sql,args)
            if ret=="success":
                return 1
            if ret=="new_id":
                return new_img_id
        except Exception as e:
            print(e)
            return 0
    def select_good_img(self,PID):
        res_query = self.get_dep_img(PID,process_type='align')

        PIDs = []
        n_stars = []
        bkg_rms_list = []
        for single_img in res_query:
            PIDs.append(single_img['birth_process_id'])
            n_stars.append(single_img['n_star_resolved'])
            bkg_rms_list.append(single_img['bkg_rms'])
        # Eliminate the images with cloud or tracking issue
        mask_star = sigma_clip(np.array(n_stars),sigma=3).mask
        # Eliminate the images with aeroplane or other issues
        mask_bkg = sigma_clip(np.array(bkg_rms_list),sigma_lower = 100,sigma_upper = 5).mask
        mask_result = ~(mask_star | mask_bkg)
        for i in range(len(PIDs)):
        
            PID_this,mask_this = PIDs[i],mask_result[i]
            mycursor = self.sql_interface.cnx.cursor()
            sql = "UPDATE img SET img.coadd_weight = %s WHERE img.birth_process_id = %s;"
            arg = (1 if mask_this else 0,PID_this)
            mycursor.execute(sql,arg)
            self.sql_interface.cnx.commit()
        return 1
        
        
    def calibration(self,process_PID,site_id,cal_img_pid, sub_img_pid = -1, div_img_pid = -1,subtract_bkg = False,debug = True):#,obs_id=1,hierarchy=2,img_type = "flat_raw",consider_flat = True,consider_bias = True,bias_id = 2,bias_hierarchy = 2,flat_id = 1,flat_hierarchy = 2,keep_origin = True,outpath="samedir",debug = False):
        # Calibration of the image (debias/flat calibration or both)
        # Mask the bad pixels

        sql = "SELECT * FROM img WHERE birth_process_id=%s;"
        args = (cal_img_pid,)
        result = self.sql_interface.query(sql,args)
        assert len(result)==1
        img_target = result.to_dict("records")[0]
        sql = "SELECT * FROM img WHERE birth_process_id=%s;"
        sub_img_id = None
        div_img_id = None
        if sub_img_pid>0:
            sub_file_path,sub_file_name = self.consumer.fs.get_dir_for_object("img",{"birth_pid":sub_img_pid})
            sub_img = fits.getdata(f"{sub_file_path}/{sub_file_name}")
            args = (sub_img_pid,)
            result = self.sql_interface.query(sql,args)
            assert len(result)==1
            sub_img_id = result.to_dict("records")[0]['image_id']

        if div_img_pid>0:
            div_file_path,div_file_name = self.consumer.fs.get_dir_for_object("img",{"birth_pid":div_img_pid})
            div_img = fits.getdata(f"{div_file_path}/{div_file_name}")         
            args = (div_img_pid,)
            result = self.sql_interface.query(sql,args)
            assert len(result)==1
            div_img_id = result.to_dict("records")[0]['image_id'] 


        # for img_target in img_2_cal:
        #image_id,jd_utc_start,jd_utc_mid,jd_utc_end,bjd_tdb_start_approximation,bjd_tdb_mid_approximation,bjd_tdb_end_approximation,n_stack,processed,image_type_id,flat_image_id,bias_image_id,x_to_template,y_to_template,obs_id_this,img_path_this,deleted_this,hierarchy_this = pic
        target_image_path,target_image_name = self.consumer.fs.get_dir_for_object("img",{"image_id":img_target["image_id"]})
        calibrated_image = fits.getdata(f"{target_image_path}/{target_image_name}")
        calibrated_image_header = fits.getheader(f"{target_image_path}/{target_image_name}")
        new_target_image_name = f"cal_{process_PID}_{target_image_name}"
        img_type_id_this = img_target["image_type_id"]
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
        calibrated_image_ret = calibrated_image.copy()
        if sub_img_pid>0:
            calibrated_image_ret = calibrated_image_ret-sub_img              
            # if img_type=="flat_raw":
            #     sp = calibrated_image.shape
            #     #print(calibrated_image)
            #     calibrated_image = calibrated_image/np.mean(calibrated_image[int(sp[0]/3):int(sp[0]/3*2),int(sp[1]/3):int(sp[1]/3*2)])
                
        if div_img_pid>0:
            calibrated_image_ret = calibrated_image_ret/div_img
        args = (img_target['jd_utc_start'],img_target['jd_utc_mid'],img_target['jd_utc_end'],img_target['n_stack'],1,img_type_id_this,div_img_id,sub_img_id,0,0,img_target['obs_id'],new_target_image_name,0,None,img_target['batch'],site_id,process_PID)
        #mycursor = self.sql_interface.cnx.cursor()
        sql = "INSERT INTO img (jd_utc_start,jd_utc_mid,jd_utc_end,n_stack,processed,image_type_id,flat_image_id,dark_image_id,x_to_template,y_to_template,obs_id,img_name,deleted,align_target_image_id,batch,store_site_id,birth_process_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        #mycursor.execute(sql,args)
        #self.sql_interface.cnx.commit()
        new_img_id = self.sql_interface.execute(sql,args,return_last_id=True)



        # mycursor = self.sql_interface.cnx.cursor()
        # mycursor.execute("SELECT LAST_INSERT_ID();")
        # myresult = mycursor.fetchall()
        # new_img_id = myresult[0][0] #auto_increment
        save_image_path,save_image_name = self.consumer.fs.get_dir_for_object("img",{"image_id":new_img_id})
        full_img_path = f"{save_image_path}/{save_image_name}"
        success = self.consumer.fs.create_dir_for_object("img",{"image_id":new_img_id})
        if sub_img_pid > 0 and div_img_pid > 0:
            # full calibration for sky fields
            bkg = sep.Background(calibrated_image_ret.astype("float32"))
            calibrated_image_debkg = calibrated_image_ret.copy()-bkg
            if subtract_bkg:
                calibrated_image_ret = calibrated_image_ret-bkg

            bkg_rms = bkg.globalrms

            # 将背景噪声 RMS 值记录到数据库中
            new_target_image_name_mask = "mask_"+new_target_image_name
            args = (img_target['jd_utc_start'],img_target['jd_utc_mid'],img_target['jd_utc_end'],img_target['n_stack'],1,img_type_id_this,div_img_id,sub_img_id,0,0,img_target['obs_id'],new_target_image_name_mask,0,None,img_target['batch'],site_id)
            #mycursor = self.sql_interface.cnx.cursor()
            sql = "INSERT INTO img (jd_utc_start,jd_utc_mid,jd_utc_end,n_stack,processed,image_type_id,flat_image_id,dark_image_id,x_to_template,y_to_template,obs_id,img_name,deleted,align_target_image_id,batch,store_site_id,is_mask) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1)"
            #mycursor.execute(sql,args)
            #self.sql_interface.cnx.commit()
            new_img_mask_id = self.sql_interface.execute(sql,args,return_last_id=True)
            print(f"bkg RMS = {bkg_rms} recorded at image_id: {new_img_id}")
            # Get mask for saturated pixels, Cosmic ray, hot pixels and airplane/satellite tracks

            mask_save_image_path,mask_save_image_name = self.consumer.fs.get_dir_for_object("img",{"image_id":new_img_mask_id})
            mask_full_img_path = f"{mask_save_image_path}/{mask_save_image_name}"
            
            success = self.consumer.fs.create_dir_for_object("img",{"image_id":new_img_mask_id})

            inst_info = self.consumer.dl.get_img_instrument_param(obs_id = int(img_target['obs_id']))
            mask,n_air = self.get_mask(calibrated_image.astype("float32"),Nbit = int(inst_info['n_bit']),sat_level = 0.9,gain = float(inst_info['gain']),readnoise = float(inst_info['readout_noise_e']))
            mask = ~mask

            fits.writeto(mask_full_img_path,mask.astype('float32'),overwrite=True)
            if n_air==0:
                star_resolve = sep.extract(calibrated_image_debkg.astype("float32"),5,err=bkg.globalrms,minarea=5)
                n_stars = len(star_resolve)
            else:
                n_stars = 0
            
            sql = "UPDATE img SET bkg_rms = %s,n_star_resolved = %s,mask_image_id = %s  WHERE image_id = %s"
            args = (float(bkg_rms), int(n_stars),new_img_mask_id,new_img_id)
            self.sql_interface.execute(sql, args)
            #fits.writeto(f"{save_image_path}/{save_image_name.strip('.fits').strip('.fit')+"_bkgrms.fits"}",bkg_rms.astype('float32'),header = calibrated_image_header,overwrite=True)


        
        fits.writeto(full_img_path,calibrated_image_ret.astype('float32'),header = calibrated_image_header,overwrite=True)
        if sub_img_pid > 0 and div_img_pid > 0:
            if n_air==0:
                wcs = self.get_image_WCS(new_img_id,target_id = int(inst_info['target_id']))
                print(wcs)
        return success
    def get_mask(self,image,Nbit = 16,sat_level = 0.9,gain = 1.3,readnoise = 1,dilation_radius = 5,saturate = False,show = False):
        # Get the masks for saturated pixels, Cosmic ray, hot pixels and airplane/satellite tracks
        # saturate
        mask_saturate = image > (2 ** Nbit * sat_level)
        cr_params = {
            'sigclip': 4.5,
            'objlim': 5.0,
            'gain': gain,      # e-/ADU - CHECK YOUR HEADER!
            'readnoise': readnoise, # e- - CHECK YOUR HEADER!
            'satlevel': 2 ** Nbit * sat_level # ADU - CHECK YOUR HEADER!
        }
        image_this = image.astype('float32')

        mask_hot_pixel = self.mask_cosmic_rays(image_this, **cr_params)

        bkg = sep.Background(image_this,mask = mask_hot_pixel | mask_saturate)
        bkg_image = bkg.back()
        bkg_rms = bkg.rms()
        image_sub = image - bkg_image
        mask_criteria = image_sub>1.5*bkg_rms
        mask_criteria = (mask_criteria*255).astype('uint8')
        kernel = np.ones((3,3),np.uint8)
        mask_criteria = cv2.erode(mask_criteria,kernel)
        kernel = np.ones((3,3),np.uint8)
        mask_criteria = cv2.dilate(mask_criteria,kernel)
        contours,hierarchy = cv2.findContours(mask_criteria,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


        
        airplane_contour = []
        for cnt in contours:
        
            rect = cv2.minAreaRect(cnt)
            long_axis = np.maximum(rect[1][0],rect[1][1])
            short_axis = np.minimum(rect[1][0],rect[1][1])

            #Used the criteria in LAHER 14 
            if long_axis>800 or long_axis>300 and short_axis<20 or long_axis>150 and short_axis<4:
                airplane_contour.append(cnt)

        print(f"find {len(airplane_contour)} airplanes")
        mask_track = np.zeros(mask_criteria.shape,dtype='uint8')
        mask_track = cv2.drawContours(mask_track,airplane_contour, -1, 255,thickness=cv2.FILLED)

        mask_track = mask_track>1
        struct = disk(dilation_radius) # Or square(2*dilation_radius + 1)
        mask_track = binary_dilation(mask_track, footprint=struct)
        mask =  mask_hot_pixel | mask_track
        if saturate:
            
            mask = mask | mask_saturate
        # compute the mask
        
        print(f'masked {np.sum(mask)} pixels')

        return mask,len(airplane_contour)
    def mask_cosmic_rays(self,data, sigclip=4.5, objlim=5.0, gain=1.0, readnoise=5.0, satlevel=60000.0, verbose=False):
        """
        Creates a mask for cosmic rays using the L.A.Cosmic algorithm.

        Args:
            data (np.ndarray): Input image data.
            sigclip (float): Sigma threshold for cosmic ray detection.
            objlim (float): Contrast limit between cosmic ray and underlying object.
            gain (float): Detector gain (e.g., e-/ADU). Get from header if possible.
            readnoise (float): Detector read noise (e.g., e-). Get from header if possible.
            satlevel (float): Saturation level (ADU). Get from header if possible.
            verbose (bool): Print progress information.

        Returns:
            np.ndarray: Boolean mask where True indicates a cosmic ray pixel.
        """
        print("Masking cosmic rays...")
        # L.A.Cosmic implementation in ccdproc
        # It returns the cleaned image and the mask
        # We need gain_apply=False if gain is in e-/ADU and data is in ADU
        cleaned_data, cr_mask = cosmicray_lacosmic(
            data,
            sigclip=sigclip,
            objlim=objlim,
            gain=gain,
            readnoise=readnoise,
            satlevel=satlevel,
            gain_apply=False, # Assume gain/readnoise are in e-, data in ADU
            verbose=verbose
        )
        print(f"Found {np.sum(cr_mask)} cosmic ray pixels.")
        # cr_mask is True where CRs were detected
        return cr_mask
if __name__=="__main__":
    # clb = calibrator()
    # #print(clb.stacking('/home/yichengrui/workspace/TianYu/pipeline/image_process/out/caliborate/superbias_mgo.fit',2,method = "median"))
    # #clb.stacking('/home/yichengrui/workspace/TianYu/pipeline/image_process/out/caliborate/superflat_mgo_uncorred.fit',1,method = "median",img_type="flat_raw")
    # #clb.calibration(obs_id=1,hierarchy=2,consider_flat = False,img_type = "flat_raw",bias_id = 2,bias_hierarchy = 2,debug = False)
    # #clb.calibration(obs_id=3,hierarchy=1,consider_flat = True,img_type = "science_raw",bias_id = 2,bias_hierarchy = 2,debug = False)
    # clb.calibration(obs_id=8,hierarchy=1,consider_flat = True,img_type = "deep_raw",bias_id = 2,bias_hierarchy = 2,debug = False)
    # clb.stacking('/home/yichengrui/workspace/TianYu/pipeline/image_process/out/reduced_res/stack_img/M81_l.fit',obs_id = 8,method = "mean",img_type="deep_processed")
    pass