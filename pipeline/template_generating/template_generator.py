#import cupy as cp
import numpy as np
import sys
import time
from astropy.io import fits
import matplotlib.pyplot as plt
import astrometry
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
#sys.path.append('/home/yichengrui/workspace/TianYu/pipeline/scheduling/')
import Tianyu_pipeline.pipeline.scheduling.data_loader as dl
#sys.path.append('/home/yichengrui/workspace/TianYu/pipeline/template_generating/')
import Tianyu_pipeline.pipeline.template_generating.image_alignment as il
#sys.path.append('/home/yichengrui/workspace/TianYu/pipeline/image_process/')
import Tianyu_pipeline.pipeline.image_process.calibrator as cl
import use_SE as SE
import flux_extractor as FE



def logodds_callback(logodds_list: list[float]) -> astrometry.Action:
    if len(logodds_list) < 3:
        return astrometry.Action.CONTINUE
    if logodds_list[1] > logodds_list[0] - 10 and logodds_list[2] > logodds_list[0] - 10:
        return astrometry.Action.STOP
    return astrometry.Action.CONTINUE

class template_generator:
    def __init__(self):
        self.dl = dl.data_loader()
        self.al = il.alignment()
        self.cl = cl.calibrator()
        self.se = SE.SE()
        self.fe = FE.flux_extractor()

    def generate_template_image_and_sky(self,alignment_img_obs_id, alignment_img_hierarchy, alignment_img_type,good_img_threshold = 0.5,good_star_threshold = 1.2,method = "mean"):
        outpath = "/home/yichengrui/workspace/TianYu/pipeline/image_process/out/reduced_res/stack_img/template_obs_"+str(alignment_img_obs_id)+"_hierarchy_"+str(alignment_img_hierarchy)+"_imgtype_"+str(alignment_img_type)+"_"+str(hash(time.time()))+".fit"
        mycursor = self.dl.cnx.cursor()
        arg = (alignment_img_obs_id,alignment_img_hierarchy,self.dl.image_type_id[alignment_img_type])
        sql = "SELECT i.obs_id,i.img_path,i.image_id from img as i where i.obs_id = %s and hierarchy = %s and i.image_type_id = %s;"
        mycursor.execute(sql,arg)
        myresult = mycursor.fetchall()
        if len(myresult)<1:
            print("Cannot fing image")
            return -1
        
        base_id = myresult[0][2]
        alignment_result = self.al.get_deviation(base_id,alignment_img_obs_id, alignment_img_hierarchy, alignment_img_type,good_star_threshold = good_star_threshold)

        used_img_list = []
        for alires in alignment_result:
            if alires[3]<good_img_threshold:
                continue
            else:
                used_img_list.append(alires[0])
        new_img_id = self.cl.stacking(outpath,stack_img_id_list = used_img_list,mode = "fixed_id",method = "mean")

        mycursor = self.dl.cnx.cursor()
        arg = (new_img_id,)
        sql = "INSERT INTO sky (template_image_id) VALUES (%s);"
        mycursor.execute(sql,arg)
        self.dl.cnx.commit()

        mycursor = self.dl.cnx.cursor()
        mycursor.execute("SELECT LAST_INSERT_ID();")
        myresult = mycursor.fetchall()
        new_sky_id = myresult[0][0] #auto_increment
        
        return new_sky_id,outpath

    def generate_template_pos(self,sky_id,show = False,test = False,register = True,Gmag_lim = 20,num_class = 10):
        def show_template(img_path,star_pix,star_info,stars,cat_star_list):
            star_pix = np.array(star_pix)
            img = fits.getdata(img_path)
            extent = (0,img.shape[1],0,img.shape[0])
            #print(BG)
            plt.figure(figsize = (40,20))
            plt.imshow(img,vmin = 1300, vmax = 1500,extent=extent, origin='lower')
            
            t = np.linspace(0,2*np.pi,10)
            #r = 5
            ct = 0
            for i,xy in enumerate(star_pix):
                #print(xy[0],xy[1])
                if xy[1]>img.shape[0] or xy[1]<0 or xy[0]>img.shape[1] or xy[0]<0:
                    continue
                else:
                    r = 30-star_info[i][3]*1.5
                    if star_info[i][7]==star_info[i][7]:

                        plt.plot(xy[0]+r*np.cos(t),xy[1]+r*np.sin(t),'r',alpha = 0.8)
                    else:
                        plt.plot(xy[0]+r*np.cos(t),xy[1]+r*np.sin(t),'y',alpha = 0.8)
                    ct+=1
            for i,xy in enumerate(stars):
                #print(xy[0],xy[1])
                if xy[1]>img.shape[0] or xy[1]<0 or xy[0]>img.shape[1] or xy[0]<0:
                    continue
                else:
                    r = 10
                    plt.plot(xy[0]+r*np.cos(t),xy[1]+r*np.sin(t),'g',alpha = 0.8)
                    ct+=1
            for stmap in cat_star_list:
                plt.plot([star_pix[stmap[1]][0],stars[stmap[0]][0]],[star_pix[stmap[1]][1],stars[stmap[0]][1]],'b',alpha = 0.8)
            print(ct)
            plt.savefig("demo.pdf")



        mycursor = self.dl.cnx.cursor()
        mycursor.execute("SELECT i.img_path,tn.ra,tn.`dec`,i.jd_utc_mid from sky as s inner JOIN img as i ON s.template_image_id=i.image_id inner JOIN observation as obs ON i.obs_id = obs.obs_id inner JOIN target_n as tn ON obs.target_id = tn.target_id where s.sky_id = "+str(sky_id)+";")
        myresult = mycursor.fetchall()
        sky_template_img_path = myresult[0][0]
        sky_template_ra = myresult[0][1]
        sky_template_dec = myresult[0][2]
        sky_template_time = myresult[0][3]
        sky_template_img = fits.getdata(sky_template_img_path)

        print(sky_template_ra,sky_template_dec)
        #temp_out = "temp"+str(hash(time.time()))+".fit"
        se_out = self.se.use(sky_template_img_path,use_sep = True)
        x_stars = np.squeeze(se_out['x']).reshape(-1,1)
        y_stars = np.squeeze(se_out['y']).reshape(-1,1)
        stars = np.hstack([x_stars,y_stars])
        x_stars = np.squeeze(x_stars)
        y_stars = np.squeeze(y_stars)
        xx_stars = np.squeeze(se_out['xx']).reshape(-1,1)
        yy_stars = np.squeeze(se_out['yy']).reshape(-1,1)
        xy_stars = np.squeeze(se_out['xy']).reshape(-1,1)
        flux = np.squeeze(se_out['flux'])
        lambda1 = np.squeeze((xx_stars+yy_stars)/2+np.sqrt(((xx_stars-yy_stars)/2)**2+xy_stars**2))
        lambda2 = np.squeeze((xx_stars+yy_stars)/2-np.sqrt(((xx_stars-yy_stars)/2)**2+xy_stars**2))
        gap = 30
        gap_mask = (x_stars>gap)&(x_stars<sky_template_img.shape[1]-gap)&(y_stars>gap)&(y_stars<sky_template_img.shape[0]-gap)
        lambda1 = lambda1[gap_mask]
        stars = stars[gap_mask]
        flux = flux[gap_mask]
        x_stars,y_stars = x_stars[gap_mask],y_stars[gap_mask]
        #y_stars = y_stars[(x_stars>5)&(x_stars<sky_template_img.shape[1]-5)&(y_stars>5)&(y_stars<sky_template_img.shape[0]-5)]
        flux_sort = np.sort(flux)
        
        stars_cm = stars[(flux>flux_sort[-150])&(x_stars>0.05*sky_template_img.shape[1])&(x_stars<0.95*sky_template_img.shape[1])&(y_stars>0.05*sky_template_img.shape[0])&(y_stars<0.95*sky_template_img.shape[0])]

        print(len(stars_cm))
        print(stars.shape)
        solver = astrometry.Solver(
            astrometry.series_5200.index_files(
                cache_directory="/home/yichengrui/workspace/TianYu/astrometry/astrometry_cache",
                scales={6},
            )
        )
        print('resolving astrometry using astrometry.net')
        solution = solver.solve(
                    stars=stars_cm,
                    size_hint=astrometry.SizeHint(
                        lower_arcsec_per_pixel=0.85,
                        upper_arcsec_per_pixel=0.93,
                    ),
                    position_hint=astrometry.PositionHint(
                ra_deg=sky_template_ra,
                dec_deg=sky_template_dec,
                radius_deg=3.0,
            ),
                solution_parameters=astrometry.SolutionParameters(
                
            ),
        )
        # pass
        if not solution.has_match():
            return -1
        catalogue_star = self.dl.search_GDR3_by_square(ra = solution.best_match().center_ra_deg,dec = solution.best_match().center_dec_deg,fov = 1.1,Gmag_limit = Gmag_lim)
        catalogue_star_id = [star[0] for star in catalogue_star]
        catalogue_star = np.array([np.array(star) for star in catalogue_star],dtype = np.float64)
        JD_20160 = 2457388.50000
        JD_2000 = 2451544.50000
        time_year = (float(sky_template_time) - JD_20160)/365.25
        #print(time_year,np.nan_to_num(catalogue_star[:,5])/3600000*time_year)
        catalogue_star[:,1] = catalogue_star[:,1]+np.nan_to_num(catalogue_star[:,5])*time_year/3600000
        catalogue_star[:,2] = catalogue_star[:,2]+np.nan_to_num(catalogue_star[:,6])*time_year/3600000
        sky_coord = SkyCoord(ra=catalogue_star[:,1]*u.deg,dec = catalogue_star[:,2]*u.deg,frame='icrs')
        fk5_coord = sky_coord.transform_to('fk5')
        print(np.array([fk5_coord.ra.deg,fk5_coord.dec.deg]).T.shape)
        #print(catalogue_star)
        print(f"{solution.best_match().center_ra_deg=}")
        print(f"{solution.best_match().center_dec_deg=}")
        print(f"{solution.best_match().scale_arcsec_per_pixel=}")
        wcs = solution.best_match().astropy_wcs()
        pixels = wcs.all_world2pix(
            np.array([fk5_coord.ra.deg,fk5_coord.dec.deg]).T,
            0,
        )
        # print(pixels)
        # print(stars)
        cat_star_list = []

        for i,xy in enumerate(stars):
            dist = np.sqrt(np.sum((np.array(pixels)-xy)**2,axis = 1))
            if np.min(dist)<4:
                cat_star_list.append([i,np.argmin(dist)])

        if show:
            print("plotting result")
            show_template(sky_template_img_path,pixels,catalogue_star,stars,cat_star_list)
        cat_star_list = np.array(cat_star_list,dtype = int)
        print(len(set(cat_star_list[:,1])),len(cat_star_list))
        print('Matched',len(cat_star_list),'pairs.')
        selected_star_index = np.squeeze(cat_star_list[:,0])
        app_diameter = np.maximum(np.minimum((4*(np.sqrt(lambda1[selected_star_index]))+4.5).astype(int),25),3)
        star_selected = stars[selected_star_index]
        #print(app_diameter)
        print("extracting flux in template image")
        flux_myextract, flux_err_myextract,saturate = self.fe.flux_extract(sky_template_img,star_selected,app_diameter,25,29)
        print(flux_myextract,flux_err_myextract,saturate)
        
        print(flux_myextract[saturate],flux_err_myextract[saturate])
        mycursor = self.dl.cnx.cursor()
        quantile_flux = pd.qcut(flux_myextract,q = num_class, labels=False)
        plt.figure()
        args_list = []
        
        Cat_gmag_list = []
        log_flux_list = []
        for i,cm in enumerate(cat_star_list):
            x,y = stars[cm[0]]
            _,ra,dec,phot_g_mean_mag,parallax,pmra,pmdec,best_class_name_id,best_score = catalogue_star[cm[1]]
            source_id = catalogue_star_id[cm[1]]
            #print(type(source_id))
            args = [sky_id,float(x),float(y),source_id,float(flux_myextract[i]),float(flux_err_myextract[i]),bool(saturate[i]),bool(best_class_name_id==best_class_name_id),int(quantile_flux[i]),int(app_diameter[i])]
            args_list.append(args)
            Cat_gmag_list.append(phot_g_mean_mag)
            log_flux_list.append(np.log(flux_myextract[i]))
            #print(args)
        Cat_gmag_list = np.array(Cat_gmag_list).reshape(-1,1) #x1
        log_flux_list = np.array(log_flux_list).reshape(-1,1) #Y
        X = np.hstack([np.ones(Cat_gmag_list.shape),Cat_gmag_list])
        theta = np.linalg.inv((X.T.dot(X))).dot(X.T).dot(log_flux_list)
        Xtheta = X.dot(theta)
        X_hat = np.linspace(7,20,100).reshape(-1,1)
        Y = np.squeeze(np.hstack([np.ones(X_hat.shape),X_hat]).dot(theta))
        outlier_list = np.zeros(log_flux_list.shape,dtype = bool)
        deviation_list = np.zeros(log_flux_list.shape)
        for class_index in range(num_class):
            #X_this_class = X[quantile_flux==class_index]
            Xtheta_this_class = Xtheta[quantile_flux==class_index]
            flux_this_class = log_flux_list[quantile_flux==class_index]
            residue = flux_this_class-Xtheta_this_class
            med_res = np.median(residue)
            #std_sample = np.std(residue)

            low,high = np.percentile(residue, (15.87,84.13))
            mad_sample = (high-low)/2
            outlier = (residue<med_res-5.0*mad_sample)|(residue>med_res+5.0*mad_sample)

            outlier_list[quantile_flux==class_index] = outlier
            deviation_list[quantile_flux==class_index] = (residue-med_res)/mad_sample
            


        plt.plot(Cat_gmag_list[outlier_list],log_flux_list[outlier_list],'g.')
        plt.plot(Cat_gmag_list[np.logical_not(outlier_list)],log_flux_list[np.logical_not(outlier_list)],'b.')
        plt.plot(np.squeeze(X_hat),Y,'--r')

        plt.xlabel('Gmag')
        plt.ylabel('log F')
        plt.savefig("mag-flux.pdf")
        #print('plotting_cut_pixel...')
        sql = "INSERT INTO star (sky_id,x_template,y_template,gdr3_id,flux_template,flux_err_template,saturate,variability,reference_class,apparature_diameter,flux_deviation) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);"
        for i in range(len(outlier_list)):
            #args_list[i].append(float(outlier_list[i]))
            args_list[i].append(float(deviation_list[i][0]))
            mycursor.execute(sql,args_list[i])
            #if outlier_list[i]:
                #print(args_list[i][0])
                #self.fe.show_cut_pixel(sky_template_img,[args_list[i][1],args_list[i][2]],app_diameter[i],25,29,'/home/yichengrui/workspace/TianYu/pipeline/template_generating/temp_res/'+str(args_list[i][3])+'_mag_'+str(Cat_gmag_list[i][0])+'_dev_'+str(deviation_list[i][0])+'.pdf')


        if register:
            self.dl.cnx.commit()
        if test:
            mag_list = []
            long_axis =  []
            x_list = []
            y_list = []
            for cm in cat_star_list:
                x,y = stars[cm[0]]
                x_list.append(x)
                y_list.append(y)
                moment2 = np.sqrt(lambda1[cm[0]])
                #print(catalogue_star[cm[1]][3])
                #print(moment2)
                mag_list.append(catalogue_star[cm[1]][3])
                long_axis.append(moment2)
                if moment2>20:
                    print(stars[cm[0]])
            #cat_star_list = np.array(cat_star_list)
            #stars = np.array(stars)
            mag_list = np.array(mag_list).reshape(-1,1)
            x_list = (np.array(x_list).reshape(-1,1)[mag_list<11]).reshape(-1,1)
            y_list = (np.array(y_list).reshape(-1,1)[mag_list<11]).reshape(-1,1)

            plt.figure()
            plt.scatter(mag_list,long_axis)

            print('template_path =',sky_template_img_path)
            long_axis = (np.array(long_axis).reshape(-1,1)[mag_list<11]).reshape(-1,1)
            print('star_coord =',np.hstack([x_list,y_list,long_axis]))
            plt.xlabel("G [mag]")
            plt.ylabel("long axis of star image ellipse [Pix]")
            plt.savefig('scatter.pdf')


        

if __name__=="__main__":
    tg = template_generator()
    tg.generate_template_pos(1,show = False,test = False,register = True)
    
    #print(tg.generate_template_image_and_sky(5,1,"deep_processed"))

