import os
import pathlib
import mysql.connector
import numpy as np
from astropy.io import fits
import glob
import tqdm
import time
import sep
import pandas as pd
from astropy.stats import sigma_clip 
import Tianyu_pipeline.pipeline.utils.sql_interface as sql_interface
import Tianyu_pipeline.pipeline.utils.data_loader as data_loader
import Tianyu_pipeline.pipeline.dev.file_system.file_system as fs
import Tianyu_pipeline.pipeline.dev.process_management.process_publisher as process_pub 
import Tianyu_pipeline.pipeline.utils.process_site_getter as psg
import astrometry
import astropy.wcs
from sklearn.cluster import KMeans
class calibrator:
    def __init__(self,site_id=-1,group_id=-1):
        self.psg = psg.process_site_getter()
        self.site_info = self.psg.get_channel()

        # if site_id==-1:
        #     site_id = self.site_info['site_id']
        # if group_id==-1:
        #     group_id = site_id
        self.dl = data_loader.data_loader()
        self.fs = fs.file_system()
        self.sql_interface = sql_interface.sql_interface()
    def astrometric_calibration(self):
        pass
    def find_nearest_kdtree(self,x1, y1, x2, y2):
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
        distances, nearest_indices = tree.query(np.c_[x1, y1], k=[1,2,3])
        return distances, nearest_indices
    def crossmatch_external(self,sky_id, show = False):

        # 3 steps
        # 1 obtain archive star
        # 2 find Gaia star
        # crossmatch
        sql = """
SELECT 
    ts.source_id as source_id, 
    tsp.template_img_id as image_id, 
    tsp.x_template as x_template, 
    tsp.y_template as y_template, 
    sim.absolute_deviation_x as absolute_deviation_x, 
    sim.absolute_deviation_y as absolute_deviation_y ,
    tsp.flux_template as flux,
    tsp.e_flux_template as e_flux
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
    flux DESC;
"""
        args = (sky_id,)
        archive_star_result = self.sql_interface.query(sql, args)
        print(archive_star_result)
        dir_this = os.path.dirname(__file__)
        dir_data = os.path.join(dir_this,'data')
        print(dir_data)

        sql = "SELECT * FROM sky where sky_id = %s;"
        args = (sky_id,)
        sky_result = self.sql_interface.query(sql, args)

        #Try different solver to get the best result


        solvers = ['astrometry.Solver(astrometry.series_4100.index_files(cache_directory=dir_data,scales=[7,8,9,10,11,12,13,14],))'
                   ,
                   'astrometry.Solver(astrometry.series_4200.index_files(cache_directory=dir_data,scales=[6,7,8,9],))',
                     'astrometry.Solver(astrometry.series_5200.index_files(cache_directory=dir_data,scales=[6],))']
        for solver_str in solvers:
            solver = eval(solver_str)
            print('resolving astrometry using astrometry.net')
            print(sky_result.loc[0,'ra'],sky_result.loc[0,'dec'])
            solution = solver.solve(
                        stars_xs=np.array(archive_star_result['x_template'])[:50],
                        stars_ys=np.array(archive_star_result['y_template'])[:50],
                        size_hint=astrometry.SizeHint(
                            lower_arcsec_per_pixel=0.2,
                            upper_arcsec_per_pixel=2,
                        ),
                        position_hint=astrometry.PositionHint(
                    ra_deg=sky_result.loc[0,'ra'],
                    dec_deg=sky_result.loc[0,'dec'],
                    radius_deg=.5,
                ),
                solve_id = None,
                tune_up_logodds_threshold = np.log(1e6),
                output_logodds_threshold = np.log(1e9)
            )
            if solution.has_match():
                break
            else:
                print('failed, tyring next solver')
        if not solution.has_match():

            print('All solver failed!')
            return 0
        print(f"{solution.best_match().center_ra_deg=}")
        print(f"{solution.best_match().center_dec_deg=}")
        print(f"{solution.best_match().scale_arcsec_per_pixel=}")
        print('searching gdr3 targets')
        Gaia_query_res = self.dl.search_GDR3_by_square(ra = solution.best_match().center_ra_deg,dec = solution.best_match().center_dec_deg, fov = 0.1+(sky_result.loc[0,'fov_x']**2+sky_result.loc[0,'fov_y']**2)**0.5/2,Gmag_limit = 20)
        #print(Gaia_query_res['in_vari_classification_result'])
        is_variable = []
        for i in Gaia_query_res['in_vari_classification_result']:
            #print(i,type(i))
            if i==True or i==False:
                is_variable.append(True)
            else:
                is_variable.append(False)
        is_variable = np.array(is_variable)
        
        print(is_variable)
        wcs = astropy.wcs.WCS(solution.best_match().wcs_fields)
        pixels = wcs.all_world2pix(
                np.hstack([Gaia_query_res['ra'].reshape(-1,1), Gaia_query_res['dec'].reshape(-1,1)]),
                0,
            )
        
        x1 = np.squeeze(archive_star_result['x_template'])
        y1 = np.squeeze(archive_star_result['y_template'])
        x2 = np.squeeze(pixels[:,0])
        y2 = np.squeeze(pixels[:,1])
        
        distances, indices = self.find_nearest_kdtree(x1,y1,x2,y2)
        distances = distances * solution.best_match().scale_arcsec_per_pixel

        single_match = ((distances[:,1]/distances[:,0])>1.5) & (distances[:,0]<4)
        no_match = (distances[:,0]>=4)
        binary_match = ~single_match & ~no_match
        # kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
        # kmeans.fit(np.log(distances))
        # print(np.exp(kmeans.cluster_centers_))
        # record the crossmatch results in db
        arg_update = []
        arg_insert = []
        for i,r in archive_star_result.iterrows():
            source_id_this = int(r['source_id'])
            sql = "SELECT * FROM source_crossmatch WHERE source_id = %s;"
            args = (source_id_this,)
            result = self.sql_interface.query(sql,args)
            update = len(result)
            args = (int(Gaia_query_res['SOURCE_ID'][indices[i,0]]),int(Gaia_query_res['SOURCE_ID'][indices[i,1]]),int(Gaia_query_res['SOURCE_ID'][indices[i,2]]),float(distances[i,0]),float(distances[i,1]),float(distances[i,2]),int(is_variable[indices[i,0]]),int(is_variable[indices[i,1]]),int(is_variable[indices[i,2]]),int(source_id_this))
            if update:
                arg_update.append(args)
                #
            else:
                arg_insert.append(args)
                #

            #print(args)
        sql = "INSERT INTO source_crossmatch (gdr3_id1,gdr3_id2,gdr3_id3,gdr3_dist1,gdr3_dist2,gdr3_dist3,gdr3_is_variable1,gdr3_is_variable2,gdr3_is_variable3,source_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);"
        self.sql_interface.executemany(sql,arg_insert)
        sql = "UPDATE source_crossmatch SET gdr3_id1 = %s,gdr3_id2 = %s,gdr3_id3 = %s, gdr3_dist1 = %s,gdr3_dist2 = %s,gdr3_dist3 = %s,gdr3_is_variable1 = %s,gdr3_is_variable2 = %s,gdr3_is_variable3 = %s WHERE source_id = %s;"
        self.sql_interface.executemany(sql,arg_update)
        if show:
            index_target = np.argmax(Gaia_query_res['SOURCE_ID']==2188601779406152448)
            index_real_target = np.argmax(indices[:,0]==index_target)
            print(archive_star_result.iloc[index_real_target])
            import matplotlib.pyplot as plt
            plt.scatter(x1[single_match],y1[single_match],label='resolve result single_match',marker = "*",alpha = 0.2)
            plt.scatter(x1[no_match],y1[no_match],label='resolve result no_match',marker = "^",alpha = 0.2)
            plt.scatter(x1[binary_match],y1[binary_match],label='resolve result binary_match',marker = "<",alpha = 0.2)
            #plt.scatter(x1[kmeans.labels_==2],y1[kmeans.labels_==2],label='resolve result class 2',marker = ">",alpha = 0.2)
            plt.plot(x2,y2,'.',label='gdr3 result',alpha = 0.1)
            
            plt.legend()
            plt.show()
            
            plt.xscale('log')
            plt.yscale('log')
            plt.scatter(np.squeeze(distances[:,0])[single_match],np.squeeze(distances[:,1])[single_match])
            plt.scatter(np.squeeze(distances[:,0])[no_match],np.squeeze(distances[:,1])[no_match])
            plt.scatter(np.squeeze(distances[:,0])[binary_match],np.squeeze(distances[:,1])[binary_match])
            #plt.scatter(np.squeeze(distances[:,0])[kmeans.labels_==2],np.squeeze(distances[:,1])[kmeans.labels_==2])
            plt.show()
        return 1
    def absolute_photometric_calibration_single_frame(self,PID_this,flux_PID,show = True):
        # Get the raw flux of the flux_PID
        def design_matrix(stars):
            x = np.array(stars['normalized_x']).reshape(-1,1)
            y = np.array(stars['normalized_y']).reshape(-1,1)
            g_b = np.array(stars['phot_g_mean_mag']).reshape(-1,1)-np.array(stars['phot_bp_mean_mag']).reshape(-1,1)
            g_r = np.array(stars['phot_g_mean_mag']).reshape(-1,1)-np.array(stars['phot_rp_mean_mag']).reshape(-1,1)
            ones = np.ones_like(x)
            design_matrix = np.hstack([ones,x,y,x**2,y**2,x**3,y**3,x*y,x*y**2,x**2*y,g_b,g_b**2,g_r,g_r**2,g_b*g_r])
            return design_matrix

        def calculate_prediction(coefficients,stars):
            coefficients = np.array(coefficients).reshape(-1,1)
            design_matrix_this = design_matrix(stars)
            prediction = np.dot(design_matrix_this,coefficients)
            
            return prediction
        def fit_and_filter(used_star):
            # flux calibration using Gaia Data, return the mask of outliers
            # model: mg_gaia = -2.5log10(flux_raw)+C+a1 x+b1 y+ a2 x^2 + b2 y^2 + a3 x^3 + b3 y^3 + c11 xy + c12 xy^2 + c21 x^2y + cb1(mg-mb) + cb2(mg-mb)^2 +cr1(mg-mr) + cr2(mg-mr)^2 +cbr(mg-mr)(mg-mb)
            # chi2 = (mg_gaia-mg_gaia_model)^2/sigma^2; sigma = 1/phot_g_mean_flux_over_error*2.5/log(10)
            # mask the outliers that have abs((mg_gaia-mg_gaia_model)/sigma)>3
            # return the mask and coefficients
            base_mag = -2.5*np.log10(np.array(used_star['flux_raw']).reshape(-1,1))
            base_mag_error = np.squeeze(1/(np.array(used_star['phot_g_mean_flux_over_error']).reshape(-1,1))*2.5/np.log(10))
            sigmainv = np.diag(1/base_mag_error**2)
            X = design_matrix(used_star)
            Y = np.array(used_star['phot_g_mean_mag']).reshape(-1,1)-base_mag
            Y = Y.reshape(-1,1)
            print(X.shape,Y.shape,sigmainv.shape)
            theta = np.linalg.inv(X.T.dot(sigmainv).dot(X)).dot(X.T).dot(sigmainv).dot(Y)
            prediction = calculate_prediction(theta,used_star)
            residual = np.abs((np.squeeze(prediction)-np.squeeze(Y))/base_mag_error)
            print(theta)
            return np.squeeze(residual>300),theta



        sql = "SELECT spi.star_pixel_img_id,spi.source_id,spi.flux_raw,tsp.x_template,tsp.y_template,cm.gdr3_id1,cm.gdr3_dist1,cm.gdr3_is_variable1,cm.gdr3_id2,cm.gdr3_dist2,cm.gdr3_is_variable2 FROM star_pixel_img as spi LEFT JOIN tianyu_source_position as tsp ON tsp.source_id = spi.source_id LEFT JOIN source_crossmatch as cm ON cm.source_id = tsp.source_id WHERE spi.birth_process_id = %s;"

        args = (flux_PID,)
        raw_flux = self.sql_interface.query(sql,args)
        # Get the reference star
        raw_flux['normalized_x'] = (raw_flux['x_template']-np.min(raw_flux['x_template']))/(np.max(raw_flux['x_template'])-np.min(raw_flux['x_template']))*2-1
        raw_flux['normalized_y'] = (raw_flux['y_template']-np.min(raw_flux['y_template']))/(np.max(raw_flux['y_template'])-np.min(raw_flux['y_template']))*2-1

        first_source_id = raw_flux.loc[0,'source_id']
        sql = "SELECT * FROM tianyu_source JOIN sky ON sky.sky_id =tianyu_source.sky_id  WHERE source_id = %s;"
        args = (int(first_source_id),)
        result = self.sql_interface.query(sql,args)
        ra = float(result.loc[0,'ra'])
        dec = float(result.loc[0,'dec'])
        fov = 0.1+np.sqrt(float(result.loc[0,'fov_x'])**2+float(result.loc[0,'fov_y'])**2)/2
        Gaia_query_res = self.dl.search_GDR3_by_square(ra = ra,dec = dec, fov = fov,Gmag_limit = 20)
        Gaia_df = Gaia_query_res.to_pandas()
        # print(Gaia_df)
        # print(raw_flux)
        # merge the Gaia data with the raw flux according to source_id
        merge_result = pd.merge(raw_flux,Gaia_df,left_on = 'gdr3_id1',right_on = 'SOURCE_ID',how = 'left')
        
        # Filter out the utilized stars
        mask = (merge_result['gdr3_dist1']<4)&(merge_result['gdr3_is_variable1']==False)&(merge_result['flux_raw']>0)&(merge_result['gdr3_dist2']/merge_result['gdr3_dist1']>1.5) & (merge_result['phot_g_mean_mag']<18)& (merge_result['phot_bp_mean_mag']==merge_result['phot_bp_mean_mag'])& (merge_result['phot_rp_mean_mag']==merge_result['phot_rp_mean_mag'])
        used_star = merge_result[mask]
        while 1:
            mask,theta = fit_and_filter(used_star)
            if np.sum(mask)==0:
                print(theta)
                break
            else:
                print(f'{np.sum(mask)} outliers detected, refitting')
            used_star = used_star[~mask]
        prediction_all = np.squeeze(calculate_prediction(theta,merge_result))-2.5*np.log10(np.squeeze(np.array(merge_result['flux_raw'])))
        merge_result['mag_calibrated_absolute'] = prediction_all
        predice_used = np.squeeze(calculate_prediction(theta,used_star))-2.5*np.log10(np.squeeze(np.array(used_star['flux_raw'])))
        used_star['mag_calibrated_absolute'] = predice_used
        merge_result['mag_calibrated_absolute_error'] = 1/merge_result['phot_g_mean_flux_over_error']*2.5/np.log(10)
        upload_result = merge_result.dropna(subset=['mag_calibrated_absolute'])
        if show:
            import matplotlib.pyplot as plt
            print(len(upload_result),len(merge_result),len(used_star))
            plt.scatter(upload_result['phot_g_mean_mag'],upload_result['mag_calibrated_absolute'],alpha = 0.2,label = 'all calibrated')
            plt.scatter(used_star['phot_g_mean_mag'],used_star['mag_calibrated_absolute'],alpha = 0.2,label = 'used calibrated')
            plt.plot([10,20],[10,20],label = 'y=x')
            plt.xlabel('Gaia G mag')
            plt.ylabel('calibrated G mag')
            plt.legend()
            plt.show()

        return 1
    def select_reference_star(self,PID,template_generation_PID,template_crossmatch_PID,dis_quantile_threshold = 0.5, minimum_marginal_deviation = 200):
        # load database, find best reference star
        sql = "SELECT * FROM sky_image_map WHERE process_id = %s;"
        args = (template_generation_PID,)
        result = self.sql_interface.query(sql,args)
        assert len(result)==1
        image_id = result.loc[0,'image_id']
        # Only resolved star can be treated as reference star
        sql = "SELECT * FROM tianyu_source_position NATURAL JOIN source_crossmatch WHERE template_img_id = %s ;"
        args = (int(image_id),)
        result = self.sql_interface.query(sql,args)
        x_template = np.array(result['x_template'])
        y_template = np.array(result['y_template'])
        main_is_not_variable = ~np.array(result['gdr3_is_variable1'])
        is_single_match = (np.array(result['gdr3_dist2'])/np.array(result['gdr3_dist1']))>1.5
        #flux_template = np.array(result['flux_template'])
        center_star_mask = (x_template<np.max(x_template)-minimum_marginal_deviation)&(x_template>minimum_marginal_deviation)&(y_template<np.max(y_template)-minimum_marginal_deviation)&(y_template>minimum_marginal_deviation)
        distance, index = self.find_nearest_kdtree(x_template,y_template,x_template,y_template)
        dis_threshold = np.quantile(distance[:,1],dis_quantile_threshold)
        reference_star_indices = np.where((distance[:,1] > dis_threshold)&center_star_mask&is_single_match&main_is_not_variable)
        reference_star_source_id = result.loc[reference_star_indices,'source_id']
        sql = "SELECT * FROM img where image_id = %s;"
        args = (int(image_id),)
        result = self.sql_interface.query(sql,args)
        obs_id = int(result.loc[0,'obs_id'])
        # Delete the old reference star
        sql = "DELETE FROM reference_star where obs_id = %s;"
        args = (obs_id,)
        self.sql_interface.execute(sql,args)

        sql = "INSERT INTO reference_star (obs_id, source_id, process_id) VALUES (%s,%s,%s);"
        args = [(int(obs_id),int(i),int(PID)) for i in reference_star_source_id]
        #print(len(args))
        self.sql_interface.executemany(sql,args)
        # print(reference_star_source_id,'\n',obs_id)
        print(sql,args)
        #return {'reference_id':reference_star_source_id,"reference_x":x_template[reference_star_indices],"reference_y":y_template[reference_star_indices]}

        return 1
    def select_reference_star_and_calibrate(self,PID,template_generation_PID,dis_quantile_threshold = 0.5,mean_number_reference = 6,pos_quantile_number = 3):
        def mask_select(item,bins,index):
            #print(index,bins)
            eps = 1e-7
            if len(bins)<=2:
                #print('only two bin')
                return np.ones(item.shape).astype(bool)
            if index ==0:
                #print('first bin')
                mask = item<=bins[1]#+eps
                return mask
            if index == len(bins)-2:
                #print('last bin')
                mask = item>bins[-2]#-eps
                return mask
            print('middle bin')
            mask = (item>bins[index]) & (item<=bins[index+1])
            return mask
        # load database, find init reference star
        print('Getting init reference star')
        sql = "SELECT * FROM sky_image_map WHERE process_id = %s;"
        args = (template_generation_PID,)
        result = self.sql_interface.query(sql,args)
        assert len(result)==1
        temp_image_id = result.loc[0,'image_id']
        # Only resolved star can be treated as reference star
        sql = "SELECT * FROM tianyu_source_position NATURAL JOIN source_crossmatch INNER JOIN img ON img.image_id = template_img_id  WHERE template_img_id = %s Order by source_id;"
        args = (int(temp_image_id),)
        all_source_info_df = self.sql_interface.query(sql,args)
        obs_id = int(all_source_info_df.loc[0,'obs_id'])

        star_grouped_sql = 'WITH temp AS (SELECT source_id, COUNT(*) AS ct FROM star_pixel_img INNER JOIN img ON img.image_id = star_pixel_img.image_id WHERE img.obs_id = %s GROUP BY source_id), max_ct AS (SELECT MAX(ct) AS max_ct FROM temp) SELECT * FROM temp NATURAL JOIN tianyu_source_position NATURAL JOIN source_crossmatch where ct = (select max_ct from max_ct) ORDER BY source_id;'
        args = (obs_id,)
        source_id_complete = self.sql_interface.query(star_grouped_sql,args)
        number_of_complete_star = len(source_id_complete)
        N_f = int(source_id_complete.loc[0,'ct'])
        print(source_id_complete)
        print(number_of_complete_star,N_f)



        star_complete_source_sql = '''SELECT 
    spi.star_pixel_img_id AS star_pixel_img_id, 
    spi.source_id, 
    spi.flux_raw AS flux_raw, 
    spi.flux_raw_error AS flux_raw_error, 
    tsp.x_template AS x_template, 
    tsp.y_template AS y_template, 
    tsp.flux_template AS flux_template
FROM 
    star_pixel_img AS spi
INNER JOIN 
    tianyu_source_position AS tsp 
    ON spi.source_id = tsp.source_id
INNER JOIN 
    img 
    ON img.image_id = spi.image_id
INNER JOIN 
    (
        WITH temp AS (
            SELECT 
                star_pixel_img.source_id, 
                COUNT(*) AS ct 
            FROM 
                star_pixel_img 
            INNER JOIN 
                img 
                ON img.image_id = star_pixel_img.image_id 
            WHERE 
                img.obs_id = %s 
            GROUP BY 
                star_pixel_img.source_id
        ), 
        max_ct AS (
            SELECT 
                MAX(ct) AS max_ct 
            FROM 
                temp
        ) 
        SELECT 
            * 
        FROM 
            temp 
        WHERE 
            ct = (SELECT max_ct FROM max_ct)
    ) AS besttab ON spi.source_id = besttab.source_id
WHERE 
    img.obs_id = %s AND img.coadd_weight>0.5
ORDER BY 
    spi.source_id, img.jd_utc_mid;'''
        args = (obs_id,obs_id)
        star_complete_source = self.sql_interface.query(star_complete_source_sql,args)
        print(star_complete_source)
        raw_flux_all = np.array(star_complete_source['flux_raw'].values).reshape(number_of_complete_star,-1)


        # initial select of reference star

        x_template = np.array(source_id_complete['x_template'])
        y_template = np.array(source_id_complete['y_template'])
        flux_template = np.array(source_id_complete['flux_template'])
        main_is_not_variable = ~np.array(source_id_complete['gdr3_is_variable1'])
        is_single_match = (np.array(source_id_complete['gdr3_dist2'])/np.array(source_id_complete['gdr3_dist1']))>1.5
        #flux_template = np.array(result['flux_template'])
        #center_star_mask = (x_template<np.max(x_template)-minimum_marginal_deviation)&(x_template>minimum_marginal_deviation)&(y_template<np.max(y_template)-minimum_marginal_deviation)&(y_template>minimum_marginal_deviation)
        distance, index = self.find_nearest_kdtree(x_template,y_template,x_template,y_template)
        dis_threshold = np.quantile(distance[:,1],dis_quantile_threshold)
        mask_positive = np.all(raw_flux_all>0,axis = 1)
        reference_star_mask = ((distance[:,1] > dis_threshold)&is_single_match&main_is_not_variable&mask_positive).astype(bool)
        reference_star_indices = np.where(reference_star_mask)
        reference_star_source_id = source_id_complete.loc[reference_star_indices,'source_id']
        # select the reference star according to iterative selection algorithm
        reference_star_source_id = [int(i) for i in reference_star_source_id]
        # sql_create = "CREATE TEMPORARY TABLE reference_star_this (source_id BIGINT);"#,  FOREIGN KEY (source_id) REFERENCES tianyu_source(source_id));"
        # sql_insert = "INSERT INTO reference_star_this (source_id) VALUES (%s);"
        # arg_insert = [(i,) for i in reference_star_source_id]
        # sql_query = "SELECT reference_star_this.source_id, tsp.template_img_id as template_img_id, tsp.x_template as x_template, tsp.y_template as y_template, tsp.flux_template as flux_template FROM reference_star_this INNER JOIN tianyu_source_position as tsp on reference_star_this.source_id = tsp.source_id ORDER BY reference_star_this.source_id;"
        # arg_query = ()
        # reference_star = self.sql_interface.querytemp(sql_create,sql_insert,arg_insert,sql_query,arg_query)
        # print(reference_star)
        #print(reference_star_mask)
        reference_star  = source_id_complete[np.squeeze(reference_star_mask)]
        
        raw_flux_reference_all = raw_flux_all[reference_star_indices]
        print(raw_flux_all,raw_flux_reference_all)
        reference_star['x_quantile'],bin_x = pd.qcut(reference_star['x_template'], pos_quantile_number, labels=False,retbins=True) 
        reference_star['y_quantile'],bin_y = pd.qcut(reference_star['y_template'], pos_quantile_number, labels=False,retbins=True)
        print(bin_x)
        print(bin_y)
        mask_reference_selected = []
        dict_pos_flux_bin = {}
        dict_reference_group = {}
        reference_group_index = 1
        for index_bin_x in range(len(bin_x)-1):
            for index_bin_y in range(len(bin_y)-1):
                mask_all_x = mask_select(x_template,bin_x,index_bin_x)
                mask_all_y = mask_select(y_template,bin_y,index_bin_y)
                mask_pos = mask_all_x & mask_all_y
                mask_reference_pos = mask_pos & reference_star_mask
                reference_star_this_pos_bin = source_id_complete[mask_reference_pos]
                num_reference_star_this_pos_bin = len(reference_star_this_pos_bin)
                print(len(reference_star_this_pos_bin),f'reference stars in this position; {np.sum(mask_pos)} stars in this position')
                n_bin_flux = min(num_reference_star_this_pos_bin//mean_number_reference,int(num_reference_star_this_pos_bin**0.5))
                quantile,bin_flux = pd.qcut(reference_star_this_pos_bin['flux_template'],n_bin_flux, labels=False,retbins=True)
                dict_pos_flux_bin[(index_bin_x,index_bin_y)] = bin_flux
                for index_bin_flux in range(len(bin_flux)-1):
                    dict_reference_group[(index_bin_x,index_bin_y,index_bin_flux)] = reference_group_index
                    reference_group_index+=1
                    mask_all_flux = mask_select(flux_template,bin_flux,index_bin_flux)
                    mask_all_this = (mask_all_flux & mask_pos).astype(bool)
                    mask_reference_this = (mask_reference_pos & mask_all_flux).astype(bool)
                    index_reference_this = np.where(mask_reference_this)[0]

                    mask_reference_subspace = mask_reference_this[mask_all_this]
                    index_mask_reference_subspace = np.where(mask_reference_subspace)[0]
                    n_reference_this_bin = np.sum(mask_reference_this)
                    star_index = np.squeeze(np.where(mask_all_this))
                    #print(mask_reference_subspace.shape,np.sum(mask_reference_this),np.sum(mask_all_this))
                    all_flux_this = raw_flux_all[star_index]
                    ref_flux_this = all_flux_this[mask_reference_subspace]
                    mask_reference_subspace_vert = mask_reference_subspace.reshape(-1,1)
                    weight_all = (np.sum(ref_flux_this, axis =0).reshape(1,-1)-mask_reference_subspace_vert*all_flux_this)/(n_reference_this_bin-mask_reference_subspace_vert)
                    flux_all_calibrated_this_bin = all_flux_this/weight_all
                    mean_flux_all_calibrated = np.mean(flux_all_calibrated_this_bin , axis = 1)
                    std_flux_all = np.std(flux_all_calibrated_this_bin , axis = 1)
                    SNR_baseline = np.sum(np.abs(mean_flux_all_calibrated)/std_flux_all**2)/np.sqrt(np.sum(1/std_flux_all**2))
                    print(f'SNR baseline = {SNR_baseline}')
                    #print('index_mask_subspace = \n',type(index_mask_reference_subspace))
                    mask_reference_selected.append(mask_reference_this.copy())
                    ct_excluded = 0
                    SNR_excluded_list = []
                    index_reference_chosen_this_list = []
                    for index_choose_reference,index_reference_chosen_this in zip(index_mask_reference_subspace,index_reference_this):
                        mask_reference_subspace_excluded = mask_reference_subspace.copy()
                        
                        mask_reference_subspace_excluded[index_choose_reference] = 0
                        #print(index_mask_reference_subspace)
                        ref_flux_excluded_this = all_flux_this[mask_reference_subspace_excluded]
                        mask_reference_subspace_excluded_vert = mask_reference_subspace_excluded.reshape(-1,1)
                        #print((n_reference_this_bin-1-mask_reference_subspace_excluded_vert))
                        weight_excluded_all = (np.sum(ref_flux_excluded_this, axis =0).reshape(1,-1)-mask_reference_subspace_excluded_vert*all_flux_this)/(n_reference_this_bin-1-mask_reference_subspace_excluded_vert)
                        flux_all_calibrated_this_excluded_bin = all_flux_this/weight_excluded_all
                        mean_flux_all_excluded_calibrated = np.mean(flux_all_calibrated_this_excluded_bin , axis = 1)
                        std_flux_excluded_all = np.std(flux_all_calibrated_this_excluded_bin , axis = 1)  
                        SNR_excluded = np.sum(mean_flux_all_excluded_calibrated/std_flux_excluded_all**2)/np.sqrt(np.sum(1/std_flux_excluded_all**2))     
                        print(f'Excluded SNR = {SNR_excluded}')   
                        SNR_excluded_list.append(SNR_excluded)
                        index_reference_chosen_this_list.append(index_reference_chosen_this)  
                        # if SNR_excluded>SNR_baseline:
                        #     mask_reference_selected[-1][index_reference_chosen_this] = 0
                        #     ct_excluded+=1
                    SNR_excluded_list = np.array(SNR_excluded_list)
                    index_reference_chosen_this_list = np.array(index_reference_chosen_this_list,dtype = int)
                    if np.sum(SNR_excluded_list<SNR_baseline)>=2:
                        ct_excluded = np.sum(SNR_excluded_list>=SNR_baseline)
                        mask_reference_selected[-1][index_reference_chosen_this_list[SNR_excluded_list>=SNR_baseline]] = 0
                    else:
                        mask_reference_selected[-1][index_reference_chosen_this_list[np.argsort(SNR_excluded_list)[2:]]] = 0
                        ct_excluded = len(index_mask_reference_subspace)-2
                    print(f'{ct_excluded} reference star excluded!')
        print('reference star selection complete!')
        mask_reference_selected = np.array(mask_reference_selected)
        #print(mask_reference_selected.shape)
        mask_reference = np.any(mask_reference_selected,axis = 0)
        mask_group = np.sum(mask_reference_selected*(np.arange(len(mask_reference_selected))+1).reshape(-1,1),axis = 0)
        #print(mask_group)

        reference_star_indices = np.where(mask_reference)
        reference_star_source_id = source_id_complete.loc[reference_star_indices,'source_id']
        reference_star_source_id = [int(i) for i in reference_star_source_id]
        reference_star_group = mask_group[mask_reference]
        reference_star_group = [int(i) for i in reference_star_group]
        sql_create = "CREATE TEMPORARY TABLE reference_star_this (source_id BIGINT);"#,  FOREIGN KEY (source_id) REFERENCES tianyu_source(source_id));"
        sql_insert = "INSERT INTO reference_star_this (source_id) VALUES (%s);"
        arg_insert = [(i,) for i in reference_star_source_id]
        sql_query = "SELECT tsp.source_id, tsp.x_template AS x_template, tsp.y_template AS y_template, tsp.flux_template AS flux_template,IF(rst.source_id IS NULL, False,True) as is_reference FROM tianyu_source_position as tsp LEFT JOIN reference_star_this as rst ON rst.source_id = tsp.source_id  WHERE template_img_id = %s ;"
        args = (int(temp_image_id),)
        all_star = self.sql_interface.querytemp(sql_create,sql_insert,arg_insert,sql_query,args)
        print(np.sum(all_star['is_reference']))
        print('total number of star is',len(all_star))
        reference_star_mask = np.array(all_star['is_reference']).astype(bool)
        source_id_list = np.array(all_star['source_id']).astype(int)
        x_template = np.array(all_star['x_template']).astype(np.float64)
        y_template = np.array(all_star['y_template']).astype(np.float64)
        flux_template = np.array(all_star['flux_template']).astype(np.float64)
        print('number of stars is ',x_template.shape,np.sum(1-np.isnan(x_template)))
        
        # calculate the relative flux of all stars, regardless whether it is fully observed
        # Obtain the raw flux of all stars
        sql = "SELECT * FROM img where obs_id = %s and align_process_id = align_process_id order by jd_utc_mid;"
        args = (obs_id,)
        images_df = self.sql_interface.query(sql,args)
        image_id_list = np.array(images_df['image_id']).astype(int)
        image_id_index_dict = {}
        for i in range(len(image_id_list)):
            image_id_index_dict[image_id_list[i]] = i
        source_id_index_dict = {}
        for i in range(len(source_id_list)):
            source_id_index_dict[source_id_list[i]] = i
        N_star = len(all_star)
        N_epoch = len(images_df)

        flux_array = np.full((N_star,N_epoch), np.nan,np.float64)
        flux_error_array = np.full((N_star,N_epoch), np.nan,np.float64)
        flux_id = np.full((N_star,N_epoch), np.nan,int)
        cal_flux_array = np.full((N_star,N_epoch), np.nan,np.float64)
        cal_flux_error_array = np.full((N_star,N_epoch), np.nan,np.float64)
        cal_group_array = np.full((N_star,N_epoch), 0,int)
        print(flux_array.shape)
        print('download raw flux')
        sql = "SELECT spi.star_pixel_img_id, img.image_id as image_id, spi.source_id as source_id, flux_raw, flux_raw_error FROM star_pixel_img as spi INNER JOIN img ON img.image_id = spi.image_id  where img.obs_id = %s order by source_id, img.jd_utc_mid;"
        args = (obs_id,)
        raw_flux = self.sql_interface.query(sql,args)
        print('loading flux')

        for i,j in raw_flux.iterrows():
            flux_id[source_id_index_dict[j['source_id']],image_id_index_dict[j['image_id']]] = int(j['star_pixel_img_id'])
            flux_array[source_id_index_dict[j['source_id']],image_id_index_dict[j['image_id']]] = float(j['flux_raw'])
            flux_error_array[source_id_index_dict[j['source_id']],image_id_index_dict[j['image_id']]] = float(j['flux_raw_error'])
        print(np.sum(1-np.isnan(flux_array)))
        print(len(raw_flux))
        filled_flux = np.nan_to_num(flux_array)
        print('load flux complete')
        s_star = 0
        loc_star = 0
        used_group = []
        for index_bin_x in range(len(bin_x)-1):
            for index_bin_y in range(len(bin_y)-1):
                mask_x = mask_select(x_template,bin_x,index_bin_x)
                mask_y = mask_select(y_template,bin_y,index_bin_y)
                loc_star += np.sum(mask_x & mask_y)
                bin_flux = dict_pos_flux_bin[(index_bin_x,index_bin_y)]
                for index_bin_flux in range(len(bin_flux)-1):
                    reference_group_index_this = dict_reference_group[(index_bin_x,index_bin_y,index_bin_flux)]
                    used_group.append(reference_group_index_this)
                    mask_flux = mask_select(flux_template,bin_flux,index_bin_flux)
                    mask_this_bin = (mask_x & mask_y & mask_flux).astype(bool)
                    ind_this = np.where(mask_this_bin)
                    s_star += np.sum(mask_this_bin)
                    
                    mask_reference_this = (reference_star_mask & mask_this_bin).astype(bool)
                    mask_reference_subspace = mask_reference_this[mask_this_bin].astype(bool)
                    index_mask_reference_subspace = np.where(mask_reference_subspace)[0]
                    
                    
                    N_ref = np.sum(mask_reference_this)
                    filled_flux_this = filled_flux[ind_this]
                    mask_reference_subspace_vert = mask_reference_subspace.reshape(-1,1)
                    #print(flux_array[ind_this])
                    weight = (np.sum(filled_flux_this[mask_reference_subspace],axis = 0).reshape(1,-1)-mask_reference_subspace_vert*filled_flux_this)/(N_ref-mask_reference_subspace_vert)
                    #print('weight = ',weight)
                    cal_flux_array[ind_this] = flux_array[ind_this].copy()/weight
                    cal_flux_error_array[ind_this] = flux_error_array[ind_this].copy()/weight
                    cal_group_array[ind_this] = reference_group_index_this
        # print(s_star)
        # print(np.sum(1-np.isnan(cal_flux_array)))
        # print(cal_group_array)
        # print(set(range(1,reference_group_index))-set(used_group))
        # print(loc_star)
                    



        




        # sql = "SELECT * FROM img where image_id = %s;"
        # args = (int(image_id),)
        # result = self.sql_interface.query(sql,args)
        # obs_id = int(result.loc[0,'obs_id'])
        # # Delete the old reference star
        print('registering reference star')
        sql = "DELETE FROM reference_star where obs_id = %s;"
        args = (obs_id,)
        self.sql_interface.execute(sql,args)

        sql = "INSERT INTO reference_star (obs_id, reference_group,source_id, process_id) VALUES (%s,%s,%s,%s);"
        args = [(int(obs_id),int(reference_star_group[i]),int(reference_star_source_id[i]),int(PID)) for i in range(len(reference_star_source_id))]
        #print(len(args))
        self.sql_interface.executemany(sql,args)
        # print(reference_star_source_id,'\n',obs_id)
        print(sql,args)
        print('generating flux sql')
        sql = "UPDATE star_pixel_img SET flux_relative = %s,flux_relative_error = %s,reference_group=%s, relative_process_id = %s where star_pixel_img_id = %s; "
        
        mask = np.where(~np.isnan(cal_flux_array))
        cal_flux = np.squeeze(cal_flux_array[mask])
        cal_flux_error = np.squeeze(cal_flux_error_array[mask])
        cal_group = np.squeeze(cal_group_array[mask])
        starpixelids = np.squeeze(flux_id[mask])
        print('inserting relative flux into sql')
        args = [[float(cal_flux[i]),float(cal_flux_error[i]),int(cal_group[i]),int(PID),int(starpixelids[i])] for i in range(len(cal_flux))]
        #self.sql_interface.executemany(sql,args)
        batch_size = 4000
        self.sql_interface.execute('SET foreign_key_checks = 0;',[])
        for i in tqdm.tqdm(range(0, len(args), batch_size)):
            print(f'{i}th batch')
            batch_args = args[i:i+batch_size]
            self.sql_interface.executemany(sql, batch_args)
        
        self.sql_interface.execute('SET foreign_key_checks = 1;',[])
        return 1
    
    def relative_photometric_calibration(self,PID,PID_reference,PID_flux_extraction,flux_quantile_number = 5,pos_quantile_number = 3):
        def mask_select(df,item,bins,index):
            if index ==0:
                mask =  df[item]<=bins[1]
                return mask
            if index == len(bins)-1:
                mask = df[item]>bins[-2]
                return mask
            mask = (df[item]>bins[index]) & (df[item]<=bins[index+1])
            return mask
        # load database
        sql = "SELECT reference_star.source_id as source_id, tsp.template_img_id as template_img_id, tsp.x_template as x_template, tsp.y_template as y_template, tsp.flux_template as flux_template FROM reference_star INNER JOIN tianyu_source_position as tsp on reference_star.source_id = tsp.source_id WHERE reference_star.process_id = %s;"
        args = (PID_reference,)
        reference_star = self.sql_interface.query(sql,args)

        #print(reference_star.columns)
        template_id = int(reference_star.loc[0,'template_img_id'])
        sql = "SELECT * FROM tianyu_source_position WHERE template_img_id = %s;"
        args = (template_id,)
        all_star = self.sql_interface.query(sql,args)
        #reference_star_source_id = np.array(reference_star['source_id'],dtype = int)
        sql = "SELECT spi.star_pixel_img_id as star_pixel_img_id, spi.source_id as source_id, spi.flux_raw as flux_raw, spi.flux_raw_error as flux_raw_error, tsp.x_template as x_template, tsp.y_template as y_template, tsp.flux_template as flux_template  FROM star_pixel_img as spi INNER JOIN tianyu_source_position as tsp on spi.source_id = tsp.source_id where spi.birth_process_id = %s;"
        args = (PID_flux_extraction,)
        raw_flux = self.sql_interface.query(sql,args)
        # Group reference stars by quantile of flux
        all_star['flux_quantile'],bin_flux = pd.qcut(all_star['flux_template'], flux_quantile_number, labels=False,retbins=True)
        all_star['x_quantile'],bin_x = pd.qcut(all_star['x_template'], pos_quantile_number, labels=False,retbins=True) 
        all_star['y_quantile'],bin_y = pd.qcut(all_star['y_template'], pos_quantile_number, labels=False,retbins=True)

        # print(reference_star)
        # print(all_star)
        # print(raw_flux)
        # print(raw_flux.columns)
        args_calibration = []
        
        for index_flux in range(len(bin_flux)-1):
            for index_bin_x in range(len(bin_x)-1):
                for index_bin_y in range(len(bin_y)-1):
                    mask_flux_reference = mask_select(reference_star,'flux_template',bin_flux,index_flux)
                    mask_x_reference = mask_select(reference_star,'x_template',bin_x,index_bin_x)
                    mask_y_reference = mask_select(reference_star,'y_template',bin_y,index_bin_y)
                    mask_reference = mask_flux_reference & mask_x_reference & mask_y_reference
                    reference_star_this_bin = reference_star[mask_reference]
                    mask_flux_raw = mask_select(raw_flux,'flux_template',bin_flux,index_flux)
                    mask_x_raw = mask_select(raw_flux,'x_template',bin_x,index_bin_x)
                    mask_y_raw = mask_select(raw_flux,'y_template',bin_y,index_bin_y)
                    mask_raw = mask_flux_raw & mask_x_raw & mask_y_raw
                    raw_flux_this_bin = raw_flux[mask_raw]
                    # print(raw_flux_this_bin,'\n',reference_star_this_bin)
                    reference_flux = pd.merge(raw_flux_this_bin,reference_star_this_bin,how = 'inner',on = 'source_id')
                    reference_flux.rename(columns={'flux_raw': 'flux_raw_reference', 'flux_raw_error': 'flux_raw_error_reference','star_pixel_img_id':'pixel_id'},inplace = True)
                    #print(reference_flux.columns)
                    # print(reference_flux)
                    join_flux = pd.merge(raw_flux_this_bin,reference_flux,how = 'left',on = 'source_id')
                    
                    join_flux_filled = join_flux.fillna(0)
                    is_reference = ~join_flux['flux_raw_reference'].isna()
                    num_reference = len(reference_flux)
                    #print('num_reference = ',num_reference,'at',index_flux,index_bin_x,index_bin_y)
                    average_reference = (np.sum(join_flux_filled['flux_raw_reference'])-np.array(join_flux_filled['flux_raw_reference']))/(num_reference-np.array(is_reference,dtype = int))
                    join_flux['relative_flux'] = join_flux['flux_raw']/average_reference
                    join_flux['relative_flux_error'] = join_flux['flux_raw_error']/average_reference
                    #print(join_flux)
                    for i,r in join_flux.iterrows():
                        args_calibration.append((float(r['relative_flux']),float(r['relative_flux_error']),PID,int(r['star_pixel_img_id'])))
        
        sql = "UPDATE star_pixel_img SET flux_relative = %s,flux_relative_error = %s, relative_process_id = %s where star_pixel_img_id = %s; "
        self.sql_interface.executemany(sql,args_calibration)
        return 1
                    # print(~join_flux['flux_raw_reference'].isna())
                    # print(join_flux['flux_raw'])
                    # print(reference_flux)
                    # plt.scatter(raw_flux_this_bin['x_template'],raw_flux_this_bin['y_template'])
                    # plt.scatter(reference_flux['x_template_x'],reference_flux['y_template_x'])
                    # plt.show()
                    #eference_star_this_bin = reference_star[(reference_star['flux_template']>=bin_flux[index_flux]-epsilon)&(reference_star['flux_template']<bin_flux[index_flux+1])&(reference_star['flux_template']>=bin_flux[index_flux]-epsilon)&(reference_star['flux_template']<bin_flux[index_flux+1])]
        # print(all_star)
        # print(raw_flux)
        # print(bin_flux)
        # print(bin_x)
        # print(bin_y)


        