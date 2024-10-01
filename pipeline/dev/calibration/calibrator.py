import os
import pathlib
import mysql.connector
import numpy as np
from astropy.io import fits
import glob
from tqdm import tqdm
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
        solver = astrometry.Solver(
            astrometry.series_4100.index_files(
                cache_directory=dir_data,
                scales=[7,8,9],
            )
        )
        stars_cm = np.hstack([np.array(archive_star_result['x_template']).reshape(-1,1),np.array(archive_star_result['y_template']).reshape(-1,1)])
        print('resolving astrometry using astrometry.net')
        print(stars_cm)
        solution = solver.solve(
                    stars_xs=np.array(archive_star_result['x_template'])[:70],
                    stars_ys=np.array(archive_star_result['y_template'])[:70],
                    size_hint=astrometry.SizeHint(
                        lower_arcsec_per_pixel=0.2,
                        upper_arcsec_per_pixel=0.4,
                    ),
                    position_hint=astrometry.PositionHint(
                ra_deg=sky_result.loc[0,'ra'],
                dec_deg=sky_result.loc[0,'dec'],
                radius_deg=1.0,
            ),
            solve_id = None,
            tune_up_logodds_threshold = np.log(1e6),
            output_logodds_threshold = np.log(1e9)
        )
        print(f"{solution.best_match().center_ra_deg=}")
        print(f"{solution.best_match().center_dec_deg=}")
        print(f"{solution.best_match().scale_arcsec_per_pixel=}")
        print('searching gdr3 targets')
        Gaia_query_res = self.dl.search_GDR3_by_square(ra = solution.best_match().center_ra_deg,dec = solution.best_match().center_dec_deg, fov = 0.1+(sky_result.loc[0,'fov_x']**2+sky_result.loc[0,'fov_y']**2)**0.5/2,Gmag_limit = 20)
        print(Gaia_query_res)
        if solution.has_match():
            wcs = astropy.wcs.WCS(solution.best_match().wcs_fields)
            pixels = wcs.all_world2pix(
                np.hstack([Gaia_query_res['ra'].reshape(-1,1), Gaia_query_res['dec'].reshape(-1,1)]),
                0,
            )
        else:
            return 0
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
        for i,r in archive_star_result.iterrows():
            source_id_this = int(r['source_id'])
            sql = "SELECT * FROM source_crossmatch WHERE source_id = %s;"
            args = (source_id_this,)
            result = self.sql_interface.query(sql,args)
            update = len(result)
            if update:
                sql = "UPDATE source_crossmatch SET gdr3_id1 = %s,gdr3_id2 = %s,gdr3_id3 = %s, gdr3_dist1 = %s,gdr3_dist2 = %s,gdr3_dist3 = %s WHERE source_id = %s"
            else:
                sql = "INSERT INTO source_crossmatch (gdr3_id1,gdr3_id2,gdr3_id3,gdr3_dist1,gdr3_dist2,gdr3_dist3,source_id) VALUES (%s,%s,%s,%s,%s,%s,%s)"
            args = (int(Gaia_query_res['SOURCE_ID'][indices[i,0]]),int(Gaia_query_res['SOURCE_ID'][indices[i,1]]),int(Gaia_query_res['SOURCE_ID'][indices[i,2]]),float(distances[i,0]),float(distances[i,1]),float(distances[i,2]),int(source_id_this))
            print(args)
            self.sql_interface.execute(sql,args)
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
    def absolute_photometric_calibration(self):
        pass
    def choose_reference_star(self,PID,template_generation_PID,dis_quantile_threshold = 0.5, minimum_marginal_deviation = 200):
        # load database, find best reference star
        sql = "SELECT * FROM sky_image_map WHERE process_id = %s;"
        args = (template_generation_PID,)
        result = self.sql_interface.query(sql,args)
        assert len(result)==1
        image_id = result.loc[0,'image_id']
        sql = "SELECT * FROM tianyu_source_position WHERE template_img_id = %s;"
        args = (int(image_id),)
        result = self.sql_interface.query(sql,args)
        x_template = np.array(result['x_template'])
        y_template = np.array(result['y_template'])
        flux_template = np.array(result['flux_template'])
        center_star_mask = (x_template<np.max(x_template)-minimum_marginal_deviation)&(x_template>minimum_marginal_deviation)&(y_template<np.max(y_template)-minimum_marginal_deviation)&(y_template>minimum_marginal_deviation)
        distance, index = self.find_nearest_kdtree(x_template,y_template,x_template,y_template)
        dis_threshold = np.quantile(distance[:,1],dis_quantile_threshold)
        reference_star_indices = np.where((distance[:,1] < dis_threshold)&center_star_mask)
        reference_star_source_id = result.loc[reference_star_indices,'source_id']
        sql = "SELECT * FROM img where image_id = %s;"
        args = (int(image_id),)
        result = self.sql_interface.query(sql,args)
        obs_id = result.loc[0,'obs_id']
        sql = "INSERT INTO reference_star (obs_id, source_id, process_id) VALUES (%s,%s,%s);"
        args = [(obs_id,int(i),PID) for i in reference_star_source_id]
        self.sql_interface.executemany(sql,args)
        # print(reference_star_source_id,'\n',obs_id)
        # print(sql,args)
        #return {'reference_id':reference_star_source_id,"reference_x":x_template[reference_star_indices],"reference_y":y_template[reference_star_indices]}

        return 1

        
    def relative_photometric_calibration(self,PID,PID_reference,PID_flux_extraction,flux_quantile_number = 5,pos_quantile_number = 3):
        
        # load database
        sql = "SELECT * FROM reference_star INNER JOIN tianyu_source_position as tsp on reference_star.source_id = tsp.source_id WHERE process_id = %s;"
        args = (PID_reference,)
        reference_star = self.sql_interface.query(sql,args)
        #
        template_id = int(reference_star.loc[0,'template_img_id'])
        sql = "SELECT * FROM tianyu_source_position WHERE template_img_id = %s;"
        args = (template_id,)
        all_star = self.sql_interface.query(sql,args)
        #reference_star_source_id = np.array(reference_star['source_id'],dtype = int)
        sql = "SELECT * FROM star_pixel_img as spi INNER JOIN tianyu_source_position as tsp on spi.source_id = tsp.source_id where spi.birth_process_id = %s;"
        args = (PID_flux_extraction,)
        raw_flux = self.sql_interface.query(sql,args)
        # Group reference stars by quantile of flux
        reference_star['flux_quantile'] = pd.qcut(reference_star['flux_template'], flux_quantile_number, labels=False)
        raw_flux['flux_quantile'] = pd.qcut(raw_flux['flux_template'], flux_quantile_number, labels=False) 
        raw_flux['pos_quantile'] = pd.qcut(raw_flux['x_template'], pos_quantile_number, labels=False)


        