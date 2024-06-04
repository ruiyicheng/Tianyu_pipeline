import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from astropy.io import fits
sys.path.append("/home/yichengrui/workspace/TianYu/pipeline/scheduling/")
import data_loader as dl
sys.path.append("/home/yichengrui/workspace/TianYu/pipeline/template_generating")
import image_alignment as il

class flux_extractor:
    def __init__(self,Generate = False):
        self.dl = dl.data_loader()
        self.al = il.alignment()
        self.circ_mask_path = '/home/yichengrui/workspace/TianYu/pipeline/image_process/circ_mask/'
        if Generate:
            self.generate_mask()
        else:
            self.read_mask()
        #print(self.mask_dict)
    # def extract_noise(self,img,star_pos,star_aper_rad,sky_inner_aper_rad,sky_outer_aper_rad):
    #     pass
    def generate_mask(self):
        self.mask_dict = {}
        for mask_size in range(3,102):
            #temp_arr = np.zeros((mask_size,mask_size))
            x_coord = np.ones((mask_size,mask_size))*(np.arange(mask_size)).reshape(1,-1)
            y_coord = np.ones((mask_size,mask_size))*(np.arange(mask_size)).reshape(-1,1)
            cent_coord = (mask_size-1)/2
            mask = (((x_coord-cent_coord)**2+(y_coord-cent_coord)**2)<cent_coord**2+0.3)
            self.mask_dict[mask_size] = mask
            np.save(self.circ_mask_path+'D_'+str(mask_size)+'_bin_mask.npy',mask)



    def read_mask(self):
        self.mask_dict = {}
        for mask_size in range(3,102):
            self.mask_dict[mask_size] = np.load(self.circ_mask_path+'D_'+str(mask_size)+'_bin_mask.npy')

    def generate_ring_musk(self,r1,r2):
        if r1 % 2 != r2 % 2 or r1<3 or r2<3 or r1>101 or r2>101:
            print("invalid parameter for generating ring")
            return -1
        else:
            inner_rad = np.min([r1,r2])
            outer_rad = np.max([r1,r2])
            outer_mask = self.mask_dict[outer_rad]
            inner_mask = self.mask_dict[inner_rad]
            din = inner_mask.shape[0]
            dout = outer_mask.shape[0]
            d0 = (dout-din)//2
            inner_mask_pad = np.zeros(outer_mask.shape)
            inner_mask_pad[d0:d0+din,d0:d0+din] = inner_mask
            ring = outer_mask&np.logical_not(inner_mask_pad)
            return ring
    def show_cut_pixel(self,img,star_pos,star_aper_rad,sky_inner_aper_rad,sky_outer_aper_rad,savepath):
        ring_sky = self.generate_ring_musk(sky_inner_aper_rad,sky_outer_aper_rad)
        cut_r = int(ring_sky.shape[0]//2)
        residue_r = int(ring_sky.shape[0]%2)
        star_mask = self.mask_dict[star_aper_rad]
        din = star_mask.shape[0]
        dout = ring_sky.shape[0]
        jitter = not dout%2 and din%2
        d0 = (dout-din)//2+jitter
        star_mask_pad = np.zeros(ring_sky.shape,dtype = bool)
        star_mask_pad[d0:d0+din,d0:d0+din] = star_mask
        inv_whole_mask_pad = np.logical_not(star_mask_pad|ring_sky)
        #img[int(star_pos_single[1]-0.5)][int(star_pos_single[0]-0.5)] = 0
        cut_pix = img[int(star_pos[1]-0.5)-cut_r:int(star_pos[1]-0.5)+cut_r+residue_r,int(star_pos[0]-0.5)-cut_r:int(star_pos[0]-0.5)+cut_r+residue_r]
        star_ma = np.ma.masked_array(cut_pix,mask= inv_whole_mask_pad)
        plt.figure()
        plt.imshow(star_ma)
        plt.savefig(savepath)
    def show_sky_star(self,img,star,high = 1500,low =1200,r = 15):
        plt.figure()
        extent = (0,img.shape[1],0,img.shape[0])
        #print(BG)
        plt.figure(figsize = (40,20))
        plt.imshow(img,vmin = low, vmax = high,extent=extent, origin='lower')
        t = np.linspace(0,2*np.pi,100)
        for s in star:
            plt.plot(r*np.cos(t)+s[0],r*np.sin(t)+s[1],'r')
        plt.savefig('result.pdf')
    def flux_extract(self,img,star_pos,star_aper_rad,sky_inner_aper_rad,sky_outer_aper_rad,device = "cpu",saturate_threshold = 60000):
        if device=="cpu":
            ring_sky = self.generate_ring_musk(sky_inner_aper_rad,sky_outer_aper_rad)
            inv_ring_sky = np.logical_not(ring_sky)
            cut_r = int(ring_sky.shape[0]//2)
            residue_r = int(ring_sky.shape[0]%2)
            #print(ring_sky.shape)
            fixed_rad = False
            if type(star_aper_rad)==int:
                print('Fixed_rad!')
                star_aper_rad = np.ones(len(star_pos),dtype = int)*star_aper_rad
                fixed_rad = True
            star_mask_calculated = False
            flux_list = []
            saturate_list = []
            err_list = []
            for i,star_pos_single in enumerate(star_pos):
                if not fixed_rad or not star_mask_calculated:
                    
                    star_mask = self.mask_dict[star_aper_rad[i]]
                    din = star_mask.shape[0]
                    dout = ring_sky.shape[0]
                    jitter = not dout%2 and din%2
                    d0 = (dout-din)//2+jitter
                    star_mask_pad = np.zeros(ring_sky.shape)
                    star_mask_pad[d0:d0+din,d0:d0+din] = star_mask
                    inv_star_mask_pad = np.logical_not(star_mask_pad)
                    star_pix_n = np.sum(star_mask)
                    star_mask_calculated = True

                #img[int(star_pos_single[1]-0.5)][int(star_pos_single[0]-0.5)] = 0
                cut_pix = img[int(star_pos_single[1]-0.5)-cut_r:int(star_pos_single[1]-0.5)+cut_r+residue_r,int(star_pos_single[0]-0.5)-cut_r:int(star_pos_single[0]-0.5)+cut_r+residue_r]
                saturate_list.append(np.max(cut_pix)>saturate_threshold)
                sky_ma = np.ma.masked_array(cut_pix,mask= inv_ring_sky)
                star_ma = np.ma.masked_array(cut_pix,mask= inv_star_mask_pad)
                sky_low,sky_mid,sky_high = np.nanpercentile(sky_ma.filled(np.nan), (15.87,50,84.13))
                sky_std = (sky_high-sky_low)/2
                sigma_sky = sky_std**2
                sky_err_sq = sigma_sky*np.sum(star_pix_n)
                star_flux = np.ma.sum(star_ma-sky_mid)
                flux_err = np.sqrt(star_flux+sky_err_sq)
                flux_list.append(star_flux)
                err_list.append(flux_err)
            flux_list = np.array(flux_list)
            err_list = np.array(err_list)
            saturate_list = np.array(saturate_list)
            # plt.imshow(np.ma.masked_array(cut_pix,mask= inv_star_mask_pad))
            # plt.imshow(np.ma.masked_array(cut_pix,mask= inv_ring_sky))
            # # print(cut_pix.shape)
            # # plt.figure()
            # # plt.imshow(cut_pix)
            # plt.savefig('pixel.pdf')
            #break
            return flux_list,err_list,saturate_list
                
    def relative_flux_extraction(self,flux,e_flux,is_reference):
        num_reference = np.sum(is_reference)
        reference_flux = np.ones(flux.shape)*np.mean(flux[is_reference])
        reference_flux[is_reference] = (num_reference*reference_flux[is_reference]-flux[is_reference])/(num_reference-1)
        flux_relative = flux/reference_flux
        e_flux_relative = e_flux/reference_flux
        return flux_relative,e_flux_relative



    def light_curve_extraction(self,sky_id,obs_id,hierarchy,save_raw_pix = False,alignment_img_type="science_processed",aligned = False,sky_boundary = 30,record = True):
        #obs_id->image->Sky->star
        #                  ->template_img_id
        #
        # Function: alignment+choose reference star+extraction
        #

        
        sql = "SELECT * FROM sky where sky_id = %s;"
        args = (sky_id,)
        sky_df = self.dl.query(sql,args)

        sky_template_id = (int(sky_df['template_image_id'].iloc[0]),)
        sql = "SELECT * from img where image_id = %s;"
        sky_img_df = self.dl.query(sql,sky_template_id)
        template_img = fits.getdata(sky_img_df['img_path'].iloc[0])

        if not aligned:
            alignment_res = self.al.get_deviation(sky_template_id[0],obs_id,hierarchy,alignment_img_type)
        sql = "SELECT * FROM img where obs_id = %s and hierarchy = %s and image_type_id = %s;"
        args = (obs_id,hierarchy,self.dl.image_type_id[alignment_img_type])
        image_info_df = self.dl.query(sql,args)

        sql = "SELECT * FROM star where sky_id = %s;"
        args = (sky_id,)
        star_template = self.dl.query(sql,args)
        template_coordinate = np.array(star_template[["x_template","y_template"]],dtype = float)
        star_id = np.squeeze(np.array(star_template["star_id"],dtype = int))
        tmid = np.squeeze(np.array(image_info_df["jd_utc_mid"],dtype = float))
        dx = np.squeeze(np.array(image_info_df["x_to_template"],dtype = int))
        dy = np.squeeze(np.array(image_info_df["y_to_template"],dtype = int))
        saturate = np.squeeze(np.array(star_template["saturate"],dtype = bool))
        variability = np.squeeze(np.array(star_template["variability"],dtype = bool))
        flux_deviation = np.squeeze(np.array(star_template["flux_deviation"],dtype = float))
        apparature_diameter = np.squeeze(np.array(star_template["apparature_diameter"],dtype = float))
        reference_class = np.squeeze(np.array(star_template["reference_class"],dtype = int))
        num_class = np.max(reference_class)+1
        is_reference = ((np.squeeze(template_coordinate[:,0])-np.min(dx))<(template_img.shape[1]-sky_boundary))&((np.squeeze(template_coordinate[:,0])-np.max(dx))>(sky_boundary))&((np.squeeze(template_coordinate[:,1])-np.min(dy))<(template_img.shape[0]-sky_boundary))&((np.squeeze(template_coordinate[:,1])-np.max(dy))>(sky_boundary))&np.logical_not(saturate)&(np.abs(flux_deviation)<5)&np.logical_not(variability)
        print(is_reference)
        #reference_star_coord = template_coordinate[is_reference]

        #self.show_sky_star(template_img,reference_star_coord)
        ct = 1
        args_list = []
        # result_dict = {}
        # for i,r in star_template.iterrows():
        #     result_dict[int(r['star_id'])] = {"image_id":[]}
        for _,r in image_info_df.iterrows():
            print('processing frame',ct)
            ct +=1
            image = fits.getdata(r['img_path'])
            star_pix = np.zeros(template_coordinate.shape)
            
            dx = int(r['x_to_template'])
            dy = int(r['x_to_template'])
            star_pix[:,0] = template_coordinate[:,0]-dx
            star_pix[:,1] = template_coordinate[:,1]-dy
            extract_in_this_frame = ((np.squeeze(template_coordinate[:,0])-dx)<(template_img.shape[1]-sky_boundary))&((np.squeeze(template_coordinate[:,0])-dx)>(sky_boundary))&((np.squeeze(template_coordinate[:,1])-dy)<(template_img.shape[0]-sky_boundary))&((np.squeeze(template_coordinate[:,1])-dy)>(sky_boundary))
            for cn in range(num_class):
                this_class = reference_class==cn
                reference_this_frame = is_reference[extract_in_this_frame&this_class]
                star_id_this_frame = star_id[extract_in_this_frame&this_class]
                flux,err,_ = self.flux_extract(image,star_pix[extract_in_this_frame&this_class],apparature_diameter[extract_in_this_frame&this_class],25,29)
                flux_relative,e_flux_relative = self.relative_flux_extraction(flux,err,reference_this_frame)
                for i in range(len(star_id_this_frame)):
                    if not save_raw_pix:
                        args_list.append([int(star_id_this_frame[i]),int(r['image_id']),float(flux[i]),float(err[i]),float(flux_relative[i]),float(e_flux_relative[i]),bool(reference_this_frame[i])])
            #print(flux_relative)

        # plot target
        if record:
            mycursor = self.dl.cnx.cursor()
            if not save_raw_pix:
                sql = "INSERT INTO star_pixel_img (star_id,image_id,flux_raw,flux_raw_error,flux_calibrated,flux_calibrated_error,is_reference) VALUES (%s,%s,%s,%s,%s,%s,%s);"
                for arg in args_list:
                    mycursor.execute(sql,arg)
                self.dl.cnx.commit()

        else:
            flux_l = []
            image_id_list = []
            for arg in args_list:
                if arg[0]==655:
                    flux_l.append(arg[4])
                    image_id_list.append(arg[1])
            alignment_res = np.array(alignment_res)
            pix_num = alignment_res[:,4]
            flux_l = np.array(flux_l)
            pix_num = np.array(pix_num)
            #print(image_id_list[(flux_l/np.median(flux_l)*np.median(pix_num))>0.7])
            plt.figure()
            plt.plot(tmid,flux_l/np.median(flux_l)*np.median(pix_num),'.b')
            plt.plot(tmid,pix_num,'-r')
            plt.xlabel('jd-utc')
            plt.ylabel('flux')
            plt.savefig('hat-p-20.pdf')
            np.save("num_star_pix.npy",pix_num)
            np.save("flux.npy",flux_l)
            np.save("tmid.npy",tmid)
        





        print(len(args_list))






    
if __name__=="__main__":
    from astropy.io import fits
    fe = flux_extractor(Generate = False)
    # r = fe.generate_ring_musk(14,20)
    # star_apps = list(range(3,25))
    # snr_list = []
    # for app in star_apps:
    #     flux,err = fe.flux_extract(fits.getdata('/home/share/muguang/image/frame/2024-02-16/HAT-P-20-0600_corrected_at_1961808571370754644.fits'),[[1642.4832,1068.5398 ]],app,25,29)
    #     #flux,err = fe.flux_extract(fits.getdata('/home/yichengrui/workspace/TianYu/pipeline/image_process/out/reduced_res/stack_img/template_obs_3_hierarchy_1_imgtype_science_processed_1083345510022026818.fit'),[[1642.4832,1068.5398 ]],app,25,29)
    #     snr_list.append(flux/err)
    # print(star_apps[np.argmax(snr_list)])
    # plt.plot(star_apps,snr_list,'.')
    # plt.savefig('app-snr.pdf')
    fe.light_curve_extraction(1,3,1,aligned=False,record = False)

    # mask_arr = np.ma.masked_array(np.ones(r.shape),mask = np.logical_not(r))
    # # print(np.ma.sum(np.ma.masked_array(np.ones(r.shape),mask = np.logical_not(r))))
    # # print(np.sum(r))
    # plt.figure()
    # plt.imshow(mask_arr*np.arange(r.shape[0]))
    # plt.savefig('ring.pdf')

