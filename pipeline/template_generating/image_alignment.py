from astropy.io import fits
import sys
import numpy as np
sys.path.append('/home/yichengrui/workspace/TianYu/pipeline/scheduling/')
sys.path.append('/home/yichengrui/workspace/TianYu/pipeline/image_process/')
from use_SE import SE
import data_loader as dl
import time
import os

class alignment:
    def __init__(self):
        self.dl = dl.data_loader()
        self.se = SE()
    def get_deviation(self,template_img_id,alignment_img_obs_id, alignment_img_hierarchy, alignment_img_type,good_star_threshold = 2):

        mycursor = self.dl.cnx.cursor()
        mycursor.execute("SELECT i.obs_id,i.img_path from img as i where i.image_id = "+str(template_img_id)+';')
        myresult_template = mycursor.fetchall()
        print(myresult_template)
        if len(myresult_template)!=1:
            print("Cannot fing image")
            return -1
        

        template_output = "/home/yichengrui/workspace/TianYu/pipeline/template_generating/template_"+str(hash(time.time()))+".fit"
        fits_data = self.se.use(myresult_template[0][1],template_output,keep_out = False)
        #fits_res = fits.open(template_output)
        # print(fits_res[2].header)
        #print(fits_res[2].data['X_IMAGE'])
        #os.system("rm "+template_output)
        x_stars_template = np.squeeze(fits_data['X_IMAGE'])
        y_stars_template = np.squeeze(fits_data['Y_IMAGE'])



        # flag_star = np.squeeze(fits_res[2].data['CLASS_STAR'])
        # Y_max = np.squeeze(fits_res[2].data['YMAX_IMAGE'])
        # BG = np.squeeze(fits_res[2].data['BACKGROUND'])


        mycursor = self.dl.cnx.cursor()
        arg = (alignment_img_obs_id,alignment_img_hierarchy,self.dl.image_type_id[alignment_img_type])
        sql = "SELECT i.obs_id,i.img_path,i.image_id from img as i where i.obs_id = %s and hierarchy = %s and i.image_type_id = %s;"
        mycursor.execute(sql,arg)
        myresult = mycursor.fetchall()
        res_list = []
        for res in myresult:
            #print(res)
            res_output = "/home/yichengrui/workspace/TianYu/pipeline/template_generating/res_"+str(hash(time.time()))+".fit"
            fits_data = self.se.use(res[1],res_output,keep_out = False)
            #fits_res = fits.open(res_output)
            #os.system("rm "+res_output)
            x_stars_this = np.squeeze(fits_data['X_IMAGE'])
            y_stars_this = np.squeeze(fits_data['Y_IMAGE'])
            xx_stars = np.squeeze(fits_data['X2_IMAGE']).reshape(-1,1)
            yy_stars = np.squeeze(fits_data['Y2_IMAGE']).reshape(-1,1)
            xy_stars = np.squeeze(fits_data['XY_IMAGE']).reshape(-1,1)
            lambda1 = ((xx_stars+yy_stars)/2+np.sqrt(((xx_stars-yy_stars)/2)**2+xy_stars**2)).reshape(-1,1)
            lambda2 = ((xx_stars+yy_stars)/2-np.sqrt(((xx_stars-yy_stars)/2)**2+xy_stars**2)).reshape(-1,1)
            print(np.sum((lambda1/lambda2)<1.2)/len(lambda1),np.mean(lambda1/lambda2),np.median(lambda1/lambda2),np.min(lambda1/lambda2))

            dx = (x_stars_template.reshape(-1,1)-x_stars_this.reshape(1,-1)).reshape(-1,1)
            dy = (y_stars_template.reshape(-1,1)-y_stars_this.reshape(1,-1)).reshape(-1,1)

            xhist,xbins = np.histogram(dx,range=[-400,400],bins=801)
            yhist,ybins = np.histogram(dy,range=[-400,400],bins=801)

            idx = np.argmax(xhist)
            xshift = int((xbins[idx]+xbins[idx+1])/2.0)
            idx = np.argmax(yhist)
            yshift = int((ybins[idx]+ybins[idx+1])/2.0)
            res_list.append([res[2],xshift,yshift,np.sum((lambda1/lambda2)<good_star_threshold)/len(lambda1),len(lambda2)])
            print(xshift,yshift)
            arg = (xshift,yshift,res[2])
            mycursor = self.dl.cnx.cursor()
            sql = "UPDATE img SET img.x_to_template = %s, img.y_to_template = %s where img.image_id = %s;"
            mycursor.execute(sql,arg)
            self.dl.cnx.commit()
            

        return res_list

if __name__=="__main__":
    al = alignment()
    print(al.get_deviation(1305,3,1,"science_processed"))
