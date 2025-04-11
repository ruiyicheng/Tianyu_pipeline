import sys
sys.path.append("D:\\code\\control\\qhyccd-python\\")
import qhyccd
import time
import numpy as np
import cv2
from time import strftime,localtime
from astropy.time import Time
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import subprocess
from astropy.table import Table
#import sep
cv2.namedWindow('image')
def nothing(x):
    pass
# create trackbars for color change
nbit = 16
cv2.createTrackbar('lower','image',120,2**nbit-1,nothing)
cv2.createTrackbar('upper','image',230,2**nbit-1,nothing)


qc = qhyccd.qhyccd()

def live_stack(file_name, ra = -999, dec = -999, save_dir = 'D:\\data_to_transfer\\ucac4522-043383\\',JD_start = Time.now().jd,duration_s = 30,exposure_s = 5,resolve = False, sub_targets = [],NAXIS1 = 9576, NAXIS2 =6388, pixel_scale = 0.30259,temp_fits_path = "D:\\Tianyu_data\\supplimentary_resource\\scamp_input\\",temp_fits_name = 'template.fits' ,cut_size = (10,10)):
    print(Time.now().jd,JD_start)
    if len(sub_targets) != 0:
        print('have sub targets, must resolve first')
    while(Time.now().jd< JD_start):
        print(f'waiting for start time {JD_start}')
    #qc.SetROI(2500, 2500, 2800, 2800)
    qc.SetGain(0)
    #qc.SetIgnoreOverscan(False)
    qc.SetBias(15)
    qc.SetExposure(1000*exposure_s)

    qc.BeginLive()
    header_main = fits.Header()
    

    last_frame_ct = 0 
    last_time = time.time()
    img_all = 0
    number_frame = 0
    total_raw_list = []
    t_start_list = []
    while(Time.now().jd - JD_start < duration_s/86400):
        #time.sleep(0.001)
        img_cap = qc.GetLiveFrame()[:NAXIS2,-NAXIS1:]
        si = int(np.sum(img_cap))
        #print(si)
        #print(si)
        if si != last_frame_ct:
            t_start_exp = Time.now().jd-exposure_s/86400
            t_start_list.append(t_start_exp)
            if number_frame == 0:
                header_main['JD'] = t_start_exp
            number_frame += 1
            total_raw_list.append(img_cap.copy())
            img_deal = img_cap.copy().astype(np.float32)
            img_all += img_deal

            lower = cv2.getTrackbarPos('lower','image')
            upper = cv2.getTrackbarPos('upper','image')
            #print(np.max(img))
            #print(np.min(img))
            last_frame_ct = si
            #print(si)
            #print(si/img.shape[0]/img.shape[1])
            seperation = time.time() - last_time
            print(f'seperation = {seperation}')
            last_time = time.time()
            #print(lower, upper)
            
            img_s = np.maximum(0,np.minimum(1,(img_all[::10,::10]/number_frame-lower)/(upper-lower)))
            cv2.imshow('image',img_s)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
               break
        else:
            continue
        #print(np.sum(img))
        #server.send_img( img )
        #print( time.monotonic() )
    qc.StopLive()

    img_all /= number_frame
    m = np.mean(img_all)
    s = np.std(img_all)
    header_main['EXPOSURE'] = number_frame*exposure_s
    # header_main['NAXIS1'] = NAXIS1
    # header_main['NAXIS2'] = NAXIS2
    if ra!=-999 and dec!=-999:
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wcs.wcs.crval = [ra, dec]
        wcs.wcs.crpix = [NAXIS1/2, NAXIS2/2]
        wcs.wcs.cd = [[-pixel_scale/3600, 0],[0, pixel_scale/3600]]
        # wcs.wcs.cdelt = [-pixel_scale/3600, pixel_scale/3600]
        print(wcs.to_header())
        header_main.update(wcs.to_header())
    else:
        print('no initial ra dec provided')
    #bkg = sep.Background(img_all)
    fits.writeto(temp_fits_path+temp_fits_name,img_all,header_main,overwrite=True)
    # cmd = f"wsl cd /mnt/d/Tianyu_data/supplimentary_resource/scamp_input/; solve-field --overwrite --use-wcs {temp_fits_name}"
    # subprocess.run(cmd,shell = True)

    cmd = f"wsl cd /mnt/d/Tianyu_data/supplimentary_resource/scamp_input/; source-extractor {temp_fits_name}"
    subprocess.run(cmd,shell = True)
    cmd = f"wsl cd /mnt/d/Tianyu_data/supplimentary_resource/scamp_input/;scamp test.cat -c default.scamp"
    subprocess.run(cmd,shell = True)

    with open(temp_fits_path+"test.head", "r") as f:
        head_data = f.read()

    # Convert the .head data into a FITS header object
    head_split = head_data.split('\n')
    s = ''
    for hs in head_split[3:]:
        s += hs+'\n'
    header_scamp = fits.Header.fromstring(s, sep='\n')
    header_main.update(header_scamp)
    wcs_scamp = WCS(header_scamp)
    pixels_scamp = wcs_scamp.all_world2pix(
        np.array(sub_targets),
        0,
    )
    total_raw_list = np.array(total_raw_list,dtype = np.float32)
    hdut = Table(np.array(t_start_list).reshape(-1,1), names=('JD',))
    for i in range(len(sub_targets)):
        print(f'sub_target {i} at {pixels_scamp[i]}')
        x,y = pixels_scamp[i]
        x = int(x)
        y = int(y)
        cut_this = total_raw_list[:,y-cut_size[1]:y+cut_size[1],x-cut_size[0]:x+cut_size[0]]
        print(cut_this.shape)
        print(total_raw_list.shape)
        header_this = fits.Header()
        header_this['RA'] = sub_targets[i][0]
        header_this['DEC'] = sub_targets[i][1]
        header_this['X'] = x
        header_this['Y'] = y
        # hdu0 = fits.PrimaryHDU(header = header_this)
        # hdu1 = fits.ImageHDU(data = cut_this)
        # hdu = fits.HDUList([hdu0,hdu1])
        np.save(save_dir+file_name.split('.')[-2]+f'_{x}_{y}.npy',cut_this)
        fits.writeto(save_dir+file_name.split('.')[-2]+f'_{x}_{y}.fits',cut_this,header_this,overwrite=True)
    fits.writeto(save_dir+file_name,img_all,header_main,overwrite=True)
    
    hdut.write(save_dir+file_name.split('.')[-2]+'_time.fits',format='fits',overwrite=True)
    # plt.imshow(img_all,cmap='gray',vmin=m-1*s,vmax=m+1*s)
    # plt.show() 
if __name__ == "__main__":
    names = [f"ucac4522-043383_{i}.fits" for i in range(2)]
    ra = 118.26169634777
    dec = 14.226946985
    sub_targets = np.array([[118.26169634777,14.226946985]])
    for name in names:
        live_stack(name,JD_start = Time.now().jd,ra = ra, dec = dec,sub_targets = sub_targets)