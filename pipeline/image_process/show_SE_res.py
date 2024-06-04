from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
img = fits.getdata("/home/share/muguang/image/frame/2024-03-02/M81-0400_corrected_at_533232904610986570.fits")
fits_res = fits.open('/home/yichengrui/workspace/TianYu/pipeline/image_process/try.fit')
# print(fits_res[2].header)
print(fits_res[2].data['X_IMAGE'])
x_stars = np.squeeze(fits_res[2].data['X_IMAGE'])
y_stars = np.squeeze(fits_res[2].data['Y_IMAGE'])
xx_stars = np.squeeze(fits_res[2].data['X2_IMAGE']).reshape(-1,1)
yy_stars = np.squeeze(fits_res[2].data['Y2_IMAGE']).reshape(-1,1)
xy_stars = np.squeeze(fits_res[2].data['XY_IMAGE']).reshape(-1,1)
flag_star = np.squeeze(fits_res[2].data['CLASS_STAR'])
Y_max = np.squeeze(fits_res[2].data['YMAX_IMAGE'])
BG = np.squeeze(fits_res[2].data['BACKGROUND'])

lambda1 = ((xx_stars+yy_stars)/2+np.sqrt(((xx_stars-yy_stars)/2)**2+xy_stars**2)).reshape(-1,1)
lambda2 = ((xx_stars+yy_stars)/2-np.sqrt(((xx_stars-yy_stars)/2)**2+xy_stars**2)).reshape(-1,1)
print(np.sum((lambda1/lambda2)<1.2))
print(np.mean(lambda1/lambda2))
print(np.median(lambda1/lambda2))
print(np.min(lambda1/lambda2))
theta = np.arctan2(lambda1-xx_stars,xy_stars)
extent = (0,img.shape[1],0,img.shape[0])
#print(BG)
plt.figure(figsize = (40,20))
plt.imshow(img,vmin = 4000, vmax = 5000,extent=extent, origin='lower')

t = np.linspace(0,2*np.pi,100).reshape(1,-1)
xt = np.sqrt(lambda1)*np.cos(theta)*np.cos(t)-np.sqrt(lambda2)*np.sin(theta)*np.sin(t)
yt = np.sqrt(lambda1)*np.sin(theta)*np.cos(t)+np.sqrt(lambda2)*np.cos(theta)*np.sin(t)
ct = 0
for i in range(len(xt)):
    if Y_max[i]<2100:
        ct +=1
        if flag_star[i]<0.5:
            plt.plot(x_stars[i]+xt[i]*2.447,y_stars[i]+yt[i]*2.447,'k',alpha = 0.8)
        else:
            plt.plot(x_stars[i]+xt[i]*2.447,y_stars[i]+yt[i]*2.447,'r',alpha = 0.8)



print(ct)
plt.savefig('res.pdf')
fits_res.close()