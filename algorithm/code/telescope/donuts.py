import numpy as np
import astropy.io.fits as fits
import sep
import glob
import time
import matplotlib.pyplot as plt
class donuts:
    def __init__(self, image):
        self.image = image.copy()
        sum_image = np.sum(self.image)
        self.X_temp = np.sum(self.image, axis=0)/sum_image
        self.Y_temp = np.sum(self.image, axis=1)/sum_image
        self.X_fft = np.fft.fft(self.X_temp)
        self.Y_fft = np.fft.fft(self.Y_temp)
        

    def deviation(self,new_image):
        def linear_regression(x,y):
            model = np.polyfit(x, y, 2)
            a,b,c = model
            mid = -b/(2*a)
            return mid
            
        
        new_image = np.nan_to_num(new_image, nan=0.0)
        sum_image = np.sum(new_image)
        new_x = np.sum(new_image, axis=0)/sum_image
        new_y = np.sum(new_image, axis=1)/sum_image
        #print(new_x)
        new_x_fft = np.fft.fft(new_x)
        new_y_fft = np.fft.fft(new_y)
        ccf_x = np.fft.ifft(self.X_fft * np.conj(new_x_fft))
        ccf_y = np.fft.ifft(self.Y_fft * np.conj(new_y_fft))
        ccf_x = np.abs(ccf_x)
        ccf_y = np.abs(ccf_y)
        ccf_y[500:-500] = 0
        ccf_x[500:-500] = 0
        #print(f"CCF X: {ccf_x}, CCF Y: {ccf_y}")
        plt.figure()
        plt.plot(list(range(len(ccf_x))),ccf_x, label='CCF X')
        plt.yscale('log')
        #plt.savefig('/home/test/workspace/Tianyu_pipeline/algorithm/data/testoutput/plot/ccf_x.png')
        plt.plot(list(range(len(ccf_y))),ccf_y, label='CCF Y')
        plt.yscale('log')
        plt.savefig('/home/test/workspace/Tianyu_pipeline/algorithm/data/testoutput/plot/ccf.png')
        delta_x = np.argmax(ccf_x)
        delta_y = np.argmax(ccf_y)
        range_x = np.arange(len(ccf_x))
        #make range_x>1/2 len(ccf_x) to be negative
        range_x[range_x > len(ccf_x)/2] -= len(ccf_x)
        range_y = np.arange(len(ccf_y))
        range_y[range_y > len(ccf_y)/2] -= len(ccf_y)
        if delta_x == 0:
            x = np.array([-1,0,1])
            y = np.array([ccf_x[-1], ccf_x[0], ccf_x[1]])
        else:
            x = range_x[delta_x-1:delta_x+2]
            y = ccf_x[delta_x-1:delta_x+2]
        delta_x_fraction = linear_regression(x, y)
        plt.figure()
        plt.plot(x, y, 'ro')
        plt.axvline(x=delta_x_fraction, color='r', linestyle='--', label='Delta X Fraction')
        plt.xlabel('Shift (pixels)')
        plt.ylabel('Cross-correlation')
        plt.title('Cross-correlation peak')
        plt.savefig('/home/test/workspace/Tianyu_pipeline/algorithm/data/testoutput/plot/ccf_peak_x.png')
        
        if delta_y == 0:
            x = np.array([-1,0,1])
            y = np.array([ccf_y[-1], ccf_y[0], ccf_y[1]])
        else:
            x = range_y[delta_y-1:delta_y+2]
            y = ccf_y[delta_y-1:delta_y+2]
        delta_y_fraction = linear_regression(x, y)
        plt.figure()
        plt.plot(x, y, 'ro')
        plt.axvline(x=delta_y_fraction, color='r', linestyle='--', label='Delta Y Fraction')
        plt.xlabel('Shift (pixels)')
        plt.ylabel('Cross-correlation')
        plt.title('Cross-correlation peak')
        plt.savefig('/home/test/workspace/Tianyu_pipeline/algorithm/data/testoutput/plot/ccf_peak_y.png')
        return delta_x_fraction, delta_y_fraction
    
if __name__ == "__main__":
    template_image_path = '/home/test/workspace/Tianyu_pipeline/algorithm/data/testoutput/image/stacked_science/stacked_KOI68.fit'
    template_image = fits.getdata(template_image_path).byteswap().newbyteorder()
    template_image_no_bkg = template_image-sep.Background(template_image).back()
    donuts_instance = donuts(template_image_no_bkg)

    test_image_path = glob.glob('/home/test/workspace/Tianyu_pipeline/algorithm/data/testoutput/image/calibrated_science/*calibrated.fit')
    test_image_path.sort()
    for image_path in test_image_path:
        test_image = fits.getdata(image_path).byteswap().newbyteorder().astype('float32')
        t0 = time.time()
        bkg = sep.Background(test_image)
        test_image_no_bkg = test_image-bkg.back()
        delta_x, delta_y = donuts_instance.deviation(test_image_no_bkg)
        t1 = time.time()
        print(f"Image: {image_path}, Delta X: {delta_x}, Delta Y: {delta_y}, Time taken: {t1-t0:.2f} seconds")
    