import os
from matplotlib import pyplot as plt
import numpy as np
from photutils.aperture import EllipticalAperture
from astropy import visualization as aviz
from astropy.nddata import block_reduce
from astropy.nddata.utils import Cutout2D
from astropy import wcs
from astropy.io import fits
import pandas as pd
from astropy.modeling.models import Moffat2D
from scipy.constants import h
import glob
import cv2
import batman



def show_image(image,
               percl=99, percu=None, is_mask=False,
               figsize=(10, 10),
               cmap='viridis', log=False, clip=True,
               show_colorbar=True, show_ticks=True,
               fig=None, ax=None, input_ratio=None):
    """
    Show an image in matplotlib with some basic astronomically-appropriat stretching.

    Parameters
    ----------
    image
        The image to show
    percl : number
        The percentile for the lower edge of the stretch (or both edges if ``percu`` is None)
    percu : number or None
        The percentile for the upper edge of the stretch (or None to use ``percl`` for both)
    figsize : 2-tuple
        The size of the matplotlib figure in inches
    """
    if percu is None:
        percu = percl
        percl = 100 - percl

    if (fig is None and ax is not None) or (fig is not None and ax is None):
        raise ValueError('Must provide both "fig" and "ax" '
                         'if you provide one of them')
    elif fig is None and ax is None:
        if figsize is not None:
            # Rescale the fig size to match the image dimensions, roughly
            image_aspect_ratio = image.shape[0] / image.shape[1]
            figsize = (max(figsize) * image_aspect_ratio, max(figsize))

        fig, ax = plt.subplots(1, 1, figsize=figsize)


    #算出fig图像提供的像素大小（宽，高）
    #.dpi 属性在 Matplotlib 中表示图形的每英寸点数（Dots Per Inch），它决定了图形的分辨率

    #DPI 值越高，图形的细节和清晰度就越高。dpi默认100
    #影响输出大小：当你保存图形时，DPI会影响输出文件的像素大小。计算方法是图形的尺寸（以英寸为单位）乘以DPI。例如，如果图形的大小为8 x 6 英寸，DPI 为 100，输出的像素大小将是：
    #宽度：8 * 100 = 800 像素
    #高度：6 * 100 = 600 像素
    fig_size_pix = fig.get_size_inches() * fig.dpi
    
    #输入的image大小与fig提供的像素比值取整，取某个max方向上的
    ratio = (image.shape // fig_size_pix).max()
    
    #保证不放大，但是可以bin
    if ratio < 1:
        ratio = 1

    ratio = input_ratio or ratio
    
    #according ratio to bin, usually func=np.mean
    reduced_data = block_reduce(image, ratio)

    if not is_mask:
        # Divide by the square of the ratio to keep the flux the same in thereduced image. We don’t want to do this for images which are masks, since their values should be zero or one.
        # 降采样是通过将多个相邻像素合并成一个像素来减少图像的分辨率。如果不调整合并后像素的值，图像的整体亮度可能会变化。具体来说，原始 4 个像素的光通量（亮度总和）与新合并的像素相比，可能会导致合并后的像素亮度偏低或偏高。除以比例的平方：保证真实度
         reduced_data = reduced_data / ratio**2

    # Of course, now that we have downsampled, the axis limits are changed to match the smaller image size. Setting the extent will do the trick tochange the axis display back to showing the actual extent of the image.
    extent = [0, image.shape[1], 0, image.shape[0]]

    #对数据log拉伸或者线性拉伸
    if log:
        stretch = aviz.LogStretch()
    else:
        stretch = aviz.LinearStretch()

    #使用 aviz.ImageNormalize 对 reduced_data 进行标准化，确保图像的亮度范围适当，使用 AsymmetricPercentileInterval 来设置亮度的上下限
    #clip:布尔值，指示是否在标准化过程中剪裁图像值，避免数据溢出。clip=True 会将超出指定区间的数据值裁剪到最大或最小值
    norm = aviz.ImageNormalize(reduced_data,
                               interval=aviz.AsymmetricPercentileInterval(percl, percu),
                               stretch=stretch, clip=clip)

    if is_mask:
        # The image is a mask in which pixels should be zero or one.block_reduce may have changed some of the values, so reset here.
        #if reduced_data >0,it's saved as True[1], if not,it's saved as False[0],so reduced_data return a布尔数组
        reduced_data = reduced_data > 0
        # Set the image scale limits appropriately.
        scale_args = dict(vmin=0, vmax=1)
    else:
        scale_args = dict(norm=norm)
        
    #origin原点，camp颜色映射，extent为位置坐标xmin，xmax，ymin，ymax，aspect：'equal'：确保每个像素在 x 和 y 方向上具有相同的大小（1:1比例）
    im = ax.imshow(reduced_data, origin='lower',
                   cmap=cmap, extent=extent, aspect='equal', **scale_args)

    if show_colorbar:
        # im关联的对象，ax关联的坐标轴，fraction宽度比，pad间隔
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    if not show_ticks:
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)


# Set up the random number generator, allowing a seed to be set from the environment
seed = os.getenv('GUIDE_RANDOM_SEED', None)

if seed is not None:
    seed = int(seed)
    
# This is the generator to use for any image component which changes in each image, e.g. read noise
# or Poisson error
noise_rng = np.random.default_rng(seed)

def read_noise(image, amount, gain=1.5):
    """
    Generate simulated read noise.
    
    Parameters
    ----------
    
    image: numpy array
        Image whose shape the noise array should match.
    amount : float
        Amount of read noise, in electrons.
    gain : float, optional
        Gain of the camera, in units of electrons/ADU.
    """
    shape = image.shape
    
    noise = noise_rng.normal(scale=amount/gain, size=shape)
    
    return noise

def bias(image, value, realistic=False):
    """
    Generate simulated bias image.
    
    Parameters
    ----------
    
    image: numpy array
        Image whose shape the bias array should match.
    value: float
        Bias level to add.
    realistic : bool, optional
        If ``True``, add some columns with somewhat higher bias value (a not uncommon thing)
    """
    # This is the whole thing: the bias is really suppose to be a constant offset!
#     prebias = np.random.RandomState(seed=40)
#     bias_im = prebias.randint(0, 10, size=(image.shape[0],image.shape[1])) + value
    bias_im = np.zeros_like(image) + value
    
    # If we want a more realistic bias we need to do a little more work. 
    if realistic:
        shape = image.shape
        number_of_colums = int(0.01 * shape[0])
        
        # We want a random-looking variation in the bias, but unlike the readnoise the bias should 
        # *not* change from image to image, so we make sure to always generate the same "random" numbers.
        rng = np.random.RandomState(seed=8392) 
        columns = rng.randint(1, shape[1], size=number_of_colums)
        # This adds a little random-looking noise into the data.
        col_pattern = rng.randint(0, int(0.1 * value), size=shape[0])
        
        # Make the chosen columns a little brighter than the rest...
        for c in columns:
            bias_im[:, c] = value + col_pattern
            
    return bias_im

def dark_current(image, current, exposure_time, gain=1.5, hot_pixels=False):
    """
    Simulate dark current in a CCD, optionally including hot pixels.
    
    Parameters
    ----------
    
    image : numpy array
        Image whose shape the cosmic array should match.
    current : float
        Dark current, in electrons/pixel/second, which is the way manufacturers typically 
        report it.0.05
    exposure_time : float
        Length of the simulated exposure, in seconds.
    gain : float, optional
        Gain of the camera, in units of electrons/ADU.
    strength : float, optional
        Pixel count in the cosmic rays.    
    """
    
    # dark current for every pixel; we'll modify the current for some pixels if 
    # the user wants hot pixels.
    base_current = current * exposure_time / gain
    
    # This random number generation should change on each call.
    dark_im = noise_rng.poisson(base_current, size=image.shape)
        
    if hot_pixels:
        # We'll set 0.01% of the pixels to be hot; that is probably too high but should 
        # ensure they are visible.
        y_max, x_max = dark_im.shape
        
        n_hot = int(0.0001 * x_max * y_max)
        
        # Like with the bias image, we want the hot pixels to always be in the same places
        # (at least for the same image size) but also want them to appear to be randomly
        # distributed. So we set a random number seed to ensure we always get the same thing.
        rng = np.random.RandomState(16201649)
        hot_x = rng.randint(1, x_max, size=n_hot)
        hot_y = rng.randint(1, y_max, size=n_hot)
        
        hot_current = 10000 * current
        
        dark_im[(hot_y, hot_x)] = hot_current * exposure_time / gain
    return dark_im

def sky_background(image, sky_counts, gain=1.5):
    """
    Generate sky background, optionally including a gradient across the image (because
    some times Moons happen).
    
    Parameters
    ----------
    
    image : numpy array
        Image whose shape the cosmic array should match.
    sky_counts : float
        The target value for the number of counts (as opposed to electrons or 
        photons) from the sky.
    gain : float, optional
        Gain of the camera, in units of electrons/ADU.
    """
    sky_im = noise_rng.poisson(sky_counts * gain, size=image.shape) / gain
    
    return sky_im

def complete_image(bias_level=1100, read=10.0, gain=1.5, dark=0.1, 
                   exposure=60, hot_pixels=False, sky_counts=200):
    synthetic_image = np.zeros([8120, 8120])
    background_image = synthetic_image + read_noise(synthetic_image, read) + bias(synthetic_image, bias_level, realistic=False) + dark_current(synthetic_image, dark, exposure, hot_pixels=hot_pixels) + sky_background(synthetic_image, sky_counts)

    return background_image


bias_save = fits.getdata('/Users/kexin_li/Documents/vs_py/fig/bias.fit')
bias_re = bias_save[500:6000, 1500:7000]
bias_co = cv2.resize(bias_re.astype(np.float32), (8120, 8120), interpolation=cv2.INTER_LINEAR) 

flat_save = fits.getdata('/Users/kexin_li/Documents/vs_py/fig/flat_co_bias.fit')
flat_re = flat_save[500:6000, 1500:7000]
flat_co = cv2.resize(flat_re.astype(np.float32), (8120, 8120), interpolation=cv2.INTER_LINEAR) 

background = complete_image(bias_level=1100, read=1.6, gain=1.5, dark=0.05, exposure=60, hot_pixels=False, sky_counts=100) 
background += bias_co + flat_co

FWHM = 1
alpha = 3
gamma = FWHM / (2 * np.sqrt(2**(1/alpha) - 1))



params = batman.TransitParams()
params.t0 = 0                      #time of inferior conjunction
params.per = 3.72                      #orbital period
params.rp = 0.105                     #planet radius (in units of stellar radii)
params.a = 11.4                       #semi-major axis (in units of stellar radii)
params.inc = 88.5                     #orbital inclination (in degrees)
params.ecc = 0.                      #eccentricity
params.w = 0                       #longitude of periastron (in degrees)
params.u = [0.1, 0.3]                #limb darkening coefficients [u1, u2]
params.limb_dark = "quadratic"       #limb darkening model
time =np.linspace(3.62, 3.8, 259)
m = batman.TransitModel(params, time)    #initializes model
flux_var = m.light_curve(params)          #calculates light curve

data = pd.read_csv('~/Documents/vs_py/wasp11b.csv')
ra = data['ra'].to_list()
dec = data['dec'].to_list()
# flux_jy =data['phot_g_mean_flux'].to_list()
# flux = [10**(-26) * f_jy for f_jy in flux_jy]

mag =data['phot_g_mean_mag'].to_list()
flux = [870 * 10**(-0.4*(m+29)) for m in mag]

synthetic_image = np.zeros([8120, 8120])
t_ex = 60
gain = 1.5
A = [F * (0.5)**2 * t_ex * 0.9 * (alpha - 1)/ (gain * np.pi * gamma**2 * h * (4.82e14)) for F in flux] 
A_var = A[0] * flux_var
# def count_above_threshold(lst, threshold):
#     return sum(1 for x in lst if x > threshold)
# threshold_value = 2**18 -1
# count = count_above_threshold(A, threshold_value)
del A[0]
mapping = dict(zip(time, A_var))

# print(flux)
# print(A) 
# print(A_var)
# print(count)

# ra = np.array(ra)
# dec = np.array(dec)
# distances = np.sqrt((ra - 47.3689)**2 + (dec - 30.6734)**2)
# target_index = np.argmin(distances)
# print(target_index)

def stars(image, pixcrd, A, gamma, alpha, t, x_motion, y_motion, t_c, v_x, v_y, pix_var, A_var, map, 
        air_craft=False, motion=False, variation=False, gain=1.5):

    image_star = np.zeros(image.shape)

    x0 = [item[0] for item in pixcrd]
    y0 = [item[1] for item in pixcrd]
    
    for x0, y0, A in zip(x0, y0, A):
        x_min = max(int(x0 - 150), 0)
        x_max = min(int(x0 + 150), image.shape[0])
        y_min = max(int(y0 - 150), 0)
        y_max = min(int(y0 + 150), image.shape[1])

        if x_min >= x_max or y_min >= y_max:
            continue  

        X_sub, Y_sub = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        
        g = Moffat2D(amplitude=A, x_0=x0, y_0=y0, gamma=gamma, alpha=alpha)
        
        image_star[y_min:y_max, x_min:x_max] += g(X_sub, Y_sub)

    if air_craft:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if abs(j - (-i +4000)) < 1:
                    x_min = max(int(i - 150), 0)
                    x_max = min(int(i + 150), image.shape[0])
                    y_min = max(int(j - 150), 0)
                    y_max = min(int(j + 150), image.shape[1])

                    if x_min >= x_max or y_min >= y_max:
                        continue  
                    
                    X_sub, Y_sub = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
                    g = Moffat2D(amplitude=np.max(A), x_0=i, y_0=j, gamma=gamma, alpha=alpha)
                    image_star[y_min:y_max, x_min:x_max] += g(X_sub, Y_sub)

    if motion:
        for i in range(len(x_motion)):
            xt = x_motion[i] + v_x[i] * (t-t_c)
            yt = y_motion[i] + v_y[i] * (t-t_c)
        
            x_min = max(int(xt - 150), 0)
            x_max = min(int(xt + 150), image.shape[0])
            y_min = max(int(yt - 150), 0)
            y_max = min(int(yt + 150), image.shape[1])

            if x_min >= x_max or y_min >= y_max:
                continue  
                    
            X_sub, Y_sub = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
            g = Moffat2D(amplitude=np.median(A), x_0=xt, y_0=yt, gamma=gamma, alpha=alpha)
            image_star[y_min:y_max, x_min:x_max] += g(X_sub, Y_sub)

    if variation:
        x_var = pix_var[0]
        y_var = pix_var[1]
        for i in range(1):
            x = x_var
            y = y_var
        
            x_min = max(int(x - 150), 0)
            x_max = min(int(x + 150), image.shape[0])
            y_min = max(int(y - 150), 0)
            y_max = min(int(y + 150), image.shape[1])

            if x_min >= x_max or y_min >= y_max:
                continue  
                    
            X_sub, Y_sub = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
            g = Moffat2D(amplitude=mapping.get(t), x_0=x, y_0=y, gamma=gamma, alpha=alpha)
            image_star[y_min:y_max, x_min:x_max] += g(X_sub, Y_sub)

    normal_noise = np.random.normal(0, 1, size=image.shape)
    image_star += normal_noise * np.sqrt(np.clip(image_star, 0, None))

    # 创建一个布尔数组，判断哪些像素值过曝
    full_well = 2**18 - 1
    overexposed_mask = image_star > full_well
    # 将过曝的像素值设置为满阱值
    image_star[overexposed_mask] = full_well

    return image_star

w = wcs.WCS(naxis=2)
w.wcs.crpix = [4060, 4060]
w.wcs.cdelt = np.array([3.6e-4, 3.6e-4])
w.wcs.crval = [47.3689, 30.6734]
w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
cord = [[ra[i], dec[i]] for i in range(len(ra))]
world = np.array(cord, np.float_)
pixcrd = w.wcs_world2pix(world, 1)
pix_var = pixcrd[0]
pix = pixcrd[1:]

x_motion = [880]
y_motion = [660]
t_c = 3.62
v_x = [4.1e5]
v_y = [5.0e5]  #v:pixe/d, 1.3"/pixel

for i in range(1):
    t = time[i]
    stars_with_background = background + stars(synthetic_image, pix, A, gamma, alpha, t, x_motion, y_motion, t_c, v_x, v_y, pix_var, A_var, mapping,
        air_craft=True, motion=True, variation=True, gain=1)
    
    header = w.to_header()
    header['EXPTIME'] = (60, '/SEC')
    header['TIME'] = (t, '/JD')
    hdu = fits.PrimaryHDU(stars_with_background, header=header)
    hdu.writeto(f'/Users/kexin_li/Documents/vs_py/fig/image_moffat{i}.fits', overwrite=True)

