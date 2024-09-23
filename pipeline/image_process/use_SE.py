import os
import time
from astropy.io import fits
import numpy as np
class SE:
    def __init__(self):
        self.conf ="""
CATALOG_TYPE     FITS_LDAC     # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,
                                # ASCII_VOTABLE, FITS_1.0 or FITS_LDAC
PARAMETERS_NAME  /home/yichengrui/workspace/TianYu/pipeline/image_process/pipelines/loop/default.param  # name of the file containing catalog contents
 
#------------------------------- Extraction ----------------------------------
 
DETECT_TYPE      CCD            # CCD (linear) or PHOTO (with gamma correction)
DETECT_MINAREA   8              # min. # of pixels above threshold

DETECT_THRESH    3            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2
ANALYSIS_THRESH  3            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2
 
FILTER           N              # apply filter for detection (Y or N)?
FILTER_NAME      default.conv   # name of the file containing the filter
 
DEBLEND_NTHRESH  32             # Number of deblending sub-thresholds
DEBLEND_MINCONT  0.005          # Minimum contrast parameter for deblending
 
CLEAN            Y              # Clean spurious detections? (Y or N)?
CLEAN_PARAM      1.0            # Cleaning efficiency
 
#-------------------------------- WEIGHTing ----------------------------------

WEIGHT_TYPE      NONE           # type of WEIGHTing: NONE, BACKGROUND,
                                # MAP_RMS, MAP_VAR or MAP_WEIGHT
WEIGHT_IMAGE     weight.fits    # weight-map filename

#-------------------------------- FLAGging -----------------------------------

FLAG_IMAGE       flag.fits      # filename for an input FLAG-image
FLAG_TYPE        OR             # flag pixel combination: OR, AND, MIN, MAX
                                # or MOST

#------------------------------ Photometry -----------------------------------
 
PHOT_APERTURES   15              # MAG_APER aperture diameter(s) in pixels
PHOT_AUTOPARAMS  2.5, 3.5       # MAG_AUTO parameters: <Kron_fact>,<min_radius>
PHOT_PETROPARAMS 2.0, 3.5       # MAG_PETRO parameters: <Petrosian_fact>,
                                # <min_radius>
PHOT_AUTOAPERS   0.0,0.0        # <estimation>,<measurement> minimum apertures
                                # for MAG_AUTO and MAG_PETRO

SATUR_LEVEL      60000.0        # level (in ADUs) at which arises saturation
SATUR_KEY        SATURATE       # keyword for saturation level (in ADUs)
 
MAG_ZEROPOINT    0.0            # magnitude zero-point
MAG_GAMMA        4.0            # gamma of emulsion (for photographic scans)
GAIN             1.0            # detector gain in e-/ADU
GAIN_KEY         GAIN           # keyword for detector gain in e-/ADU
PIXEL_SCALE      0.906            # size of pixel in arcsec (0=use FITS WCS info)
 
#------------------------- Star/Galaxy Separation ----------------------------
 
SEEING_FWHM      3          # stellar FWHM in arcsec
STARNNW_NAME     /home/yichengrui/workspace/TianYu/pipeline/image_process/pipelines/loop/default.nnw    # Neural-Network_Weight table filename
 
#------------------------------ Background -----------------------------------
 
BACK_TYPE        AUTO           # AUTO or MANUAL
BACK_VALUE       0.0            # Default background value in MANUAL mode
BACK_SIZE        64             # Background mesh: <size> or <width>,<height>
BACK_FILTERSIZE  3              # Background filter: <size> or <width>,<height>
 
#------------------------------ Check Image ----------------------------------
 
CHECKIMAGE_TYPE  NONE           # can be NONE, BACKGROUND, BACKGROUND_RMS,
                                # MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND,
                                # FILTERED, OBJECTS, -OBJECTS, SEGMENTATION,
                                # or APERTURES
CHECKIMAGE_NAME  check.fits     # Filename for the check-image
 
#--------------------- Memory (change with caution!) -------------------------
 
MEMORY_OBJSTACK  3000           # number of objects in stack
MEMORY_PIXSTACK  300000         # number of pixels in stack
MEMORY_BUFSIZE   1024           # number of lines in buffer
 
#------------------------------- ASSOCiation ---------------------------------

ASSOC_NAME       sky.list       # name of the ASCII file to ASSOCiate
ASSOC_DATA       2,3,4          # columns of the data to replicate (0=all)
ASSOC_PARAMS     2,3,4          # columns of xpos,ypos[,mag]
ASSOC_RADIUS     2.0            # cross-matching radius (pixels)
ASSOC_TYPE       NEAREST        # ASSOCiation method: FIRST, NEAREST, MEAN,
                                # MAG_MEAN, SUM, MAG_SUM, MIN or MAX
ASSOCSELEC_TYPE  MATCHED        # ASSOC selection type: ALL, MATCHED or -MATCHED

#----------------------------- Miscellaneous ---------------------------------
 
VERBOSE_TYPE     NORMAL         # can be QUIET, NORMAL or FULL
HEADER_SUFFIX    .head          # Filename extension for additional headers
WRITE_XML        N              # Write XML file (Y/N)?
XML_NAME         sex.xml        # Filename for XML output
XSL_URL          file:///usr/local/share/sextractor/sextractor.xsl
                                # Filename for XSL style-sheet
"""

    def use(self,input_file_path,output_file_path = 'temp.fit',keep_out = True,ymax = 2200,use_sep = False):
        if not use_sep:
            if not keep_out:
                output_file_path = "temp_"+str(hash(time.time()))+".fit"
            new_conf = "CATALOG_NAME     "+output_file_path+'\n'+self.conf
            temp_file_name = "temp_conf_"+str(hash(time.time()))+".txt"
            with open(temp_file_name,'w') as f:
                f.write(new_conf)
            os.system('sex '+input_file_path+' -c '+temp_file_name)
            os.system('rm '+temp_file_name)
            if not keep_out:
                res = fits.open(output_file_path)
                ret_res = res[2].data
                os.system('rm '+output_file_path)
                Y_max = np.squeeze(ret_res['YMAX_IMAGE'])
                ret_res = ret_res[Y_max<ymax]
                return ret_res
        else:
            import sep
            
            data = fits.getdata(input_file_path)
            data = data.byteswap().newbyteorder()
            bkg = sep.Background(data)
            data_sub = data - bkg
            objects = sep.extract(data_sub,2, err=bkg.globalrms)
            return objects
        return -1

if __name__=="__main__":
    se = SE()
    se.use("/home/share/muguang/image/frame/2024-03-02/M81-0400_corrected_at_533232904610986570.fits","/home/yichengrui/workspace/TianYu/pipeline/image_process/try.fit")
