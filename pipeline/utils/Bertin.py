# Use Bertin's software in a Pythonic way
# Called by other components by:
# res = self.consumer.Bertin.method()
import os
import subprocess
import pandas as pd
import numpy as np
import astrometry
import astropy
from astropy.wcs import WCS
from astropy.io import fits
from Tianyu_pipeline.pipeline.middleware.consumer_component import consumer_component
from astroquery.gaia import Gaia
from astropy.table import Table
import sep

class Bertin_tools(consumer_component):
    def __init__(self,mode = "consumer_component",cache_dir = None):
        '''
        if used indepentently, set mode = "independent" and cache_dir = path to the cache directory
        '''
        super().__init__()
        self.mode = mode
        self.cache_dir = cache_dir
    def search_GDR3_by_circle(self,ra=180,dec=0,fov=1,Gmag_limit = 17,cache = True): # stateless version of the dl function. implement here to enable independent usage
        file_name = f"{self.cache_dir}/{ra}_{dec}_{fov}_{Gmag_limit}.csv"
        # if exists read from file
        print('Searching for',file_name)
        if os.path.exists(file_name):
            print('Found',file_name)
            return Table.from_pandas(pd.read_csv(file_name))
        
        sql = f'''
        SELECT g3.source_id,g3.ra,g3.dec,g3.ra_error,g3.dec_error,g3.phot_g_mean_mag,g3.phot_g_mean_flux_over_error FROM gaiadr3.gaia_source AS g3
WHERE g3.phot_g_mean_mag<{Gmag_limit} AND
CONTAINS(
POINT('ICRS',g3.ra,g3.dec),
CIRCLE('ICRS',{ra},{dec},{fov})
)=1 ORDER BY g3.phot_g_mean_mag'''      
        job = Gaia.launch_job_async(sql)
        r = job.get_results()
        if cache:
            r.to_pandas().to_csv(file_name,index = False)
        return r
    def connect(self,consumer):
        """
        Overload to enable dependent utilization
        """
        self.connected_to_consumer = True
        self.consumer = consumer
        self.consumer_id = id(consumer)
        self.cache_dir = self.consumer.cache.cache_dir



    def generate_parameter_pipe(self,parameter_dict,exportfilename):
        cmd = ""
        if len(parameter_dict)>0:
            cmd = " | sed"
            for k in parameter_dict:
                re_remove = f'^{k}  *[:/_.0-9a-zA-Z]*  *'
                cmd += f" -e 's|{re_remove}|{k} {parameter_dict[k]} |g' "
        cmd += f" > {exportfilename}"
        return cmd
    def generate_output_sextractor(self,parameter_list,exportfilename):
        cmd = ""
        if len(parameter_list)>0:
            cmd = " | sed"
            for item in parameter_list:
                cmd += f" -e 's/^#{item}/{item} #/g' "
        cmd += f" > {exportfilename}"
        return cmd
    def SExtractor(self,img_path,param_dict,output_list,out_file_name = -1):
        """
        Input: path to the image, path to the catalog
        Use sextractor
        Output: path to the catalog
        """
        if not "FILTER" in param_dict:
            param_dict["FILTER"] = "N" # No filter by default
        if not "CATALOG_TYPE" in param_dict:
            param_dict["CATALOG_TYPE"] = "FITS_LDAC"
        # Generate the parameter and config file names
        param_file_name = f"sextractor_{os.getpid()}.param"
        config_file_name = f"sextractor_{os.getpid()}.sex"
        if out_file_name == -1:
            out_file_name = f"sextractor_{os.getpid()}.cat"
        param_file_path = os.path.join(self.cache_dir,param_file_name)
        config_file_path = os.path.join(self.cache_dir,config_file_name)
        out_file_path = os.path.join(self.cache_dir,out_file_name)

        # Generate the parameter file
        command = "sex -dp"+self.generate_output_sextractor(output_list,param_file_path)
        subprocess.run(command, shell=True)

        # Generate the config file
        param_dict["CATALOG_NAME"] = out_file_path
        param_dict["PARAMETERS_NAME"] = param_file_path
        command = "sex -dd"+self.generate_parameter_pipe(param_dict,config_file_path)
        subprocess.run(command, shell=True)

        # Run sextractor to obtain the result
        command = f"sex {img_path} -c {config_file_path}"
        subprocess.run(command, shell=True)

        # Delete the parameter and config files
        subprocess.run(f"rm {param_file_path}", shell=True)
        subprocess.run(f"rm {config_file_path}", shell=True)
        return out_file_path


    def SCAMP(self,sextractor_catalog_path,param_dict,reference_catalog_name,out_file_name = -1):
        if not "WRITE_XML" in param_dict:
            param_dict["WRITE_XML"] = "N" # Not write XML
        if not "CHECKPLOT_ANTIALIAS" in param_dict:
            param_dict["CHECKPLOT_ANTIALIAS"] = "N" # Not write XML
        if not "SOLVE_PHOTOM" in param_dict:
            param_dict["SOLVE_PHOTOM"] = "Y" # Not write do photometric calibration
        config_file_name = f"scamp_{os.getpid()}.scamp"
        
        if out_file_name == -1:
            out_file_name = f"scamp_{os.getpid()}.head"
        out_file_path = os.path.join(self.cache_dir,out_file_name)
        config_file_path = os.path.join(self.cache_dir,config_file_name)
        reference_catalog_path = os.path.join(self.cache_dir,reference_catalog_name)
        # Generate the config file



        param_dict["ASTREFCAT_NAME"] = reference_catalog_path
        param_dict["HEADER_NAME"] = out_file_path

        command = "scamp -dd"+self.generate_parameter_pipe(param_dict,config_file_path)
        subprocess.run(command, shell=True)

        command = f"scamp {sextractor_catalog_path} -c {config_file_path}"
        subprocess.run(command, shell=True)

        # Delete the parameter and config files
       
        subprocess.run(f"rm {config_file_path}", shell=True)
        return out_file_path

    def SCAMP_image(self,img_path,sextractor_dict_param,ra_deg = None,dec_deg = None, arcsec_per_pixel = None,WCS_header = None):

            # A one-shot function to obtain the WCS of the image.
            # hint using ra/dec or a header with WCS
            # If ra/dec is not None, use it to generate the initialal WCS
            # If WCS_header is not None, copy it to image header directly
        def Table_to_catalog(table):
                # Convert the table to a catalog
            header_str = "\n".join([
                "SIMPLE  =                    T /",
                "BITPIX  =                   16 /",
                "NAXIS   =                    2 /",
                "NAXIS1  =                 2048 /",
                "NAXIS2  =                 2048 /",
                "END"
            ])
            # Pad each line to exactly 80 characters
            header_lines_padded = [line.ljust(80) for line in header_str]
            header_array = np.array(header_lines_padded, dtype='S80')

            col1 = fits.Column(name='Field Header Card', format='80A', array=header_array)
            hdu1 = fits.BinTableHDU.from_columns([col1], name='LDAC_IMHEAD')

            # ================================
            # 2. Create LDAC_OBJECTS extension
            # ================================
            # This is the actual reference catalog content

            # Sample data
            X_WORLD = table['ra']  # degrees
            Y_WORLD = table['dec']  # degrees
            ERRA_WORLD  = table['ra_error']/3600000  # mas->degrees
            ERRB_WORLD  = table['dec_error']/3600000
            MAG = table['phot_g_mean_mag']
            MAG_ERR = (2.5/np.log(10))/table['phot_g_mean_flux_over_error']


            # Define columns (must match SCAMP expectations!)
            cols = [
                fits.Column(name='X_WORLD', format='D', array=X_WORLD),
                fits.Column(name='Y_WORLD', format='D', array=Y_WORLD),
                fits.Column(name='ERRA_WORLD', format='D', array=ERRA_WORLD),
                fits.Column(name='ERRB_WORLD', format='D', array=ERRB_WORLD),
                fits.Column(name='MAG', format='D', array=MAG),
                fits.Column(name='MAGERR', format='D', array=MAG_ERR)
            ]
            hdu2 = fits.BinTableHDU.from_columns(cols, name='LDAC_OBJECTS')

            # ================================
            # 3. Save to FITS file
            # ================================
            hdul = fits.HDUList([fits.PrimaryHDU(), hdu1, hdu2])
            ref_catalog_name = f"ref_catalog_{os.getpid()}.fits"
            ref_catalog_path = os.path.join(self.cache_dir, ref_catalog_name)
            hdul.writeto(ref_catalog_path, overwrite=True)
            return ref_catalog_name
        def get_header_from_scamp_output(scamp_output_path):
            with open(scamp_output_path, "r") as f:
                head_data = f.read()

            # Convert the .head data into a FITS header object
            head_split = head_data.split('\n')
            s = ''
            for hs in head_split[3:]:
                s += hs+'\n'
            header_scamp = fits.Header.fromstring(s, sep='\n')
            return header_scamp
        if not "DETECT_MINAREA" in sextractor_dict_param:
            sextractor_dict_param["DETECT_MINAREA"] = 5
        if not "DETECT_THRESH" in sextractor_dict_param:
            sextractor_dict_param["DETECT_THRESH"] = 3
        WCS_KEYWORDS = [f"CD1_{i}" for i in range(1,11)]+[f"CD2_{i}" for i in range(1,11)]+[f"PC1_{i}" for i in range(1,11)]+[f"PC2_{i}" for i in range(1,11)]+[f"PV1_{i}" for i in range(0,11)]+[f"PV2_{i}" for i in range(0,11)]+['WCSAXES', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CTYPE1', 'CTYPE2','CUNIT1', 'CUNIT2','CDELT1', 'CDELT2','LONPOLE', 'LATPOLE', 'RADESYS', 'EQUINOX']
        method = None
        if ra_deg is not None and dec_deg is not None and arcsec_per_pixel is not None:
            method = "RADEC"
        if WCS_header is not None:
            method = "WCS"
        if method is None:
            method = "NONE"
        # Generate the WCS header if needed, write it into the image header
        if method == "RADEC":
            try:
                img_data = fits.getdata(img_path).astype(np.float32)#.byteswap().newbyteorder()
                bkg = sep.Background(img_data)
            except:
                img_data = fits.getdata(img_path).byteswap().newbyteorder().astype(np.float32)#
                bkg = sep.Background(img_data)                
            data_sub = img_data-bkg
            dir_data = os.path.join(self.cache_dir,'astrometry_data')
            objects = sep.extract(data_sub,sextractor_dict_param["DETECT_THRESH"],err=bkg.rms(),minarea=sextractor_dict_param["DETECT_MINAREA"])
            # print(objects['flux'])
            objects.sort(order='flux')
            # print(objects['flux'])
            # print(objects['flux'][-60:])
            pos_stars = np.hstack([objects["x"].reshape(-1,1),objects["y"].reshape(-1,1)])
            print(pos_stars)
            solvers = ['astrometry.Solver(astrometry.series_4100.index_files(cache_directory=dir_data,scales=[7,8,9,10,11,12,13,14],))'
                   ,
                   'astrometry.Solver(astrometry.series_4200.index_files(cache_directory=dir_data,scales=[4,6,7,8,9,10],))',
                     'astrometry.Solver(astrometry.series_5200.index_files(cache_directory=dir_data,scales=[5,6],))']
            for solver_str in solvers:
                solver = eval(solver_str)
                print('resolving astrometry using astrometry.net')
                print(ra_deg,dec_deg,arcsec_per_pixel)
                solution = solver.solve(
                            stars=pos_stars[-60:],
                            size_hint=astrometry.SizeHint(
                                lower_arcsec_per_pixel=arcsec_per_pixel*0.9,
                                upper_arcsec_per_pixel=arcsec_per_pixel*1.1,
                            ),
                            position_hint=astrometry.PositionHint(
                        ra_deg=ra_deg,
                        dec_deg=dec_deg,
                        radius_deg=1,
                    ),
                    solution_parameters=astrometry.SolutionParameters(
sip_order = 0,tune_up_logodds_threshold = None
                ))
                if solution.has_match():
                    break
                else:
                    print('failed, tyring next solver')
            if not solution.has_match():

                print('All solver failed!')
                return 0
            wcs = astropy.wcs.WCS(solution.best_match().wcs_fields)
            #print("Before flush")
            #print(fits.getheader(img_path))
            with fits.open(img_path, mode='update') as hdul:
                hdr = hdul[0].header
                NAXIS1 = hdr["NAXIS1"]
                NAXIS2 = hdr["NAXIS2"]
               
                for key in WCS_KEYWORDS:
                    if key in hdr:
                        #print('deleted',key)
                        del hdr[key]
                hdul.flush()
                #print(wcs.to_header())
                hdr.update(wcs.to_header())

                #print(hdr)
                hdul.flush()  # This writes the modified header back to disk
            #print("After flush")
            #print(fits.getheader(img_path))
            #input()
        if method == "WCS":
            with fits.open(img_path, mode='update') as hdul:
                hdr = hdul[0].header
                NAXIS1 = hdr["NAXIS1"]
                NAXIS2 = hdr["NAXIS2"]
                
                ra_deg = WCS_header_data["CRVAL1"]
                dec_deg = WCS_header_data["CRVAL2"]
                WCS_header_data = fits.getheader(WCS_header)
                hdr.update(WCS_header_data)
                arcsec_per_pixel = hdr["XPIXSZ"]/hdr["FOCALLEN"]/1000*206265
                hdul.flush()
        if method == "NONE":
            with fits.open(img_path, mode='update') as hdul:
                hdr = hdul[0].header
                NAXIS1 = hdr["NAXIS1"]
                NAXIS2 = hdr["NAXIS2"]
                ra_deg = hdr["CRVAL1"]
                dec_deg = hdr["CRVAL2"]
                arcsec_per_pixel = hdr["XPIXSZ"]/hdr["FOCALLEN"]/1000*206265

        #prepare the SExtractor parameters
        output_list = ["FLUX_APER","FLUXERR_APER","FLUX_AUTO","FLUXERR_AUTO","X_IMAGE","Y_IMAGE","X_IMAGE_DBL","Y_IMAGE_DBL","XWIN_IMAGE","YWIN_IMAGE","ERRAWIN_IMAGE","ERRBWIN_IMAGE","ERRTHETAWIN_IMAGE","ELONGATION","FLUX_RADIUS"]
        
        SExcat = self.SExtractor(img_path,sextractor_dict_param,output_list)

        #prepare the SCAMP parameters
        SCAMP_param_dict = {"ASTREF_CATALOG":"FILE"}

        # Use a more flexible parameter for SCAMP if no WCS header is provided
        if method == "RADEC":
            SCAMP_param_dict["POSANGLE_MAXERR"] = 2
            SCAMP_param_dict["POSITION_MAXERR"] = 1.0
            SCAMP_param_dict["MATCH_FLIPPED"] = "Y"
            SCAMP_param_dict["PIXSCALE_MAXERR"] = 1.05
        SCAMP_param_dict["PIXEL_SCALE"] = arcsec_per_pixel


        # Get the reference catalog
        fov = (NAXIS1**2+NAXIS2**2)**0.5*arcsec_per_pixel/3600/1.8
        ref_table = self.search_GDR3_by_circle(ra=ra_deg,dec=dec_deg,fov=fov,Gmag_limit = 17.5)
        ref_catalog = Table_to_catalog(ref_table)
        ref_catalog_path = os.path.join(self.cache_dir, ref_catalog)

        print(ref_table)
        # Save the reference catalog to a file

        SCAMP_cat = self.SCAMP(SExcat,SCAMP_param_dict,ref_catalog)
        # Get the header from the SCAMP output
        header_scamp = get_header_from_scamp_output(SCAMP_cat)
        # Update the image header with the SCAMP header
        with fits.open(img_path, mode='update') as hdul:
            hdr = hdul[0].header
            for key in WCS_KEYWORDS:
                if key in hdr:
                    del hdr[key]
            hdr.update(header_scamp)
            hdul.flush()
        # Delete the intermediate files
        subprocess.run(f"rm {SExcat}", shell=True)
        subprocess.run(f"rm {ref_catalog_path}", shell=True)
        subprocess.run(f"rm {SCAMP_cat}", shell=True)
        return header_scamp

    def SWARP(self,input_file_path,param_dict,outprefix):
        if not "WRITE_XML" in param_dict:
            param_dict["WRITE_XML"] = "N" # Not write XML by default
        if not "SUBTRACT_BACK" in param_dict:
            param_dict["SUBTRACT_BACK"] = "N" # Not subtract background by default
        if not "RESCALE_WEIGHTS" in param_dict:
            param_dict["RESCALE_WEIGHTS"] = "N" # Not rescale weights by default
        # if not "COMBINE_TYPE" in param_dict:
        #     param_dict["COMBINE_TYPE"] = "WEIGHTED" 
        param_dict["IMAGEOUT_NAME"] = outprefix+".fits"
        param_dict["WEIGHTOUT_NAME"] = outprefix+"_weight.fits"
        
        config_file_name = f"swarp_{os.getpid()}.swarp"
        config_file_path = os.path.join(self.cache_dir,config_file_name)
        command = "swarp -dd"+self.generate_parameter_pipe(param_dict,config_file_path)
        subprocess.run(command, shell=True)

        command = f"swarp {input_file_path} -c {config_file_path}"
        subprocess.run(command, shell=True)
        # Delete the parameter and config files
        subprocess.run(f"rm {config_file_path}", shell=True)
        return outprefix+".fits",outprefix+"_weight.fits"

        

    def SWARP_stack(self,image_path_list,param_dict,weight_path_list = None, weight_number_list = None, delete_weight = True, target_prefix = None,rot_deg = 0):
        def create_constant_weight(input_fits, weight_value, output_weight_fits):
            data = fits.getdata(input_fits)
            hdu = fits.PrimaryHDU(np.full_like(data, weight_value, dtype=np.float32))
            hdu.writeto(output_weight_fits, overwrite=True)

        if target_prefix is None:
            target_prefix = os.path.join(self.cache_dir,f"swarp_out_{os.getpid()}")
        use_weight = False
        if not (weight_path_list is None) or not (weight_number_list is None):
            param_dict["COMBINE_TYPE"] = "WEIGHTED" 
            use_weight = True
            # Generate the weight list file if only number is provided
            if weight_path_list is None:
                weight_path_list = []
                for i in range(len(image_path_list)):
                    weight_path = os.path.join(self.cache_dir,f"weight_{os.getpid()}_{i}.fits")
                    create_constant_weight(image_path_list[i], weight_number_list[i], weight_path)
                    weight_path_list.append(weight_path)
            
            # Generate the weight list file
            # Write the weight list into file, one path per line
            temp_weight_file_list_path = os.path.join(self.cache_dir,f"swarp_weight_list_{os.getpid()}.txt")
            with open(temp_weight_file_list_path, "w") as f:
                for weight_path in weight_path_list:
                    f.write(weight_path + "\n")
            param_dict["WEIGHT_IMAGE"] = "@"+temp_weight_file_list_path
             
        
        # Generate the input list file
        # Write the input list into file, one path per line
        temp_input_file_list_path = os.path.join(self.cache_dir,f"swarp_input_list_{os.getpid()}.txt")
        with open(temp_input_file_list_path, "w") as f:
            for image_path in image_path_list:
                f.write(image_path + "\n")
    
        # No additional header is needed, remove the header file in cache
        if os.path.exists(target_prefix+".head"):
            os.remove(target_prefix+".head")
        # The default WCS of the result image is the same as the first image
        # Write this WCS into target_prefix+".head"
        header_origin = fits.getheader(image_path_list[0])
        wcs_out = WCS(fits.getheader(image_path_list[0]))


        if rot_deg == 0:
            header_out = wcs_out.to_header()
            header_out['NAXIS'] = 2
            header_out['NAXIS1'] = header_origin['NAXIS1']
            header_out['NAXIS2'] = header_origin['NAXIS2']
        else:
            header_out = fits.Header()
            # Obtain the CD components and rotate according to rot_deg
            cd1_1 = header_origin['CD1_1']
            cd1_2 = header_origin['CD1_2']
            cd2_1 = header_origin['CD2_1']
            cd2_2 = header_origin['CD2_2']
            # Calculate the rotation matrix
            rotation_matrix = np.array([[np.cos(np.radians(rot_deg)), -np.sin(np.radians(rot_deg))],
                                         [np.sin(np.radians(rot_deg)), np.cos(np.radians(rot_deg))]])
            # Rotate the CD components
            cd1_1_rotated, cd1_2_rotated = np.dot(rotation_matrix, [cd1_1, cd1_2])
            cd2_1_rotated, cd2_2_rotated = np.dot(rotation_matrix, [cd2_1, cd2_2])
            # Update the header with the rotated CD components
            header_out['CD1_1'] = cd1_1_rotated
            header_out['CD1_2'] = cd1_2_rotated
            header_out['CD2_1'] = cd2_1_rotated
            header_out['CD2_2'] = cd2_2_rotated

            header_out['NAXIS1'] = int(np.abs(np.cos(np.radians(rot_deg))*header_origin['NAXIS1'])+np.abs(np.sin(np.radians(rot_deg))*header_origin['NAXIS2']))+1
            header_out['NAXIS2'] = int(np.abs(np.sin(np.radians(rot_deg))*header_origin['NAXIS1'])+np.abs(np.cos(np.radians(rot_deg))*header_origin['NAXIS2']))+1

            # Update the CRPIX values
            coord = wcs_out.all_pix2world([[header_origin['NAXIS1']//2,header_origin['NAXIS2']//2]],0)
            ra,dec = coord[0]
            print('center of image:',ra,dec)
            header_out['CRPIX1'] = header_out['NAXIS1']//2
            header_out['CRPIX2'] = header_out['NAXIS2']//2
            # Update the CRVAL values
            header_out['CRVAL1'] = ra
            header_out['CRVAL2'] = dec
            print(header_out)
        header_out.tofile(target_prefix+".head",overwrite=True)

        output_image_file_path, output_weight_file_path = self.SWARP("@"+temp_input_file_list_path,param_dict,target_prefix)
        # Delete the temporary input file
        os.remove(temp_input_file_list_path)
        # Delete the temporary weight file
        if use_weight:
            os.remove(temp_weight_file_list_path)
            if delete_weight:
                for weight_path in weight_path_list:
                    os.remove(weight_path)
            

        return output_image_file_path, output_weight_file_path

        