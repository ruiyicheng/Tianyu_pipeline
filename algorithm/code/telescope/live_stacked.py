import os
import sys
import threading
import time
import glob
import re
import queue
import numpy as np
import subprocess
import pandas as pd
from astropy.time import Time
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astroquery.gaia import Gaia
import sep
import astrometry # Assumed to be installed

# ==============================================================================
#  Provided Bertin's Tools Interface
#  (Copied as-is from your request)
# ==============================================================================

class Bertin_tools:
    def __init__(self,mode = "consumer_component",cache_dir = '/mnt/hgfs/algorithm/data/testoutput/temp'):
        '''
        if used indepentently, set mode = "independent" and cache_dir = path to the cache directory
        '''
        super().__init__()
        self.mode = mode
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
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
                re_remove = f'^{k}  *[:/_.0-9a-zA-Z,]*'
                cmd += f" -e 's|{re_remove}|{k} {parameter_dict[k]} #|g' "
        cmd += f" > {exportfilename}"
        return cmd
    def generate_output_sextractor(self,parameter_list,exportfilename):
        cmd = ""
        if len(parameter_list)>0:
            cmd = " | sed"
            for item in parameter_list:
                if type(item) == str:
                    cmd += f" -e 's/^#{item}/{item} #/g' "
                else:
                    for k in item:
                        cmd += f" -e 's/^#{k}/{item[k]} #/g' "
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
        if not "DETECT_THRESH" in param_dict:
            param_dict["DETECT_THRESH"] = 5
        if not "DETECT_MINAREA" in param_dict:
            param_dict["DETECT_MINAREA"] = 5
        if not "ANALYSIS_THRESH" in param_dict:
            param_dict["ANALYSIS_THRESH"] = 5
        if not "PHOT_APERTURES" in param_dict:
            param_dict["PHOT_APERTURES"] = "15"
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
        def Table_to_catalog(table):
            header_str = "\n".join([
                "SIMPLE  =                    T /", "BITPIX  =                   16 /",
                "NAXIS   =                    2 /", "NAXIS1  =                 2048 /",
                "NAXIS2  =                 2048 /", "END"
            ])
            header_lines_padded = [line.ljust(80) for line in header_str]
            header_array = np.array(header_lines_padded, dtype='S80')

            col1 = fits.Column(name='Field Header Card', format='80A', array=header_array)
            hdu1 = fits.BinTableHDU.from_columns([col1], name='LDAC_IMHEAD')
            
            X_WORLD, Y_WORLD = table['ra'], table['dec']
            ERRA_WORLD, ERRB_WORLD  = table['ra_error']/3600000, table['dec_error']/3600000
            MAG, MAG_ERR = table['phot_g_mean_mag'], (2.5/np.log(10))/table['phot_g_mean_flux_over_error']

            cols = [
                fits.Column(name='X_WORLD', format='D', array=X_WORLD),
                fits.Column(name='Y_WORLD', format='D', array=Y_WORLD),
                fits.Column(name='ERRA_WORLD', format='D', array=ERRA_WORLD),
                fits.Column(name='ERRB_WORLD', format='D', array=ERRB_WORLD),
                fits.Column(name='MAG', format='D', array=MAG),
                fits.Column(name='MAGERR', format='D', array=MAG_ERR)
            ]
            hdu2 = fits.BinTableHDU.from_columns(cols, name='LDAC_OBJECTS')
            hdul = fits.HDUList([fits.PrimaryHDU(), hdu1, hdu2])
            ref_catalog_name = f"ref_catalog_{os.getpid()}.fits"
            ref_catalog_path = os.path.join(self.cache_dir, ref_catalog_name)
            hdul.writeto(ref_catalog_path, overwrite=True)
            return ref_catalog_name

        def get_header_from_scamp_output(scamp_output_path):
            with open(scamp_output_path, "r") as f:
                head_data = f.read()
            head_split = head_data.split('\n')
            s = '\n'.join(head_split[3:])
            return fits.Header.fromstring(s, sep='\n')

        # This implementation uses astrometry.net via the python client for a robust initial guess
        print(f"Solving WCS for {img_path} with astrometry.net client...")
        try:
            img_data = fits.getdata(img_path).astype(np.float32)
            bkg = sep.Background(img_data)
        except (ValueError, TypeError):
            img_data = fits.getdata(img_path).byteswap().newbyteorder().astype(np.float32)
            bkg = sep.Background(img_data)
        
        data_sub = img_data - bkg
        objects = sep.extract(data_sub, sextractor_dict_param.get("DETECT_THRESH", 15.0), err=bkg.rms(), minarea=sextractor_dict_param.get("DETECT_MINAREA", 5))
        objects.sort(order='flux')
        pos_stars = np.hstack([objects["x"].reshape(-1,1), objects["y"].reshape(-1,1)])
        print(pos_stars[-30:])
        # Use astrometry.net for initial WCS
        # Note: This requires astrometry.net index files to be downloaded.
        dir_data = os.path.join(self.cache_dir, 'astrometry_data')
        solver = astrometry.Solver(
            astrometry.series_4100.index_files(cache_directory=dir_data, scales={11,12,13, 14, 15,16})+astrometry.series_4200.index_files(cache_directory=dir_data, scales={11,12, 13, 14, 15,16}) 
            #+ astrometry.series_5200.index_files(cache_directory=dir_data) 
            
        )
        
        solution = solver.solve(
            stars=pos_stars[-50:], # Use top 100 brightest stars
            size_hint=astrometry.SizeHint(
                lower_arcsec_per_pixel=arcsec_per_pixel * 0.8,
                upper_arcsec_per_pixel=arcsec_per_pixel * 1.2,
            ),
            position_hint=astrometry.PositionHint(
                ra_deg=ra_deg, dec_deg=dec_deg, radius_deg=2
            ), solution_parameters=astrometry.SolutionParameters(),
        )

        if not solution.has_match():
            print("Astrometry.net failed to find a solution.")
            return None
        else:
            print("Astrometry.net found a solution.")
        
        # We have a basic WCS, now refine with SCAMP
        wcs_initial = WCS(solution.best_match().wcs_fields)
        
        with fits.open(img_path, mode='update') as hdul:
            hdul[0].header.update(wcs_initial.to_header())
            hdul.flush()

        # Prepare for SCAMP
        output_list = ["FLUX_APER","FLUXERR_APER","FLUX_AUTO","FLUXERR_AUTO","X_IMAGE","Y_IMAGE","X_IMAGE_DBL","Y_IMAGE_DBL","XWIN_IMAGE","YWIN_IMAGE","ERRAWIN_IMAGE","ERRBWIN_IMAGE","ERRTHETAWIN_IMAGE","ELONGATION","FLUX_RADIUS"]
        
        SExcat = self.SExtractor(img_path, sextractor_dict_param, output_list)
        
        SCAMP_param_dict = {"ASTREF_CATALOG": "FILE"}
        NAXIS1, NAXIS2 = img_data.shape[1], img_data.shape[0]
        fov = (NAXIS1**2 + NAXIS2**2)**0.5 * arcsec_per_pixel / 3600 / 1.8
        ref_table = self.search_GDR3_by_circle(ra=ra_deg, dec=dec_deg, fov=fov, Gmag_limit=18.0)
        ref_catalog = Table_to_catalog(ref_table)
        
        # Run SCAMP to get a high-precision header
        SCAMP_head_file = self.SCAMP(SExcat, SCAMP_param_dict, ref_catalog)
        header_scamp = get_header_from_scamp_output(SCAMP_head_file)

        # Cleanup
        subprocess.run(f"rm {SExcat}", shell=True)
        subprocess.run(f"rm {os.path.join(self.cache_dir, ref_catalog)}", shell=True)
        subprocess.run(f"rm {SCAMP_head_file}", shell=True)
        
        return header_scamp

# ==============================================================================
#  New Simulation and Processing Logic
# ==============================================================================

def solve_wcs_worker(temp_fits_path, temp_img_data,bertin_instance, initial_ra, initial_dec, pixel_scale, wcs_queue, wcs_ready_event):
    """
    Worker function to run WCS solving in a separate thread.
    """
    try:
        print(f"[{threading.current_thread().name}] Starting WCS solution for {temp_fits_path}...")
        fits.writeto(temp_fits_path, temp_img_data, overwrite=True)
        sextractor_params = {"DETECT_THRESH": 10.0, "ANALYSIS_THRESH": 10.0, "DETECT_MINAREA": 5}
        
        wcs_header = bertin_instance.SCAMP_image(
            img_path=temp_fits_path,
            sextractor_dict_param=sextractor_params,
            ra_deg=initial_ra,
            dec_deg=initial_dec,
            arcsec_per_pixel=pixel_scale
        )
        
        if not wcs_header:
             print(f"[{threading.current_thread().name}] WCS solution failed.")
             wcs_queue.put(None)
        else:
            print(f"[{threading.current_thread().name}] WCS solution successful.")
            wcs_queue.put(wcs_header)

    except Exception as e:
        print(f"[{threading.current_thread().name}] An error occurred during WCS solving: {e}")
        wcs_queue.put(None)
    finally:
        wcs_ready_event.set() # Signal that the attempt is complete (success or fail)

def simulate_observation_and_process(
    input_dir,
    output_base_dir,
    target_name,
    sub_targets,
    initial_ra,
    initial_dec,
    pixel_scale,
    n_image_total,
    exposure_s=0.3,
    wcs_trigger_count=5,
    chop_size=16
):
    """
    Simulates observation, live stacking, WCS solving, and chopping.
    """
    # 1. Setup
    bertin_cache_dir = os.path.join(output_base_dir, 'temp_cache')
    bertin = Bertin_tools(cache_dir=bertin_cache_dir)
    # glob.glob('')
    total_raw_images = 150
    print(f"Simulating for {0.3}s, expecting ~{total_raw_images} raw images.")

    # 2. File Discovery and Grouping
    search_pattern = os.path.join(input_dir, f"{target_name}-image-*.fits")
    all_files = sorted(glob.glob(search_pattern))
    
    if not all_files:
        print(f"Error: No files found matching pattern '{search_pattern}'")
        return

    files_by_stack = {}
    for f in all_files:
        match = re.search(r'-image-(\d+)-(\d+)\.fits', os.path.basename(f))
        if match:
            n_image = int(match.group(1))
            if n_image not in files_by_stack:
                files_by_stack[n_image] = []
            files_by_stack[n_image].append(f)

    print(f"Found {len(all_files)} files, grouped into {len(files_by_stack)} stacks.")
    def get_key(path_name):
        num_this = int(path_name.split('-')[-1].split('.')[0])
        return num_this

    # 3. Main Simulation Loop
    for n_image_this in range(1, n_image_total + 1):
        n_image = n_image_this
        image_stacks = {}
        raw_images_by_stack = {}
        jd_lists_by_stack = {}

        wcs_solution_queue = queue.Queue()
        wcs_ready_event = threading.Event()
        wcs_thread_started = False
        raw_image_counter = 0
        first_stack_key = sorted(files_by_stack.keys())[0] if files_by_stack else None

        flat_file_list = []

        flat_file_list.extend(sorted(files_by_stack[n_image], key=get_key))


        for file_path in flat_file_list:#[:total_raw_images]:
            print(f"Processing raw image {raw_image_counter + 1}/{total_raw_images}: {os.path.basename(file_path)}", end='\r')
            time.sleep(exposure_s)
            
            # match = re.search(r'-image-(\d+)-', os.path.basename(file_path))
            # n_image = int(match.group(1))
            
            with fits.open(file_path) as hdul:
                img_data = np.squeeze(hdul[0].data.astype(np.int32))
                header = hdul[0].header
                jd = header.get('JD', Time.now().jd)
                
                if n_image not in image_stacks:
                    image_stacks[n_image] = np.zeros_like(img_data, dtype=np.int32)
                    raw_images_by_stack[n_image] = []
                    jd_lists_by_stack[n_image] = []

                image_stacks[n_image] += img_data
                raw_images_by_stack[n_image].append(img_data)
                jd_lists_by_stack[n_image].append(jd)

            raw_image_counter += 1
            print(f"Processed {raw_image_counter} raw images so far.", end='\r')
            # 4. Trigger WCS Thread
            if raw_image_counter == wcs_trigger_count and not wcs_thread_started:
                print("\n--- Triggering asynchronous WCS solution ---")
                
                wcs_thread_started = True
                

                temp_fits_path = os.path.join(bertin.cache_dir, 'wcs_temp_stack.fits')
                temp_img_data = np.squeeze(image_stacks[n_image].copy().astype(np.float32))
                #fits.writeto(temp_fits_path, np.squeeze(image_stacks[first_stack_key].astype(np.float32)), overwrite=True)
                
                wcs_thread = threading.Thread(
                    target=solve_wcs_worker, name="WCS-Solver",
                    args=(temp_fits_path,temp_img_data, bertin, initial_ra, initial_dec, pixel_scale, wcs_solution_queue, wcs_ready_event)
                )
                wcs_thread.daemon = True
                wcs_thread.start()
                # else:
                #     print("Warning: Cannot start WCS thread, no data in first stack yet.")
                #     wcs_ready_event.set()

        print("\n--- All raw images processed. Proceeding to final chopping. ---")

        # 5. Post-Processing: Wait for WCS and Chop
        print("Waiting for WCS solution to be ready...")
        wcs_ready_event.wait(timeout=600) # Wait up to 10 minutes
        
        if wcs_solution_queue.empty():
            print("Fatal: WCS solver timed out or failed to return a result.")
            return

        wcs_header = wcs_solution_queue.get()
        if wcs_header is None:
            print("Fatal: WCS solution failed. Cannot proceed with chopping.")
            return

        final_wcs = WCS(wcs_header)
        print("WCS solution obtained. Converting target coordinates to pixels.")
        ra_deg = sub_targets['ra'].values
        dec_deg = sub_targets['dec'].values
        sub_targets_radec = np.column_stack((ra_deg, dec_deg))
        target_pixels = final_wcs.all_world2pix(sub_targets_radec, 0)
        stacked_image = image_stacks.get(n_image, None)

        print(f"\n--- Processing stack n_image={n_image} ---")
        output_dir_n_image = os.path.join(output_base_dir, target_name, f"stack_{n_image}")
        os.makedirs(output_dir_n_image, exist_ok=True)
        
        final_header = fits.Header()
        final_header['N_FRAMES'] = len(raw_images_by_stack[n_image])
        final_header['JD_START'] = jd_lists_by_stack[n_image][0]
        final_header.update(final_wcs.to_header())
        
        stacked_fits_path = os.path.join(output_dir_n_image, f"stacked_image_{n_image}.fits")
        fits.writeto(stacked_fits_path, stacked_image.astype(np.int32), final_header, overwrite=True)
        print(f"Saved full stacked image to {stacked_fits_path}")

        time_table = Table([jd_lists_by_stack[n_image]], names=['JD'])
        time_fits_path = os.path.join(output_dir_n_image, f"times_{n_image}.fits")
        time_table.write(time_fits_path, format='fits', overwrite=True)
        print(f"Saved time data to {time_fits_path}")
        
        half_size = chop_size // 2

        ymax, xmax = raw_images_by_stack[n_image][0].shape
        print('Saving chopping image')
        for i, (ra, dec) in enumerate(sub_targets_radec):
            x, y = target_pixels[i]
            x_int, y_int = int(round(x+0.5)), int(round(y+0.5))
            
            #print(f"Chopping target {i+1} (RA={ra}, Dec={dec}) at pixel ({x:.2f}, {y:.2f})")
            # for raw in raw_images_by_stack[n_image]:
            #     print(raw.shape)
            if x_int - half_size < 0 or x_int + half_size >= xmax or y_int - half_size < 0 or y_int + half_size >= ymax:
                continue
            chopped_cube = [raw[y_int-half_size:y_int+half_size, x_int-half_size:x_int+half_size] for raw in raw_images_by_stack[n_image]]
            chopped_cube = np.array(chopped_cube, dtype=np.int32)
            
            chop_header = fits.Header()
            chop_header['RA_TARG'] = (ra, 'Target Right Ascension (deg)')
            chop_header['DEC_TARG'] = (dec, 'Target Declination (deg)')
            chop_header['X_CEN'] = (x-0.5, 'Calculated X pixel center (0-based)')
            chop_header['Y_CEN'] = (y-0.5, 'Calculated Y pixel center (0-based)')
            chop_header['N_IMAGE'] = n_image
            
            chop_filename = f"chopped_target-image{n_image}_{i+1}_radec_{ra:.4f}_{dec:.4f}.fits"
            chop_filepath = os.path.join(output_dir_n_image, chop_filename)
            fits.writeto(chop_filepath, chopped_cube, chop_header, overwrite=True)
            #print(f"Saved chopped cube to {chop_filepath}")
        print('Chopped image saved')
    print("\nSimulation and processing complete.")

def create_dummy_fits_files(directory, target_name, num_stacks, num_raw_per_stack, shape=(8192, 8192)):
    """Creates dummy FITS files for testing."""
    if os.path.exists(directory):
        print("Dummy data directory already exists. Skipping creation.")
        return
    os.makedirs(directory, exist_ok=True)
    print(f"Creating dummy FITS files in {directory}...")
    start_jd = Time.now().jd
    for i in range(num_stacks):
        for j in range(num_raw_per_stack):
            n_image = i + 1
            n_raw = j
            filename = os.path.join(directory, f"{target_name}-image-{n_image}-{n_raw}.fits")
            
            data = np.random.randint(100, 500, size=shape, dtype=np.int32)
            # Add a few fake stars for astrometry to find
            for _ in range(20):
                x_star, y_star = np.random.randint(200, shape[0]-200, 2)
                data[y_star-5:y_star+5, x_star-5:x_star+5] = 20000
            
            header = fits.Header()
            header['JD'] = start_jd + (i * num_raw_per_stack + j) * (0.3 / 86400.0)
            header['EXPTIME'] = 0.3
            
            fits.writeto(filename, data, header, overwrite=True)
    print("Dummy files created.")


if __name__ == "__main__":
    # === Configuration ===
    # --- Part 0: Create a dummy dataset for demonstration ---
    DUMMY_INPUT_DIR = '/mnt/hgfs/imggen/result/XO-4'
    DUMMY_TARGET_NAME = 'XO-4'
    # create_dummy_fits_files(
    #     directory=DUMMY_INPUT_DIR, 
    #     target_name=DUMMY_TARGET_NAME, 
    #     num_stacks=3, 
    #     num_raw_per_stack=50
    # )

    # --- Part 1: Set up the simulation parameters ---
    INPUT_DATA_DIR = DUMMY_INPUT_DIR
    OUTPUT_BASE_DIR = '/mnt/hgfs/imggen/stacked_chopped'
    TARGET_NAME = DUMMY_TARGET_NAME
    
    # These should be the true coordinates of the field center and the pixel scale
    # This is a mock value. You MUST replace it with the actual center of your images.
    FIELD_CENTER_RA = 110.388167
    FIELD_CENTER_DEC = 58.268086
    PIXEL_SCALE_ARCS = 1.33

    candidate_sub_pixel_dir = '/mnt/hgfs/imggen/code/Tianyu_simulator/photometric_iamge_simulator/sim_events/all_candidates.csv'
    df_sub_pixel = pd.read_csv(candidate_sub_pixel_dir)
    # List of interesting targets to chop out
    # SUB_TARGETS_RADEC = np.array([
    #     [180.001, 25.001], # Target 1
    #     [179.995, 24.998]  # Target 2
    # ])
    
    # --- Part 2: Run the simulation ---
    simulate_observation_and_process(
        input_dir=INPUT_DATA_DIR,
        output_base_dir=OUTPUT_BASE_DIR,
        target_name=TARGET_NAME,
        sub_targets=df_sub_pixel,
        initial_ra=FIELD_CENTER_RA,
        initial_dec=FIELD_CENTER_DEC,
        pixel_scale=PIXEL_SCALE_ARCS,
        n_image_total = 5,
        exposure_s=0.3,
        wcs_trigger_count=10,
        chop_size=16
    )