# Define the class of telescope hardware
import numpy as np
from astropy.time import Time
import pandas as pd
from astropy.coordinates import SkyCoord, AltAz
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.coordinates import get_body
from astropy.io import fits
from astropy import wcs
import cv2
from astroquery.gaia import Gaia
from astropy.modeling.models import Moffat2D
import batman
class hardware:
    def __init__(self,hardware_type,name = 'simulator'):
        self.connected = False
        self.name = name
        self.hardware_type = hardware_type
        self.telescope = None
        self._status = 'idle'
    @property
    def status(self):
        if "simulator" in self.name:            # Calculate the status according to calculation
            if self.telescope.world.time > self.idle_time:
                self._status = 'idle'
            return self._status
        else:                                   # read the status from instrumentation
            pass
    def connect(self,telescope):
        if 'simulator' in self.name:
            self.connected = True
            self.telescope = telescope
            print(self.hardware_type,self.name,'Connected to the telescope')
            for i in self.initpar:
                self.set_parameter(i,self.initpar[i])
    def set_parameter(self,attribute,value):
        if self.connected:
            if 'simulator' in self.name:
                setattr(self,attribute,value)
            else:
                setattr(self,attribute,value)
                # SET according to the hardware
                pass
        else:
            raise   RuntimeError('Please connect the hardware')

    def get_parameter(self,attribute):
        if hasattr(self,attribute):
            if 'simulator' in self.name:
                return attribute
            else:
                # read according to the hardware update the attribute in class
                pass
        else:
            raise   RuntimeError('Please connect the hardware')  
# These two classes would return the status of the instruments.

        
# pixel_size_mum, pixel_number, readout_time_s, gain, dark_current_e_s_1, full_well_capacity_e, read_noise_e     
class camera(hardware):
    def __init__(self,par_camera,name = 'simulator_cam'):
        super().__init__('camera',name = name)
        self._status = 'idle'
        self.initpar = par_camera
        self.idle_time = -1
    def capture(self,exposure_time):
        if self.status == 'busy':
            print('Camera is busy')
            return 0
        if 'simulator' in self.name:
            self._status = 'busy'
            self.exposure_s = exposure_time
            self.idle_time = self.telescope.world.time + np.max([exposure_time,self.readout_time_s])/24/3600
            print('Capturing until',self.idle_time)

    def download(self,generated_img = -1):
        if self.status != 'idle':
            print('Camera is busy')
            return 0
        if 'simulator' in self.name:
            print('Image is loaded')
            return generated_img

# tracking_mode, tracking_speed_deg_s_1, stable_time_s, goto_error_arcsec, tracking_error_arcsec_min_1
class mount(hardware):
    def __init__(self,par_mount,name = 'simulator_mount'):
        super().__init__('mount',name = name)
        self._status = 'idle'                # status = idle/running/goto
        self.initpar = par_mount
        self.ra_deg = 0
        self.dec_deg = 0
        self.idle_time = -1
    def goto(self,ra_deg,dec_deg):                      # goto an ra,dec position
        if self.status == 'busy':
            print('Mount is busy')
            return 0
        if 'simulator' in self.name:
            if self.tracking_mode == 'alt-az':
                self._status = 'busy'
                dra = self.ra_deg-ra_deg-((self.ra_deg-ra_deg)//360)*360
                dra = np.min([dra,360-dra])*np.cos(np.min(np.abs([self.dec_deg,dec_deg]))/180*np.pi)
                ddec = np.abs(self.dec_deg-dec_deg)
                time4goto = (np.sqrt(dra**2+ddec**2))
                self.idle_time = self.telescope.world.time +(self.stable_time_s+ time4goto/self.tracking_speed_deg_s_1)/24/3600
                print('Mount is moving to',ra_deg,dec_deg,'until',self.idle_time,'Using',self.stable_time_s+time4goto/self.tracking_speed_deg_s_1,'s')
                self.ra_deg = ra_deg + np.random.randn()*self.goto_error_arcsec/3600
                self.dec_deg = dec_deg + np.random.randn()*self.goto_error_arcsec/3600

                



#parameter of telescope = position, seeing, focus, diameter

class telescope:
    def __init__(self,mount,camera,par_telescope ,name = 'simulator_telescope'):
        self.name = name
        self.mount = mount
        self.camera = camera
        
        for i in par_telescope:
            setattr(self,i,par_telescope[i])
        self.mount.connect(self)
        self.camera.connect(self)
        self.arcsec_pixel_1 = self.camera.pixel_size_mum/self.focal_length_m * 206265/1e6 #arcsec
        self.fov_x = self.camera.pixel_number_x * self.arcsec_pixel_1/3600   #degree
        self.fov_y = self.camera.pixel_number_y * self.arcsec_pixel_1/3600
        self.fov_diag = np.sqrt(self.fov_x**2+self.fov_y**2)
    @property
    def position(self):
        if not hasattr(self, '_position'):
            self._position = EarthLocation.from_geodetic(lat=self.latlonalt[0]*u.deg, lon=self.latlonalt[1]*u.deg, height=self.latlonalt[2]*u.m)
        return self._position

    @property
    def wcs(self,rot_deg = 0):
        theta = rot_deg/180*np.pi
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [self.camera.pixel_number_x//2, self.camera.pixel_number_y//2]
        w.wcs.cdelt = np.array([self.arcsec_pixel_1/3600, self.arcsec_pixel_1/3600])
        w.wcs.crval = [self.mount.ra_deg, self.mount.dec_deg]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.pc = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        return w
    
    def wcs_to_target(self,ra_target,dec_target):
        ra_rad = self.mount.ra_deg * np.pi / 180
        dec_rad = self.mount.dec_deg * np.pi / 180
        ra_target_rad = ra_target * np.pi / 180
        dec_target_rad = dec_target * np.pi / 180
        pos_self = np.array([np.cos(dec_rad)*np.cos(ra_rad),np.cos(dec_rad)*np.sin(ra_rad),np.sin(dec_rad)])
        R_z_RA1 = np.array([[np.cos(ra_target_rad),np.sin(ra_target_rad),0],[-np.sin(ra_target_rad),np.cos(ra_target_rad),0],[0,0,1]])
        R_y_dec1 = np.array([[np.cos(dec_target_rad),0,-np.sin(dec_target_rad)],[0,1,0],[np.sin(dec_target_rad),0,np.cos(dec_target_rad)]])
        pos_target = np.dot(np.dot(R_z_RA1,R_y_dec1),pos_self)
        ra_new = np.arctan2(pos_target[1],pos_target[0])
        dec_new = np.arcsin(pos_target[2])
        theta_new = np.arccos(pos_target[2])
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [self.camera.pixel_number_x//2, self.camera.pixel_number_y//2]
        w.wcs.cdelt = np.array([self.arcsec_pixel_1/3600, self.arcsec_pixel_1/3600])
        w.wcs.crval = [ra_new * 180 / np.pi, dec_new * 180 / np.pi]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.pc = np.array([[np.cos(theta_new),-np.sin(theta_new)],[np.sin(theta_new),np.cos(theta_new)]])
        return w
    def build(self,world):
        self.world = world
        print("Telecope is built in world",world.name)



#parameter of world = t0
class world:
    def __init__(self,telescope,t0,name = 'simulator_world',input_schedule = None,input_event = None):
        self.name = name
        self.telescope = telescope
        self.telescope.build(self)
        if 'simulator' in self.name:
            self.img_simulator = image_simulator()
            self.img_simulator.connect(self)
        self._time = t0
        if input_schedule is not None:
            self.schedule = pd.read_csv(input_schedule)
        self.events_world = input_event

    @property
    def time(self):
        if 'simulator' in self.name:
            return self._time
        else:
            self._time = Time.now().jd
            return self._time

    
    def load_event(self,fp,type):
        pass

    def run_sim(self):
        def wait_to_ok(device):
            epsilon = 1e-9
            if device.status == 'busy':
                return device.idle_time+epsilon
            else:
                return self._time
        
        if not 'simulator' in self.name:
            raise   RuntimeError('Simulator only works in simulator world')
        for i,r in self.schedule.iterrows():
            if self.time > r['jd_utc_end']:
                continue
            else:
                if self.time < r['jd_utc_begin']: # wait until the schedule starts
                    self._time = r['jd_utc_begin']

                if r['target_name']!='bias':
                    if r['ra']==r['ra'] and r['dec']==r['dec']:
                        ra = r['ra']
                        dec = r['dec']
                        
                    if r['alt']==r['alt'] and r['azi']==r['azi']:
                        alt = r['alt']
                        azi = r['azi']
                        coord = SkyCoord(alt=alt*u.deg, az=azi*u.deg, obstime=Time(self._time,scale='utc',format='jd'), frame='altaz', location=self.telescope.position)
                        ra_dec = coord.transform_to('icrs')
                        ra = ra_dec.ra.deg
                        dec = ra_dec.dec.deg
                        
                    self.telescope.mount.goto(ra, dec)
                self._time = wait_to_ok(self.telescope.mount)
                ct = 0
                while self.time < r['jd_utc_end'] and (r['n_max_frames']<=0 or ct< r['n_max_frames']):
                    ct += 1
                    self.telescope.camera.capture(r['exposure_s'])
                    self._time = wait_to_ok(self.telescope.camera)
                    image = self.telescope.camera.download(self.img_simulator.generate_img())
                    
                    print(r['target_name'],'captured at',self.time)
                    self._time = self._time + r['delay_between_frame_s']/24/3600
                # preprocessing image (cut pixel, stacking)
                # push it into Rabbitmq
                
                # TBD
                ################################################################



    def run_real(self):
        if 'simulator' in self.name:
            raise   RuntimeError('Real world only works in real world')
        



class image_simulator:
    def __init__(self):
        pass
    def connect(self,world):
        self.world = world
    def read_noise(self,image, amount):
        """
        Generate simulated read noise.
        
        Parameters
        ----------
        
        image: numpy array
            Image whose shape the noise array should match.
        amount : float
            Amount of read noise, in electrons.
        """
        shape = image.shape
        
        noise = np.random.normal(scale=amount, size=shape)
        
        return noise
    def bias(self,image, value, realistic=False):
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

            columns = np.random.randint(1, shape[1], size=number_of_colums)
            # This adds a little random-looking noise into the data.
            col_pattern = np.random.randint(0, int(0.1 * value), size=shape[0])
            
            # Make the chosen columns a little brighter than the rest...
            for c in columns:
                bias_im[:, c] = value + col_pattern
                
        return bias_im
    def dark_current(self,image, current, exposure_time, hot_pixels=False):
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
        strength : float, optional
            Pixel count in the cosmic rays.    
        """
        
        # dark current for every pixel; we'll modify the current for some pixels if 
        # the user wants hot pixels.
        base_current = current * exposure_time 
        
        # This random number generation should change on each call.
        dark_im = np.random.poisson(base_current, size=image.shape)
            
        if hot_pixels:
            # We'll set 0.01% of the pixels to be hot; that is probably too high but should 
            # ensure they are visible.
            y_max, x_max = dark_im.shape
            
            n_hot = int(0.0000001 * x_max * y_max)
            
            # Like with the bias image, we want the hot pixels to always be in the same places
            # (at least for the same image size) but also want them to appear to be randomly
            # distributed. So we set a random number seed to ensure we always get the same thing.
            
            hot_x = np.random.randint(1, x_max, size=n_hot)
            hot_y = np.random.randint(1, y_max, size=n_hot)
            
            hot_current = 10000 * current
            
            dark_im[(hot_y, hot_x)] = hot_current * exposure_time
        return dark_im
    def flat(self,image,path_template = '/Users/ruiyicheng/Documents/code/projects/TianYu/debug_Tianyu_file_system/supplimentary_resource/sim_image/flat_co_bias.fit',range_in_template = (500,6000, 1500,7000)):
        flat_save = fits.getdata(path_template)
        flat_re = flat_save[range_in_template[0]:range_in_template[1], range_in_template[2]:range_in_template[3]]
        flat_co = cv2.resize(flat_re.astype(np.float32), (image.shape[0], image.shape[1]), interpolation=cv2.INTER_LINEAR) 
        flat_co /= np.mean(flat_co)
        return flat_co
    def sky(self,img):# to simplify the problem, we set the image to align with dec axis
        def get_direction(ra1,dec1,ra2,dec2):
            ra1_rad = ra1 * np.pi / 180
            dec1_rad = dec1 * np.pi / 180
            ra2_rad = ra2 * np.pi / 180
            dec2_rad = dec2 * np.pi / 180
            eRA = np.array([-np.sin(ra1_rad),np.cos(ra1_rad),0]).reshape(-1,1)
            eDEC = np.array([-np.sin(dec1_rad)*np.cos(ra1_rad),-np.sin(dec1_rad)*np.sin(ra1_rad),np.cos(dec1_rad)]).reshape(-1,1)
            r1 = np.array([np.cos(dec1_rad)*np.cos(ra1_rad),np.cos(dec1_rad)*np.sin(ra1_rad),np.sin(dec1_rad)]).reshape(-1,1)
            r2 = np.array([np.cos(dec2_rad)*np.cos(ra2_rad),np.cos(dec2_rad)*np.sin(ra2_rad),np.sin(dec2_rad)]).reshape(-1,1)
            direction = r2 - r1
            comp_ra = np.squeeze(eRA.T.dot(direction))
            comp_dec = np.squeeze(eDEC.T.dot(direction))
            norm = np.sqrt(comp_ra**2+comp_dec**2)
            comp_ra,comp_dec = comp_ra/norm, comp_dec/norm
            return comp_ra,comp_dec
        #consider sun and moon
        def moon_sky(rho_moon_deg,Z,Zm,alpha): # https://arxiv.org/pdf/1304.7107
            def f(rho):
                PA = 1.5
                PB = 0.9
                rho_rad = rho/180*np.pi
                fR = 10**5.36*(1.06+np.cos(rho_rad)**2)
                if rho>=10:
                    fM = 10**(6.15-rho/40)
                else:
                    fM = 6.2*10**7*rho**(-2)
                return PA*fM+PB*fR
                
            def X(Z):
                Z_rad = Z*np.pi/180
                return (1-0.96*np.sin(Z_rad)**2)**(-0.5)
            K = 0.23
            m = -12.73+0.026*np.abs(alpha)+4*10**(-9)*alpha**4
            I_star = 10**(-0.4*(m+16.57))
            B_moon = f(rho_moon_deg)*I_star*10**(-0.4*K*X(Zm))*(1-10**(-0.4*K*X(Z)))
            V_moon = (20.7233-np.log(B_moon/34.08))/0.92104
            return V_moon
        
        time = Time(self.world.time,scale = 'utc',format = 'jd')
        # get moon phase
        Altaz_obs = AltAz(obstime=time, location=self.world.telescope.position)
        sun = get_body('sun',time)#.transform_to(Altaz_obs)
        moon = get_body('moon',time)#.transform_to(Altaz_obs)
        # Compute elongation (angular separation between Sun and Moon)
        
        elongation = sun.transform_to(Altaz_obs).separation(moon.transform_to(Altaz_obs)).deg  
        alpha = 180 - elongation
        coord_pointing = SkyCoord(ra=self.world.telescope.mount.ra_deg*u.deg, dec=self.world.telescope.mount.dec_deg*u.deg, obstime=time, frame='gcrs', location=self.world.telescope.position)#
        rho = coord_pointing.separation(moon).deg
        Zm = 90-moon.transform_to(Altaz_obs).alt.deg
        Z = 90-coord_pointing.transform_to(Altaz_obs).alt.deg
        M_moon = moon_sky(rho,Z,Zm,alpha)
        coord_pointing_ra_deviated = SkyCoord(ra=(self.world.telescope.mount.ra_deg+1/3600)*u.deg, dec=self.world.telescope.mount.dec_deg*u.deg, obstime=time, frame='gcrs', location=self.world.telescope.position)
        coord_pointing_dec_deviated = SkyCoord(ra=self.world.telescope.mount.ra_deg*u.deg, dec=(self.world.telescope.mount.dec_deg+1/3600)*u.deg, obstime =time, frame='gcrs', location=self.world.telescope.position)
        rho_ra_deviated = coord_pointing_ra_deviated.separation(moon).deg
        rho_dec_deviated = coord_pointing_dec_deviated.separation(moon).deg
        Z_ra_deviated = 90-coord_pointing_ra_deviated.transform_to(Altaz_obs).alt.deg
        Z_dec_deviated = 90-coord_pointing_dec_deviated.transform_to(Altaz_obs).alt.deg
        M_moon_ra_deviated = moon_sky(rho_ra_deviated,Z_ra_deviated,Zm,alpha)    
        M_moon_dec_deviated = moon_sky(rho_dec_deviated,Z_dec_deviated,Zm,alpha)
        dMoon_dRA = (M_moon_ra_deviated-M_moon) #mag per arcsec
        dMoon_dDEC = (M_moon_dec_deviated-M_moon) #mag per arcsec

        # Compute the AltAz coordinates of the Sun/Moon
        
        sun_altaz = sun.transform_to(Altaz_obs)
        sun_alt = sun_altaz.alt.deg
        sun_az = sun_altaz.az.deg
        obs_az = coord_pointing.transform_to(Altaz_obs).az.deg
        theta = (sun_az-obs_az)
        if theta<0:
            theta += 360
        sep_sun_obs_az = np.min(((360-theta),theta))#https://arxiv.org/pdf/1407.8283 
        G_Sun = -2.5/np.log(10)*10**(-0.005555*sep_sun_obs_az-1)/3600     # mag per arcsec
        M_sun = np.min([30,np.max([1,8-1.03*sun_alt])])#https://arxiv.org/pdf/1407.8283
        racomp, deccomp = get_direction(coord_pointing.ra.deg,coord_pointing.dec.deg,sun.ra.deg,sun.dec.deg)
        dSun_dRA = G_Sun*racomp
        dSun_dDEC = G_Sun*deccomp
        # print(f'sep to moon = {rho} deg; \nphase of Moon = {alpha}\nZenith distance of moon = {Zm}\nAzi of moon = {moon.transform_to(Altaz_obs).az.deg}\nAlt of moon = {moon.transform_to(Altaz_obs).alt.deg}')
        # print(f'pointing alt = {90-Z}; pointing az = {obs_az}')
        # print(f"Telescope RA: {coord_pointing.ra.deg}, DEC: {coord_pointing.dec.deg}")
        # print(f"Moon RA: {moon.ra.deg}, DEC: {moon.dec.deg}")
        print(M_sun,sun_alt)
        # print(dMoon_dRA,dMoon_dDEC,dSun_dRA,dSun_dDEC)
        ny, nx = img.shape

        # Generate pixel grid
        x = np.arange(nx)
        y = np.arange(ny)
        xx, yy = np.meshgrid(x, y)
        arcsec_pixel_1 = self.world.telescope.arcsec_pixel_1
        M_sun_field = M_sun + dSun_dRA*(xx-nx//2)*arcsec_pixel_1 + dSun_dDEC*(yy-ny//2)*arcsec_pixel_1
        M_moon_field = M_moon + dMoon_dRA*(xx-nx//2)*arcsec_pixel_1 + dMoon_dDEC*(yy-ny//2)*arcsec_pixel_1
        M_raw_sky = 22.5
        M_all = -2.5*np.log10(10**(-0.4*M_raw_sky)+10**(-0.4*M_sun_field)+10**(-0.4*M_moon_field))
        flux_c = 870 * 10**(-0.4*(M_all+29))
        hnu = 4.82*6.62607015*10**(-20)
        c =  (0.5)**2 * np.pi * self.world.telescope.camera.exposure_s/ hnu *flux_c # number of photons
        if False:
            import matplotlib.pyplot as plt
            plt.imshow(M_all)
            plt.colorbar()
            plt.show()
        return c
    def star(self,img,gaia_catalog = 'online', Gmag_limit = 18,alpha = 5):
        if gaia_catalog == 'online':
            # Get the gaia source that is not galaxy
            sql = f'''
            SELECT g3.source_id,g3.ra,g3.dec,g3.phot_g_mean_mag,g3.phot_g_mean_flux_over_error, g3.parallax,g3.pmra,g3.pmdec from gaiadr3.gaia_source as g3 LEFT JOIN 	
gaiadr3.galaxy_candidates as ggc ON ggc.source_id = g3.source_id
WHERE g3.phot_g_mean_mag<{Gmag_limit} AND
CONTAINS(
    POINT('ICRS',g3.ra,g3.dec),
    CIRCLE('ICRS',{self.world.telescope.mount.ra_deg},{self.world.telescope.mount.dec_deg},{self.world.telescope.fov_diag/2})
)=1 AND ggc.source_id IS NULL AND g3.phot_g_mean_mag=g3.phot_g_mean_mag;'''      
            job = Gaia.launch_job_async(sql)
            r = job.get_results()
        # Draw the image
        
        
        hnu = 4.82*6.62607015e-20
        FWHM = 1 / self.world.telescope.seeing_arcsec 
        gamma = FWHM / (2 * np.sqrt(2**(1/alpha) - 1))
        flux = 870 * 10**(-0.4*(r['phot_g_mean_mag']+29))
        normalize_factor =  (self.world.telescope.diameter_m/2)**2 * self.world.telescope.camera.exposure_s  * (alpha - 1)/ ( gamma**2 * hnu)
        n_photon =  normalize_factor * flux
        image_star = np.zeros(img.shape)
        radec_star = np.array([[r['ra'][i],r['dec'][i]] for i in range(len(r))])
        pixcrd = self.world.telescope.wcs.wcs_world2pix(radec_star, 0)
        print('Brightest star gmag', np.min(r['phot_g_mean_mag']))
        print('at',pixcrd[np.argmin(r['phot_g_mean_mag'])])
        x0 = [item[0] for item in pixcrd]
        y0 = [item[1] for item in pixcrd]
        for x0, y0, A in zip(x0, y0, n_photon):
            x_min = max(int(x0 - 150), 0)
            x_max = min(int(x0 + 150), img.shape[0])
            y_min = max(int(y0 - 150), 0)
            y_max = min(int(y0 + 150), img.shape[1])
            if x_min >= x_max or y_min >= y_max:
                continue  
            X_sub, Y_sub = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
            g = Moffat2D(amplitude=A, x_0=x0, y_0=y0, gamma=gamma, alpha=alpha)
            image_star[y_min:y_max, x_min:x_max] += g(X_sub, Y_sub)
        return image_star
    def generate_img(self):
        img = np.zeros((self.world.telescope.camera.pixel_number_x,self.world.telescope.camera.pixel_number_y))
        print(img.shape)
        dark = self.dark_current(img,self.world.telescope.camera.dark_current_e_s_1,self.world.telescope.camera.exposure_s)
        read_noise = self.read_noise(img,self.world.telescope.camera.read_noise_e)
        bias = self.bias(img,self.world.telescope.camera.bias_level)
        flat = self.flat(img)
        sky = self.sky(img)
        star = self.star(img)
        galaxy = 0
        other_events = 0
        print('exposure',self.world.telescope.camera.exposure_s)
        total_photoelectron = (sky + star + galaxy + other_events)*flat*self.world.telescope.camera.QE
        total_photoelectron_variation = np.sqrt(total_photoelectron)*np.random.randn(*img.shape)
        total_electron = (dark + read_noise + total_photoelectron + total_photoelectron_variation) 
        total_electron = np.minimum(self.world.telescope.camera.full_well_capacity_ke*1000,total_electron)
        img =  total_electron / self.world.telescope.camera.gain + bias
        img = np.minimum(2**self.world.telescope.camera.bit_per_pixel-1, img)


        if True:
            import matplotlib.pyplot as plt
            m = np.mean(img)
            s = np.std(img)
            plt.imshow(img,vmin = m-0.7*s,vmax = m+0.7*s,cmap='gray',origin='lower')
            plt.colorbar()
            plt.show()

        print('mean',np.mean(img))
        return img

if __name__ == '__main__':
    camera_par = {'pixel_size_mum':10,'pixel_number_x':2000,'pixel_number_y':2000,'readout_time_s':1/2.9,'gain':0.44,'dark_current_e_s_1':0.05,'full_well_capacity_ke':119,'read_noise_e':1.6,'QE':0.7,'bit_per_pixel':18,'bias_level':1000}
    mount_par = {'tracking_mode':'alt-az','tracking_speed_deg_s_1':10,'stable_time_s':5,'goto_error_arcsec':10,'tracking_error_arcsec_min_1':0.1}
    telescope_par = {'latlonalt':(38.6068,93.8961,4000),'seeing_arcsec':1,'focal_length_m':1.57,'diameter_m':1}
    cam = camera(camera_par)
    mnt = mount(mount_par)
    tel = telescope(mnt,cam,telescope_par)
    wd = world(tel,2460741.45,input_schedule ='/Users/ruiyicheng/Documents/code/projects/TianYu/Tianyu_pipeline/pipeline/hardware/sim_events/schedule.csv')
    # print(wd.telescope.camera.status)
    # wd.telescope.camera.capture(1)
    # print(wd.telescope.camera.status)
    # wd._time = wd._time+1
    # print(wd.telescope.camera.status)
    # wd.telescope.mount.goto(10,10)

    wd.run_sim()
