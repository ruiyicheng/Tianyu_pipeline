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
from scipy import special
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm


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
            # print('Image is loaded')
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
                self.base_ra_deg = ra_deg
                self.base_dec_deg = dec_deg

                



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
    def Altaz_obs(self):
        return AltAz(obstime=self.world.time_astropy, location=self.position)
    @property
    def pointing(self):
        return SkyCoord(ra=self.mount.ra_deg*u.deg, dec=self.mount.dec_deg*u.deg, obstime=self.world.time_astropy, frame='gcrs', location=self.world.telescope.position)
    
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
    def __init__(self,telescope,sim,output,t0,name = 'simulator_world',input_schedule = None,input_event = None):
        self.name = name
        self.telescope = telescope
        self.telescope.build(self)
        self.output = output
        if 'simulator' in self.name:
            self.img_simulator = sim
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
    @property
    def time_astropy(self):
        return Time(self.time,scale = 'utc',format = 'jd',location = self.telescope.position)
    
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
        ct_name = {}

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
                ct_name[r['target_name']] = 0
                # number of img is the number of the same target_name after the last last_of_this_obs = 1
                n_img = i+1
                while self.time < r['jd_utc_end'] and (r['n_max_frames']<=0 or ct_name[r['target_name']]< r['n_max_frames']):
                    ct_name[r['target_name']] += 1
                    target_dir = Path(self.output) / r['target_name']
                    target_dir.mkdir(parents=True, exist_ok=True)
                    self.target_dir = target_dir
                    self.schedule_target_name = r['target_name']
                    out = target_dir
                    self.telescope.camera.capture(r['exposure_s'])
                    self._time = wait_to_ok(self.telescope.camera)
                    image = self.telescope.camera.download(self.img_simulator.generate_img(is_bias= 'bias' in r['target_name']))
                    header = fits.Header()
                    header['JD'] = self.time
                    print('saving img')
                    fits.writeto(out / (r['target_name'] + f'-image-{n_img}-' + str(ct_name[r['target_name']]) + '.fits'), image.astype('int32'), overwrite=True, header=header)
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
    gaia_dict = {}
    vsx_dict = {}
    def __init__(self, star_config):
        self.initpar = star_config
        for key, val in star_config.items():
            setattr(self, key, val) 
        self.previous_exposure_s = -1
    def connect(self, world):
        self.world = world
    cache_dir = "/mnt/hgfs/imggen/cache"
    @classmethod
    def update_gaia_data_dict(cls, target_key, data):
        cls.gaia_dict[target_key] = data

    @classmethod
    def update_vsx_data_dict(cls, target_key, star, id, parameters):
        cls.vsx_dict['target_key'] = target_key
        cls.vsx_dict['star'] = star
        cls.vsx_dict['id'] = id
        cls.vsx_dict['parameters'] = parameters
   

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
    def bias(self,image, value = 100):
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
        
        
        realistic=self.bias_realistic
        # If we want a more realistic bias we need to do a little more work. 
        if realistic:
            bias_im = self.real_bias.copy()
        else:
            bias_im = np.zeros_like(image) + value
        return bias_im
    @property
    def real_bias(self):
        """
        Return whether the dark current should include hot pixels.
        """
        if not hasattr(self, '_real_bias'):
            # If the user did not set this, we assume they want hot pixels.
            self._real_bias = fits.getdata(self.measured_bias, ext=0).astype(np.float64)
        return self._real_bias 
    @property
    def real_dark_hot_pixels(self):
        """
        Return whether the dark current should include hot pixels.
        """
        if not hasattr(self, '_dark_hot_pixels_1s'):
            # If the user did not set this, we assume they want hot pixels.
            self._dark_hot_pixels_1s = fits.getdata(self.measured_dark, ext=0).astype(np.float64)
            self._dark_hot_pixels_1s = (self._dark_hot_pixels_1s - self.real_bias)/60
        if self.world.telescope.camera.exposure_s != self.previous_exposure_s:
            self.previous_exposure_s = self.world.telescope.camera.exposure_s
            self._dark_hot_pixels = self._dark_hot_pixels_1s * self.world.telescope.camera.exposure_s/self.world.telescope.camera.gain


        return self._dark_hot_pixels
    def dark_current(self,image, current, exposure_time, mode = 'Fake'):
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
        if mode == 'Fake':
            hot_pixels = self.dark_hot_pixels

            # dark current for every pixel; we'll modify the current for some pixels if 
            # the user wants hot pixels.
            base_current = current * exposure_time 
            
            # This random number generation should change on each call.
            dark_im = np.random.poisson(base_current, size=image.shape)
                
            if hot_pixels:
                # We'll set 0.01% of the pixels to be hot; that is probably too high but should 
                # ensure they are visible.
                y_max, x_max = image.shape
                
                n_hot = y_max*x_max* 0.0001  # 0.01% of the pixels
                
                # Like with the bias image, we want the hot pixels to always be in the same places
                # (at least for the same image size) but also want them to appear to be randomly
                # distributed. So we set a random number seed to ensure we always get the same thing.
                
                hot_x = np.random.randint(1, x_max, size=n_hot)
                hot_y = np.random.randint(1, y_max, size=n_hot)
                
                hot_current = 10000 * current
                
                dark_im[(hot_y, hot_x)] = hot_current * exposure_time
        else:
            dark_im = self.real_dark_hot_pixels.copy()
        return dark_im
    def flat(self,image):
        path_template = self.flat_catalog
        range_in_template = self.range
        flat_save = fits.getdata(path_template)
        flat_re = flat_save[range_in_template[0]:range_in_template[1], range_in_template[2]:range_in_template[3]]
        flat_co = cv2.resize(flat_re.astype(np.float64), (image.shape[0], image.shape[1]), interpolation=cv2.INTER_LINEAR) 
        flat_co /= np.mean(flat_co)
        return flat_co
    def sky(self,img,bias = False):# to simplify the problem, we set the image to align with dec axis
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
        
        if bias: 
            return 0
        # get moon phase
        sun = get_body('sun',self.world.time_astropy)#.transform_to(Altaz_obs)
        moon = get_body('moon',self.world.time_astropy)#.transform_to(Altaz_obs)
        # Compute elongation (angular separation between Sun and Moon)
        
        elongation = sun.transform_to(self.world.telescope.Altaz_obs).separation(moon.transform_to(self.world.telescope.Altaz_obs)).deg  
        alpha = 180 - elongation
        #coord_pointing = SkyCoord(ra=self.world.telescope.mount.ra_deg*u.deg, dec=self.world.telescope.mount.dec_deg*u.deg, obstime=self., frame='gcrs', location=self.world.telescope.position)#
        rho = self.world.telescope.pointing.separation(moon).deg
        Zm = 90-moon.transform_to(self.world.telescope.Altaz_obs).alt.deg
        Z = 90-self.world.telescope.pointing.transform_to(self.world.telescope.Altaz_obs).alt.deg
        M_moon = moon_sky(rho,Z,Zm,alpha)
        coord_pointing_ra_deviated = SkyCoord(ra=(self.world.telescope.mount.ra_deg+1/3600)*u.deg, dec=self.world.telescope.mount.dec_deg*u.deg, obstime=self.world.time_astropy, frame='gcrs', location=self.world.telescope.position)
        coord_pointing_dec_deviated = SkyCoord(ra=self.world.telescope.mount.ra_deg*u.deg, dec=(self.world.telescope.mount.dec_deg+1/3600)*u.deg, obstime =self.world.time_astropy, frame='gcrs', location=self.world.telescope.position)
        rho_ra_deviated = coord_pointing_ra_deviated.separation(moon).deg
        rho_dec_deviated = coord_pointing_dec_deviated.separation(moon).deg
        Z_ra_deviated = 90-coord_pointing_ra_deviated.transform_to(self.world.telescope.Altaz_obs).alt.deg
        Z_dec_deviated = 90-coord_pointing_dec_deviated.transform_to(self.world.telescope.Altaz_obs).alt.deg
        M_moon_ra_deviated = moon_sky(rho_ra_deviated,Z_ra_deviated,Zm,alpha)    
        M_moon_dec_deviated = moon_sky(rho_dec_deviated,Z_dec_deviated,Zm,alpha)
        dMoon_dRA = (M_moon_ra_deviated-M_moon) #mag per arcsec
        dMoon_dDEC = (M_moon_dec_deviated-M_moon) #mag per arcsec

        # Compute the AltAz coordinates of the Sun/Moon
        
        sun_altaz = sun.transform_to(self.world.telescope.Altaz_obs)
        sun_alt = sun_altaz.alt.deg
        sun_az = sun_altaz.az.deg
        obs_az = self.world.telescope.pointing.transform_to(self.world.telescope.Altaz_obs).az.deg
        theta = (sun_az-obs_az)
        if theta<0:
            theta += 360
        sep_sun_obs_az = np.min(((360-theta),theta))#https://arxiv.org/pdf/1407.8283 
        G_Sun = -2.5/np.log(10)*10**(-0.005555*sep_sun_obs_az-1)/3600     # mag per arcsec
        M_sun = np.min([30,np.max([1,8-1.03*sun_alt])])#https://arxiv.org/pdf/1407.8283
        racomp, deccomp = get_direction(self.world.telescope.pointing.ra.deg,self.world.telescope.pointing.dec.deg,sun.ra.deg,sun.dec.deg)
        dSun_dRA = G_Sun*racomp
        dSun_dDEC = G_Sun*deccomp
        # print(f'sep to moon = {rho} deg; \nphase of Moon = {alpha}\nZenith distance of moon = {Zm}\nAzi of moon = {moon.transform_to(Altaz_obs).az.deg}\nAlt of moon = {moon.transform_to(Altaz_obs).alt.deg}')
        # print(f'pointing alt = {90-Z}; pointing az = {obs_az}')
        # print(f"Telescope RA: {coord_pointing.ra.deg}, DEC: {coord_pointing.dec.deg}")
        # print(f"Moon RA: {moon.ra.deg}, DEC: {moon.dec.deg}")
        # print(M_sun,sun_alt)
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
    
    def star(self, img, bias):
        gaia_catalog = self.gaia_catalog
        Gmag_limit = self.Gmag_limit
        alpha = np.random.normal(loc=self.alpha, scale=0.1*self.alpha)
        if alpha < 2 or alpha > 4.5:
            alpha = 3
        transit_catalog = self.transit_catalog
        supernova_erupt_catalog = self.supernova_erupt_catalog
        binary_catalog = self.binary_catalog
        flare_catalog = self.flare_catalog
        satellite_catalog = self.satellite_catalog
        gal_catalog = self.gal_catalog
        supernova_erupt = self.supernova_erupt
        satellite_flag = self.satellite_flag
        

        if bias:
            return 0
        
        target_key = (self.world.telescope.mount.base_ra_deg, self.world.telescope.mount.base_dec_deg,Gmag_limit,self.world.telescope.fov_diag/2)
        print('sky_region:', target_key)
        target_path = Path(self.cache_dir) / f'gaia_{target_key[0]}_{target_key[1]}_{target_key[2]}_{target_key[3]}.fits'
        if target_key in self.gaia_dict:
            r = self.gaia_dict[target_key]
            print('using cached gaia data in memory')
        else:
            if target_path.exists():
                r = fits.getdata(target_path)
                print('using cached gaia data in disk')
            elif gaia_catalog == 'online':
                # Get the gaia source that is not galaxy
                print('getting gaia data from online')
                sql = f'''
                SELECT g3.source_id as source_id,g3.ra,g3.dec,g3.phot_g_mean_mag,g3.phot_g_mean_flux_error,g3.phot_g_n_obs,g3.phot_g_mean_flux, g3.phot_variable_flag,g3.parallax,g3.pmra,g3.pmdec from gaiadr3.gaia_source as g3 LEFT JOIN 	
    gaiadr3.galaxy_candidates as ggc ON ggc.source_id = g3.source_id
    WHERE g3.phot_g_mean_mag<{Gmag_limit} AND
    CONTAINS(
        POINT('ICRS',g3.ra,g3.dec),
        CIRCLE('ICRS',{self.world.telescope.mount.base_ra_deg},{self.world.telescope.mount.base_dec_deg},{self.world.telescope.fov_diag/2+0.1})
    )=1 AND g3.phot_g_mean_mag=g3.phot_g_mean_mag;'''     
                print('getting gaia data from online') 
                job = Gaia.launch_job_async(sql)
                r = job.get_results()
                # Save the data to cache
                if 'phot_variable_flag' in r.colnames:
                    max_len = max(len(x) for x in r['phot_variable_flag'])
                    r['phot_variable_flag'] = r['phot_variable_flag'].astype(f'U{max_len}')

                # Save the fixed data
                print('saving gaia data to cache file')
                r.write(target_path, format='fits', overwrite=True)

        mag_raw = r['phot_g_mean_mag']


        print('generating scintillation')
        # Get the magnitude error
        total_electron_gaia = 4.42 * r['phot_g_n_obs'] * r['phot_g_mean_flux']
        error_photometric = np.sqrt(total_electron_gaia)/r['phot_g_n_obs']
        error_total = r['phot_g_mean_flux_error']
        error_jitter = np.sqrt(np.maximum(0,error_total**2-error_photometric**2))
        error_fraction_jitter = error_jitter/r['phot_g_mean_flux'] # * (~(r['phot_variable_flag']=='VARIABLE')) # Variable star have larger error, process correspondingly 
        error_fraction_jitter_sample = 1+error_fraction_jitter * np.random.randn(len(r))


        CY=1.5
        z = np.pi/2-self.world.telescope.pointing.transform_to(self.world.telescope.Altaz_obs).alt.rad
        airmass = (1.002432*np.cos(z)**2+0.148386*np.cos(z)+0.0096467)/(np.cos(z)**3+0.149864*np.cos(z)**2+0.0102963*np.cos(z)+0.000303978) #https://opg.optica.org/ao/abstract.cfm?uri=ao-33-6-1108
        print("Airmass:", airmass)
        if airmass<0.1:
            airmass = 0.1
        scintillation = np.sqrt(10**(-5)*CY**2*np.array(self.world.telescope.diameter_m)**(-4/3)/(self.world.telescope.camera.exposure_s)*(airmass)**3*np.exp(-2*self.world.telescope.position.height.to(u.m)/8000/u.m))
        print('scintillation=',scintillation)
        error_scintillation = 1+scintillation * np.random.randn(len(r))
        try:
            star_all = SkyCoord(ra=r['ra']*u.deg, dec=r['dec']*u.deg, obstime=self.world.time_astropy, frame='icrs', location=self.world.telescope.position)
        except:
            star_all = SkyCoord(ra=r['ra'], dec=r['dec'], obstime=self.world.time_astropy, frame='icrs', location=self.world.telescope.position)
        z_all = np.pi/2-star_all.transform_to(self.world.telescope.Altaz_obs).alt.rad
        airmass_all = (1.002432*np.cos(z_all)**2+0.148386*np.cos(z_all)+0.0096467)/(np.cos(z_all)**3+0.149864*np.cos(z_all)**2+0.0102963*np.cos(z_all)+0.000303978) #https://opg.optica.org/ao/abstract.cfm?uri=ao-33-6-1108
        K = 0.23
        luminosity_fraction_extinction = 10**(-0.4*(K*airmass_all))

        delta_mag_var = 0
        mag = mag_raw + delta_mag_var
        
        t_tdb_bjd = self.world.time_astropy.tdb.jd
        # Consider the events of transits
        
        transit_relative_flux = np.ones(mag_raw.shape)
        print('done')
        if type(transit_catalog)==str:
            print('calculating transit')
            transit_events = pd.read_csv(transit_catalog)
            #sprint(transit_events)

            for index,row in transit_events.iterrows():
                if not row['dr3_source_id'] in r['source_id']:
                    continue
                
                target_id = row['dr3_source_id']
                target = r[r['source_id']==row['dr3_source_id']]
                
                target_pos = SkyCoord(ra=target['ra'][0]*u.deg, dec=target['dec'][0]*u.deg, frame='gcrs',location=self.world.telescope.position)
                ltt_bary = self.world.time_astropy.light_travel_time(target_pos)
                
                time_barycentre = t_tdb_bjd - ltt_bary  
                # print(time_barycentre,time_barycentre)

                # Use batman to generate the light curve
                params = batman.TransitParams()
                params.t0 = row['tm_tdb_bjd']                        #time of inferior conjunction
                params.per = row['period_d']                       #orbital period
                params.rp = row['radius_star_radius']                       #planet radius (in units of stellar radii)
                params.a = row['semi_major_axis_stellar_radius']                       #semi-major axis (in units of stellar radii)
                params.inc = row['inclination_deg']                     #orbital inclination (in degrees)
                params.ecc = row['e']                      #eccentricity
                params.w = 90.                        #longitude of periastron (in degrees)
                params.limb_dark = "nonlinear"        #limb darkening model
                params.u = [row['u05'], row['u1'], row['u15'], row['u2']]
                m = batman.TransitModel(params, np.array([time_barycentre.jd]))
                flux = m.light_curve(params)
                transit_relative_flux[np.where(r['source_id']==row['dr3_source_id'])] = flux



        # Consider binary
        binary_relative_flux = np.ones(mag_raw.shape)
        if type(binary_catalog)==str:
            print('calculating binary')
            binary_events = pd.read_csv(binary_catalog)
            #sprint(transit_events)

            for index,row in binary_events.iterrows():
                if not row['dr3_source_id'] in r['source_id']:
                    continue

                target_id = row['dr3_source_id']
                target = r[r['source_id']==row['dr3_source_id']]
                target_pos = SkyCoord(ra=target['ra'][0]*u.deg, dec=target['dec'][0]*u.deg, frame='gcrs',location=self.world.telescope.position)
                ltt_bary = self.world.time_astropy.light_travel_time(target_pos)

                time_barycentre = t_tdb_bjd - ltt_bary

                period = row['period']
                phase_shift = row['phase_shift']
                amplitude_primary = row['amplitude_primary']
                amplitude_secondary = row['amplitude_secondary']

                primary = amplitude_primary * np.sin(2 * np.pi * time_barycentre.jd / period + phase_shift)
                secondary = amplitude_secondary * np.cos(4 * np.pi * time_barycentre.jd / period + phase_shift)
                flux = 1 + primary + secondary
                binary_relative_flux[np.where(r['source_id']==row['dr3_source_id'])] = flux


        


        # consider flare
        print('calculating flare')
        flare_relative_flux = np.ones(mag_raw.shape)
        # flare_relative_flux = np.ones(mag_raw.shape)
        
        # the Llamaradas-Estelares include flare_eqn and flare_model
        def flare_eqn(t,tpeak,fwhm,ampl):

            '''
            The equation that defines the shape for the Continuous Flare Model
            '''
            #Values were fit & calculated using MCMC 256 walkers and 30000 steps

            A,B,C,D1,D2,f1 = [0.9687734504375167,-0.251299705922117,0.22675974948468916,
                              0.15551880775110513,1.2150539528490194,0.12695865022878844]

            # We include the corresponding errors for each parameter from the MCMC analysis

            A_err,B_err,C_err,D1_err,D2_err,f1_err = [0.007941622683556804,0.0004073709715788909,0.0006863488251125649,
                                                      0.0013498012884345656,0.00453458098656645,0.001053149344530907 ]

            f2 = 1-f1

            eqn = ((1 / 2) * np.sqrt(np.pi) * A * C * f1 * np.exp(-D1 * t + ((B / C) + (D1 * C / 2)) ** 2)
                                * special.erfc(((B - t) / C) + (C * D1 / 2))) + ((1 / 2) * np.sqrt(np.pi) * A * C * f2
                                * np.exp(-D2 * t+ ((B / C) + (D2 * C / 2)) ** 2) * special.erfc(((B - t) / C) + (C * D2 / 2)))
            return eqn * ampl


        def flare_model(t,tpeak, fwhm, ampl, upsample=False, uptime=10):
            '''
            The Continuous Flare Model evaluated for single-peak (classical) flare events.
            Use this function for fitting classical flares with most curve_fit
            tools. Reference: Tovar Mendoza et al. (2022) DOI 10.3847/1538-3881/ac6fe6

            References
            --------------
            Tovar Mendoza et al. (2022) DOI 10.3847/1538-3881/ac6fe6
            Davenport et al. (2014) http://arxiv.org/abs/1411.3723
            Jackman et al. (2018) https://arxiv.org/abs/1804.03377

            Parameters
            ----------
            t : 1-d array
                The time array to evaluate the flare over

            tpeak : float
                The center time of the flare peak

            fwhm : float
                The Full Width at Half Maximum, timescale of the flare

            ampl : float
                The amplitude of the flare


            Returns
            -------
            flare : 1-d array
                The flux of the flare model evaluated at each time

                A continuous flare template whose shape is defined by the convolution of a Gaussian and double exponential
                and can be parameterized by three parameters: center time (tpeak), FWHM, and ampitude
            '''

            t_new = (t-tpeak)/fwhm

            if upsample:
                dt = np.nanmedian(np.diff(np.abs(t_new)))
                timeup = np.linspace(min(t_new) - dt, max(t_new) + dt, t_new.size * uptime)

                flareup = flare_eqn(timeup,tpeak,fwhm,ampl)

                # and now downsample back to the original time...

                downbins = np.concatenate((t_new - dt / 2.,[max(t_new) + dt / 2.]))
                flare,_,_ = binned_statistic(timeup, flareup, statistic='mean',bins=np.sort(downbins))
            else:
            
                flare = flare_eqn(t_new,tpeak,fwhm,ampl)

            return flare

        
        
        if type(flare_catalog)==str:
            flare_events = pd.read_csv(flare_catalog)

            for index,row in flare_events.iterrows():
                if not row['dr3_source_id'] in r['source_id']:
                    continue
                target_id = row['dr3_source_id']
                target = r[r['source_id']==row['dr3_source_id']]
                
                target_pos = SkyCoord(ra=target['ra'][0]*u.deg, dec=target['dec'][0]*u.deg, frame='gcrs',location=self.world.telescope.position)
                ltt_bary = self.world.time_astropy.light_travel_time(target_pos)
                
                time_barycentre = t_tdb_bjd - ltt_bary  

                # Use Llamaradas-Estelares to generate the light curve
                t_peak = row['tpeak']
                fwhm = row['fwhm']
                amplitude = row['amp']
                parameter = [t_peak, fwhm, amplitude]
                flux = flare_model(time_barycentre.jd, *parameter, upsample=False) + 1
                flare_relative_flux[np.where(r['source_id']==row['dr3_source_id'])] = flux

        flux_prod = transit_relative_flux * flare_relative_flux * binary_relative_flux * error_scintillation * error_fraction_jitter_sample * luminosity_fraction_extinction
        print('generate PSF img')
        hnu = 4.82*6.62607015e-20
        FWHM = self.world.telescope.seeing_arcsec / self.world.telescope.arcsec_pixel_1
        gamma = FWHM / (2 * np.sqrt(2**(1/alpha) - 1))
        flux = 870 * 10**(-0.4*(mag+29))

        normalize_factor =  (self.world.telescope.diameter_m/2)**2 * self.world.telescope.camera.exposure_s  * (alpha - 1)/ ( gamma**2 * hnu)
        n_photon =  normalize_factor * flux * flux_prod
        image_star = np.zeros(img.shape)
        radec_star = np.array([[r['ra'][i],r['dec'][i]] for i in range(len(r))])
        pixcrd = self.world.telescope.wcs.wcs_world2pix(radec_star, 0)

        x0 = [item[0] for item in pixcrd]
        y0 = [item[1] for item in pixcrd]

        ny, nx = img.shape
        Y_full, X_full = np.mgrid[0:ny, 0:nx]
        width_box = 16
        for x0, y0, A in tqdm.tqdm(zip(x0, y0, n_photon)):
            x_min = max(int(x0 - width_box), 0)
            x_max = min(int(x0 + width_box), img.shape[0])
            y_min = max(int(y0 - width_box), 0)
            y_max = min(int(y0 + width_box), img.shape[1])
            if x_min >= x_max or y_min >= y_max:
                continue
            X_sub = X_full[y_min:y_max, x_min:x_max]
            Y_sub = Y_full[y_min:y_max, x_min:x_max]  
            # X_sub, Y_sub = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
            g = Moffat2D(amplitude=A, x_0=x0, y_0=y0, gamma=gamma, alpha=alpha)
            image_star[y_min:y_max, x_min:x_max] += g(X_sub, Y_sub)
        print('done')
    

        # # galaxy 
        # image_gal = np.zeros(img.shape)
        # gal_data = pd.read_csv(gal_catalog)
        # ra_min = self.world.telescope.mount.ra_deg - self.world.telescope.fov_diag/2/np.cos(self.world.telescope.mount.dec_deg*np.pi/180)
        # ra_max = self.world.telescope.mount.ra_deg + self.world.telescope.fov_diag/2/np.cos(self.world.telescope.mount.dec_deg*np.pi/180)
        # dec_min = self.world.telescope.mount.dec_deg - self.world.telescope.fov_diag/2
        # dec_max = self.world.telescope.mount.dec_deg + self.world.telescope.fov_diag/2
        # select_data = gal_data[
        #     (gal_data['ra'] >= ra_min) & (gal_data['ra'] <= ra_max) & 
        #     (gal_data['dec'] >= dec_min) & (gal_data['dec'] <= dec_max) & 
        #     (gal_data['phot_g_mean_mag'] <= 22) &
        #     gal_data['posangle_sersic'].notna() &
        #     gal_data['radius_sersic'].notna() &
        #     gal_data['ellipticity_sersic'].notna() &
        #     gal_data['n_sersic'].notna() &
        #     gal_data['phot_g_mean_mag'].notna() 
        # ]
        # print("select galaxy:", select_data)
        # if not select_data.empty:
        #     gal_id = np.array(select_data['source_id'])
        #     gal_ra = np.array(select_data['ra'])       
        #     gal_dec = np.array(select_data['dec'])        
        #     gal_posangle = np.array(select_data['posangle_sersic'])
        #     gal_e = np.array(select_data['ellipticity_sersic'])
        #     gal_q = 1 - gal_e 
        #     gal_n = np.array(select_data['n_sersic'])
        #     gal_re_mas = np.array(select_data['radius_sersic'])
        #     gal_re = gal_re_mas / 1000 
        #     gal_mag = np.array(select_data['phot_g_mean_mag'])
        #     gal_flux_g = 870 * 10**(-0.4*(gal_mag+29))

        #     gal_cord = np.column_stack((gal_ra, gal_dec))
        #     print("galaxy's ra/dec:", gal_cord)
        #     gal_pix = self.world.telescope.wcs.wcs_world2pix(gal_cord, 0)
        #     print("galaxy's pix:", gal_pix)

        #     normalize_factor_gal =  (self.world.telescope.diameter_m/2)**2 * self.world.telescope.camera.exposure_s  * (alpha - 1)/ ( gamma**2 * hnu)

        #     # Get the magnitude error
        #     total_electron_gaia_gal = 4.42 * select_data['phot_g_n_obs'] * select_data['phot_g_mean_flux']
        #     error_photometric_gal = np.sqrt(total_electron_gaia_gal)/select_data['phot_g_n_obs']
        #     error_total_gal = select_data['phot_g_mean_flux_error']
        #     error_jitter_gal = np.sqrt(np.maximum(0,error_total_gal**2-error_photometric_gal**2))
        #     error_fraction_jitter_gal = error_jitter_gal/select_data['phot_g_mean_flux'] * (~(select_data['phot_variable_flag']=='VARIABLE')) # Variable star have larger error, process correspondingly 
        #     error_fraction_jitter_sampl_gal = 1+error_fraction_jitter_gal * np.random.randn(len(select_data))

        #     error_scintillation_gal = 1+scintillation * np.random.randn(len(select_data))

        #     gal_all = SkyCoord(ra=gal_ra * u.deg, dec=gal_dec * u.deg, obstime=self.world.time_astropy, frame='icrs', location=self.world.telescope.position)
        #     gal_z_all = np.pi/2-gal_all.transform_to(self.world.telescope.Altaz_obs).alt.rad
        #     gal_airmass_all = (1.002432*np.cos(gal_z_all)**2+0.148386*np.cos(gal_z_all)+0.0096467)/(np.cos(gal_z_all)**3+0.149864*np.cos(gal_z_all)**2+0.0102963*np.cos(gal_z_all)+0.000303978) #https://opg.optica.org/ao/abstract.cfm?uri=ao-33-6-1108
        #     K = 0.23
        #     luminosity_fraction_extinction_gal = 10**(-0.4*(K*gal_airmass_all))

        #     gal_flux = np.array(gal_flux_g, dtype=np.float64)
        #     gal_ADU = normalize_factor_gal * gal_flux * error_fraction_jitter_sampl_gal * error_scintillation_gal * luminosity_fraction_extinction_gal

        #     x0 = [item[0] for item in gal_pix]
        #     y0 = [item[1] for item in gal_pix]
        #     for x0, y0, A in zip(x0, y0, gal_ADU):
        #         x_min = max(int(x0 - 150), 0)
        #         x_max = min(int(x0 + 150), img.shape[0])
        #         y_min = max(int(y0 - 150), 0)
        #         y_max = min(int(y0 + 150), img.shape[1])
        #         if x_min >= x_max or y_min >= y_max:
        #             continue  
        #         X_sub, Y_sub = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        #         g = Moffat2D(amplitude=A, x_0=x0, y_0=y0, gamma=gamma, alpha=alpha)
        #         image_gal[y_min:y_max, x_min:x_max] += g(X_sub, Y_sub)

                
        image_star_gal = image_star
        return image_star_gal
    
    def generate_img(self,is_bias = False):
        img = np.zeros((self.world.telescope.camera.pixel_number_x,self.world.telescope.camera.pixel_number_y))
        #print(img.shape)
        print('generating dark')
        dark = self.dark_current(img,self.world.telescope.camera.dark_current_e_s_1,self.world.telescope.camera.exposure_s,mode = 'Real')
        print('generating read noise')
        read_noise = self.read_noise(img,self.world.telescope.camera.read_noise_e)
        print('generating bias')
        bias = self.bias(img,self.world.telescope.camera.bias_level)
        print('generating flat')
        flat = self.flat(img)
        print('generating sky')
        sky = self.sky(img,bias = is_bias)
        print('generating star')
        star = self.star(img, bias = is_bias)
        galaxy = 0
        other_events = 0
        print('exposure',self.world.telescope.camera.exposure_s)
        print('generating noise')
        total_photoelectron = (sky + star + galaxy + other_events)*flat*self.world.telescope.camera.QE
        total_photoelectron_variation = np.sqrt(np.abs(total_photoelectron))*np.random.randn(*img.shape)
        total_electron = (dark + read_noise + total_photoelectron + total_photoelectron_variation) 
        total_electron = np.minimum(self.world.telescope.camera.full_well_capacity_ke*1000,total_electron)
        img =  total_electron / self.world.telescope.camera.gain + bias
        img = np.minimum(2**self.world.telescope.camera.bit_per_pixel-1, img)


        if False:
            import matplotlib.pyplot as plt
            m = np.mean(img)
            s = np.std(img)
            plt.imshow(img,vmin = m-0.7*s,vmax = m+0.7*s,cmap='gray',origin='lower')
            plt.colorbar()
            plt.show()

        #print('mean',np.mean(img))
        return img

if __name__ == '__main__':

    camera_par = {'pixel_size_mum':10,'pixel_number_x':8120,'pixel_number_y':8120,'readout_time_s':1/2.9,'gain':1.44,'dark_current_e_s_1':0.05,'full_well_capacity_ke':119,'read_noise_e':1.6,'QE':0.7,'bit_per_pixel':18,'bias_level':1000}
    mount_par = {'tracking_mode':'alt-az','tracking_speed_deg_s_1':5,'stable_time_s':5,'goto_error_arcsec':0.2,'tracking_error_arcsec_min_1':0.1}
    telescope_par = {'latlonalt':(38.6068,93.8961,4000),'seeing_arcsec':2.1,'focal_length_m':1.57,'diameter_m':1}
    star_config = {# supernova_erupt mean using rejection sampling to generate supernova on the FOV galaxies, the number depends on the rows of the supernova_erupt_catalog
                    # satellite_flag mean whether to consider the satellite of the satellite_catalog, satellite_catalog is time-sensitive and needs to often downloaded and update manually, or each time via a URL
                    'gaia_catalog':'online','Gmag_limit':25,'alpha':3, 'supernova_erupt':True,'satellite_flag':False, 
                    'flat_catalog':'/mnt/hgfs/imggen/code/Tianyu_simulator/photometric_iamge_simulator/fig/flat_co_bias.fit', 'range':(500,6000, 1500,7000),'dark_hot_pixels':False, 'bias_realistic':True,
                    'transit_catalog':'/mnt/hgfs/imggen/code/Tianyu_simulator/photometric_iamge_simulator/sim_events/transit.csv',
                    'supernova_erupt_catalog':'/mnt/hgfs/imggen/code/Tianyu_simulator/photometric_iamge_simulator/sim_events/supernova_erupt.csv',
                    'binary_catalog':'/mnt/hgfs/imggen/code/Tianyu_simulator/photometric_iamge_simulator/sim_events/binary.csv',
                    'flare_catalog':'/mnt/hgfs/imggen/code/Tianyu_simulator/photometric_iamge_simulator/sim_events/flare.csv',
                    'satellite_catalog':'/mnt/hgfs/imggen/code/Tianyu_simulator/photometric_iamge_simulator/satellite.csv',
                    'gal_catalog':'/mnt/hgfs/imggen/code/Tianyu_simulator/photometric_iamge_simulator/gaia/gaiadr3/galaxy_gaia.csv',
                    'measured_bias':'/mnt/hgfs/imggen/code/Tianyu_simulator/photometric_iamge_simulator/fig/bias.fits',
                    'measured_dark':'/mnt/hgfs/imggen/code/Tianyu_simulator/photometric_iamge_simulator/fig/dark_60.fits'
    }
    cam = camera(camera_par)
    mnt = mount(mount_par)
    tel = telescope(mnt,cam,telescope_par)
    sim = image_simulator(star_config)
    wd = world(tel,sim,'/mnt/hgfs/imggen/result',2460668.3,input_schedule ='/mnt/hgfs/imggen/code/Tianyu_simulator/photometric_iamge_simulator/sim_events/schedule.csv')

    wd.run_sim()




