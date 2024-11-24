# Define the class of telescope hardware
class hardware:
    def __init__(self,hardware_type,name = 'simulator'):
        self.connected = False
        self.name = name
        self.hardware_type = hardware_type
        self.telescope = None
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
                # SET according to the hardware
                pass
        else:
            raise   RuntimeError('Please connect the hardware')

    def get_parameter(self,attribute):
        if hasattr(self,attribute):
            if 'simulator' in self.name:
                return attribute
            else:
                # read according to the hardware
                pass
        else:
            raise   RuntimeError('Please connect the hardware')  

# pixel_size_mum, pixel_number, readout_time_s, gain, dark_current_e_s-1, full_well_capacity_e, read_noise_e     
class camera(hardware):
    def __init__(self,par_camera,name = 'simulator_cam'):
        super().__init__('camera',name = name)
        self.initpar = par_camera
    def capture(self,exposure_time,mode,binning = 1):
        pass

# tracking_mode, tracking_speed_deg_s-1, stable_time_s, goto_error_arcsec, tracking_error_arcsec_min-1
class mount(hardware):
    def __init__(self,par_mount,name = 'simulator_mount'):
        super().__init__('mount',name = name)
        self.status = 'idle'                # status = idle/running/goto
        self.initpar = par_mount
    def goto(self,pos):
        if self.tracking_mode == 'alt-az':
            pass


#parameter of telescope = position, seeing, focus, diameter

class telescope:
    def __init__(self,mount,camera,par_telescope ,name = 'simulator_telescope'):
        self.name = name
        self.mount = mount
        self.camera = camera
        self.mount.connect(self)
        self.camera.connect(self)
    def build(self,world):
        self.world = world
        print("Telecope is built in world",world.name)
    def get_wcs(self):
        pass
    def goto(self,pos):
        pass
    def capture(self,exposure_time,mode,binning = 1):
        pass
    def wait(self,time,unit):
        pass


#parameter of world = t0

class world:
    def __init__(self,telescope, par_world,name = 'simulator_world'):
        self.name = name
        self.telescope = telescope
        self.telescope.build(self)
        self.par_world = par_world
        self.events = []
    def run_sim(self):
        pass



if __name__ == '__main__':
    camera_par = {'pixel_size_mum':10,'pixel_number_x':8120,'pixel_number_y':8120,'readout_time_s':1/2.9,'gain':1,'dark_current_e_s-1':0.05,'full_well_capacity_ke':119,'read_noise_e':1.6,'QE':0.7,'bit_per_pixel':18}
    mount_par = {'tracking_mode':'alt-az','tracking_speed_deg_s-1':1,'stable_time_s':5,'goto_error_arcsec':10,'tracking_error_arcsec_m-1':0.1}
    telescope_par = {'position':(38.6068,93.8961,4000),'seeing_arcsec':1,'focal_length_m':1.57,'diameter_m':1}
    cam = camera(camera_par)
    mnt = mount(mount_par)
    tel = telescope(mnt,cam,telescope_par)
    wd = world(tel,{'t0':0})