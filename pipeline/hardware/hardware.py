# Define the class of telescope hardware
class hardware:
    def __init__(self,hardware_type,name = 'simulator'):
        self.connect = False
        self.name = name
        self.hardware_type = hardware_type
    def connect(self):
        if self.name=='simulator':
            self.connect = True
            
    def set_parameter(self,attribute,value):
        if self.connect:
            if self.name == 'simulator':
                setattr(self,attribute,value)
            else:
                # SET according to the hardware
                pass
        else:
            raise   RuntimeError('Please connect the hardware')
        
class camera(hardware):
    def __init__(self):
        pass
class mount(hardware):
    def __init__(self,mode = 'alt-az'):
        pass

class telescope:
    def __init__(self, name, diameter, focal_length, pixel_size, pixel_scale, fov, plate_scale, filter_set):
        pass