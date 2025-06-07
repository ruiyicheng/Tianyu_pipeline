import os
import sys
sys.path.append(os.path.abspath("/home/test/workspace/Tianyu_pipeline/algorithm/code"))
import utils.dataloader as dataloader
import utils.Bertin as Bertin


class pipeline_component:
    def __init__(self,free = False):
        self.connected = False
        if free:
            self.data_loader = dataloader.dataloader()
            self.Bertin = Bertin.Bertin_tools()

    def connect(self, runner):
        self.runner = runner
        self.data_loader = self.runner.data_loader
        self.Bertin = self.runner.Bertin
        self.connected = True
    def disconnect(self):
        self.connected = False
        self.runner = None
        self.data_loader = None



