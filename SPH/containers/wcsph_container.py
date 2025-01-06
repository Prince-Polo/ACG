import taichi as ti 
from .base_container import BaseContainer
from ..utils import SimConfig

@ti.data_oriented
class WCSPHContainer(BaseContainer):
    def __init__(self, config: SimConfig, GGUI=False):
        super().__init__(config, GGUI)
        
        self.gamma = 7.0
        self.stiffness = 50000.0