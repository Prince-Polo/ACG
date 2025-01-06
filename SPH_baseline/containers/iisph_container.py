import taichi as ti 
from .base_container import BaseContainerBaseline
from ..utils import SimConfig

@ti.data_oriented
class IISPHContainerBaseline(BaseContainerBaseline):
    def __init__(self, config: SimConfig, GGUI=False):
        super().__init__(config, GGUI)
        self.density_error = ti.field(dtype=float, shape=())
        
        self.iisph_a_ii = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.iisph_source = ti.field(dtype=ti.f32, shape=self.particle_max_num)
    
        self.iisph_pressure_accelerations = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.iisph_pressure = ti.field(dtype=ti.f32, shape=self.particle_max_num)
        self.iisph_laplacian = ti.field(dtype=ti.f32, shape=self.particle_max_num)    
        self.iisph_sum_dij = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.particle_max_num)