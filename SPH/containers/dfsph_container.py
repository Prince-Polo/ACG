import taichi as ti 
from .base_container import BaseContainer
from ..utils import SimConfig

@ti.data_oriented
class DFSPHContainer(BaseContainer):
    def __init__(self, config: SimConfig, GGUI=False):
        super().__init__(config, GGUI)
        self.particle_dfsph_derivative_densities = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_dfsph_factor_k = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_dfsph_predict_density = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_dfsph_pressure = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_dfsph_pressure_v = ti.field(dtype=float, shape=self.particle_max_num)

