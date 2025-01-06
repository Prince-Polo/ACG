import numpy as np
from .base_container import BaseContainerBaseline
from ..utils import SimConfig

class DFSPHContainerBaseline(BaseContainerBaseline):
    def __init__(self, config: SimConfig, GGUI=False):
        super().__init__(config, GGUI)
        self.particle_dfsph_derivative_densities = np.zeros(self.particle_max_num, dtype=np.float32)
        self.particle_dfsph_factor_k = np.zeros(self.particle_max_num, dtype=np.float32)
        self.particle_dfsph_predict_density = np.zeros(self.particle_max_num, dtype=np.float32)
        self.particle_dfsph_pressure = np.zeros(self.particle_max_num, dtype=np.float32)
        self.particle_dfsph_pressure_v = np.zeros(self.particle_max_num, dtype=np.float32)

        self.m_max_iterations_v = 1000
        self.m_max_iterations = 1000

        self.m_eps = 1e-5

        self.max_error_V = 0.001
        self.max_error = 0.0001