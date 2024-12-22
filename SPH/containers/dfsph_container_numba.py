from ..utils.config_builder import SimConfig
import numpy as np
from numba import cuda, float32, int32
import math
from .base_container_numba import BaseContainer

class DFSPHContainer(BaseContainer):
    def __init__(self, config: SimConfig, GGUI=False):
        """初始化DFSPH容器"""
        super().__init__(config, GGUI)
        
        # DFSPH特定参数
        self.particle_dfsph_derivative_densities = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.float32))
        self.particle_dfsph_factor_k = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.float32))
        self.particle_dfsph_predict_density = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.float32))
        self.particle_dfsph_pressure = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.float32))
        self.particle_dfsph_pressure_v = cuda.to_device(np.zeros(self.particle_max_num, dtype=np.float32))