import numpy as np
from .base_solver_numba import BaseSolver
from ..containers import DFSPHContainer
from ..utils import SimConfig
import os

class DFSPHSolver(BaseSolver):
    def __init__(self, config: SimConfig):
        """初始化DFSPH求解器"""
        super().__init__(config)
        
        # DFSPH特定参数
        self.enable_divergence_solve = self.cfg.get_cfg("enableDivergenceSolve", True)
        self.max_iterations = self.cfg.get_cfg("maxIterations", 100)
        self.max_iterations_v = self.cfg.get_cfg("maxIterationsV", 100)
        self.density_error = self.cfg.get_cfg("densityError", 0.01)
        self.divergence_error = self.cfg.get_cfg("divergenceError", 0.01)
        self.omega = self.cfg.get_cfg("omega", 0.5)
        
    def _init_container(self):
        """初始化DFSPH容器"""
        self.container = DFSPHContainer(self.cfg, self.GGUI)
        
        # 设置DFSPH特定参数
        self.container.max_iterations = self.max_iterations
        self.container.max_iterations_v = self.max_iterations_v
        self.container.density_error = self.density_error
        self.container.divergence_error = self.divergence_error
        self.container.omega = self.omega
    
    def step(self):
        """执行一个DFSPH时间步"""
        # 更新邻居搜索
        self.container.prepare_neighbor_search()
        
        # 计算基本SPH
        self.container.compute_density()
        self.container.compute_non_pressure_forces()
        
        # 预测速度
        dt = self.container.dt
        for i in range(self.container.particle_num):
            for d in range(3):
                self.container.d_particle_velocities[i, d] += \
                    dt * self.container.d_particle_forces[i, d] / self.container.d_particle_masses[i]
        
        # 求解速度散度（如果启用）
        if self.enable_divergence_solve:
            self.container.divergence_solve()
        
        # 求解压力
        self.container.pressure_solve()
        
        # 更新位置
        for i in range(self.container.particle_num):
            for d in range(3):
                self.container.d_particle_positions[i, d] += \
                    dt * self.container.d_particle_velocities[i, d]
        
        # 处理边界碰撞
        self.container.handle_boundary_collision()
        
        # 更新时间
        self.total_time += dt
        self.current_step += 1
    
    def _output(self):
        """输出模拟结果"""
        # 确保输出目录存在
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        # 构建输出文件名
        filename = f"{self.output_prefix}_{self.current_step:06d}.npz"
        filepath = os.path.join(self.output_path, filename)
        
        # 获取数据
        positions = self.container.d_particle_positions.copy_to_host()[:self.container.particle_num]
        velocities = self.container.d_particle_velocities.copy_to_host()[:self.container.particle_num]
        densities = self.container.d_particle_densities.copy_to_host()[:self.container.particle_num]
        pressures = self.container.d_particle_pressures.copy_to_host()[:self.container.particle_num]
        object_ids = self.container.d_particle_object_ids.copy_to_host()[:self.container.particle_num]
        
        # 保存数据
        np.savez(filepath,
                 positions=positions,
                 velocities=velocities,
                 densities=densities,
                 pressures=pressures,
                 object_ids=object_ids,
                 time=self.total_time)
    
    def get_solver_info(self):
        """获取求解器信息"""
        return {
            "type": "DFSPH",
            "particle_num": self.container.particle_num,
            "fluid_particle_num": self.container.fluid_particle_num,
            "rigid_particle_num": self.container.rigid_particle_num,
            "time_step": self.container.dt,
            "current_time": self.total_time,
            "current_step": self.current_step,
            "enable_divergence_solve": self.enable_divergence_solve,
            "max_iterations": self.max_iterations,
            "max_iterations_v": self.max_iterations_v,
            "density_error": self.density_error,
            "divergence_error": self.divergence_error
        }
    
    def get_simulation_stats(self):
        """获取模拟统计信息"""
        # 获取粒子数据
        densities = self.container.d_particle_densities.copy_to_host()[:self.container.particle_num]
        velocities = self.container.d_particle_velocities.copy_to_host()[:self.container.particle_num]
        
        # 计算统计信息
        stats = {
            "density": {
                "min": np.min(densities),
                "max": np.max(densities),
                "mean": np.mean(densities),
                "std": np.std(densities)
            },
            "velocity": {
                "magnitude_mean": np.mean(np.linalg.norm(velocities, axis=1)),
                "magnitude_max": np.max(np.linalg.norm(velocities, axis=1))
            }
        }
        
        return stats 