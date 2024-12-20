import numpy as np
from ..utils import SimConfig
from ..containers import BaseContainer

class BaseSolver:
    def __init__(self, config: SimConfig):
        """初始化基础求解器"""
        self.cfg = config
        self.container = None
        self.total_time = 0.0
        self.current_step = 0
        self.max_steps = self.cfg.get_cfg("maxSteps")
        self.output_interval = self.cfg.get_cfg("outputInterval")
        self.output_path = self.cfg.get_cfg("outputPath")
        self.output_prefix = self.cfg.get_cfg("outputPrefix")
        self.GGUI = self.cfg.get_cfg("GGUI", False)
        
        # 初始化容器
        self._init_container()
        
        # 初始化场景
        self._init_scene()
    
    def _init_container(self):
        """初始化粒子容器"""
        self.container = BaseContainer(self.cfg, self.GGUI)
    
    def _init_scene(self):
        """初始化场景"""
        # 添加流体
        self._add_fluid()
        
        # 添加刚体
        self._add_rigid_bodies()
        
        # 准备模拟
        self.container.prepare_neighbor_search()
    
    def _add_fluid(self):
        """添加流体"""
        # 从配置中获取流体参数
        fluid_blocks = self.cfg.get_cfg("fluidBlocks", [])
        fluid_bodies = self.cfg.get_cfg("fluidBodies", [])
        
        # 添加流体块
        for block in fluid_blocks:
            start = np.array(block.get("start", [0, 0, 0]))
            end = np.array(block.get("end", [1, 1, 1]))
            velocity = np.array(block.get("velocity", [0, 0, 0]))
            density = block.get("density", 1000.0)
            
            self.container.add_cube(
                object_id=self.container.object_num,
                start=start,
                end=end,
                material=self.container.material_fluid,
                velocity=velocity,
                density=density,
                is_dynamic=True,
                color=(51, 153, 255)
            )
            
            self.container.object_num += 1
            self.container.fluid_object_num += 1
        
        # 添加流体物体
        for body in fluid_bodies:
            file_path = body.get("file", "")
            scale = np.array(body.get("scale", [1, 1, 1]))
            translation = np.array(body.get("translation", [0, 0, 0]))
            velocity = np.array(body.get("velocity", [0, 0, 0]))
            density = body.get("density", 1000.0)
            
            if file_path:
                self.container.add_body(
                    object_id=self.container.object_num,
                    file_path=file_path,
                    scale=scale,
                    translation=translation,
                    material=self.container.material_fluid,
                    velocity=velocity,
                    density=density,
                    is_dynamic=True,
                    color=(51, 153, 255)
                )
                
                self.container.object_num += 1
                self.container.fluid_object_num += 1
    
    def _add_rigid_bodies(self):
        """添加刚体"""
        rigid_bodies = self.cfg.get_cfg("rigidBodies", [])
        
        for body in rigid_bodies:
            file_path = body.get("file", "")
            scale = np.array(body.get("scale", [1, 1, 1]))
            translation = np.array(body.get("translation", [0, 0, 0]))
            velocity = np.array(body.get("velocity", [0, 0, 0]))
            density = body.get("density", 1000.0)
            is_dynamic = body.get("isDynamic", True)
            
            if file_path:
                self.container.add_body(
                    object_id=self.container.object_num,
                    file_path=file_path,
                    scale=scale,
                    translation=translation,
                    material=self.container.material_rigid,
                    velocity=velocity,
                    density=density,
                    is_dynamic=is_dynamic,
                    color=(128, 128, 128)
                )
                
                self.container.object_num += 1
                self.container.rigid_object_num += 1
    
    def step(self):
        """执行一个时间步"""
        self.container.step()
        self.total_time += self.container.dt
        self.current_step += 1
    
    def run(self):
        """运行模拟"""
        while self.current_step < self.max_steps:
            self.step()
            
            # 输出
            if self.current_step % self.output_interval == 0:
                self._output()
    
    def _output(self):
        """输出结果"""
        # 可以在子类中实现具体的输出逻辑
        pass
    
    def get_current_time(self):
        """获取当前模拟时间"""
        return self.total_time
    
    def get_particle_positions(self):
        """获取粒子位置"""
        return self.container.d_particle_positions.copy_to_host()[:self.container.particle_num]
    
    def get_particle_colors(self):
        """获取粒子颜色"""
        return self.container.d_particle_colors.copy_to_host()[:self.container.particle_num] 