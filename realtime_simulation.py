import os
import argparse
import taichi as ti
import numpy as np
from SPH.utils import SimConfig
from SPH.containers import DFSPHContainer, IISPHContainer, WCSPHContainer
from SPH.fluid_solvers import DFSPHSolver, IISPHSolver, DFSPH_LSolver, WCSPHSolver

ti.init(arch=ti.gpu, device_memory_fraction=0.8)

class InteractiveFluidSimulator:
    """交互式流体模拟器"""
    
    def __init__(self, scene_path: str):
        """初始化模拟器"""
        self.scene_path = scene_path
        self.scene_name = os.path.splitext(os.path.basename(scene_path))[0]
        self.cfg = SimConfig(scene_file_path=scene_path)
        self._init_simulation_params()
        self._init_visualization()
        self._init_solver()
        
    def _init_simulation_params(self):
        """初始化模拟参数"""
        # 时间相关参数
        self.fps = self.cfg.get_cfg("fps") or 60
        self.dt = self.cfg.get_cfg("timeStepSize")
        self.total_time = self.cfg.get_cfg("totalTime") or 10.0
        self.max_steps = int(self.total_time / self.dt)
        
        # 输出相关参数
        self.output_interval = self.cfg.get_cfg("outputInterval") or \
                             int(1.0 / (self.fps * self.dt))
        self.save_frames = self.cfg.get_cfg("exportFrame")
        self.save_ply = self.cfg.get_cfg("exportPly")
        self.save_obj = self.cfg.get_cfg("exportObj")
        
        # 场景参数
        self.domain_size = self.cfg.get_cfg("domainEnd")
        self.dim = len(self.domain_size)
        self.hidden_objects = self.cfg.get_cfg("invisibleObjects") or []
        
    def _init_visualization(self):
        """初始化可视化组件"""
        # 创建窗口和场景
        self.window = ti.ui.Window('SPH', (1024, 1024), show_window=True, vsync=False)
        self.scene = ti.ui.Scene()
        
        # 设置相机
        self.camera = ti.ui.Camera()
        self.camera.position(5.5, 2.5, 4.0)
        self.camera.up(0.0, 1.0, 0.0)
        self.camera.lookat(-1.0, 0.0, 0.0)
        self.camera.fov(70)
        self.scene.set_camera(self.camera)
        
        # 创建边界框
        self._setup_boundary_box()
        
    def _setup_boundary_box(self):
        """设置边界框"""
        if self.dim == 3:
            self.box_points = ti.Vector.field(3, dtype=ti.f32, shape=8)
            self._init_3d_box()
        else:
            self.box_points = ti.Vector.field(2, dtype=ti.f32, shape=4)
            self._init_2d_box()
            
        # 设置边界线索引
        self.box_edges = ti.field(int, shape=24 if self.dim == 3 else 8)
        self._init_box_edges()
        
    def _init_3d_box(self):
        """初始化3D边界框顶点"""
        x, y, z = self.domain_size
        corners = [
            (0,0,0), (x,0,0), (x,y,0), (0,y,0),
            (0,0,z), (x,0,z), (x,y,z), (0,y,z)
        ]
        for i, pos in enumerate(corners):
            self.box_points[i] = pos
            
    def _init_2d_box(self):
        """初始化2D边界框顶点"""
        x, y = self.domain_size
        corners = [(0,0), (x,0), (x,y), (0,y)]
        for i, pos in enumerate(corners):
            self.box_points[i] = pos
            
    def _init_box_edges(self):
        """初始化边界框边线"""
        if self.dim == 3:
            edges = [0,1, 1,2, 2,3, 3,0,  # 底面
                    4,5, 5,6, 6,7, 7,4,  # 顶面
                    0,4, 1,5, 2,6, 3,7]  # 连接线
        else:
            edges = [0,1, 1,2, 2,3, 3,0]  # 2D边界
            
        for i, idx in enumerate(edges):
            self.box_edges[i] = idx
            
    def _init_solver(self):
        """初始化求解器"""
        method = self.cfg.get_cfg("simulationMethod")
        if method == "dfsph":
            self.container = DFSPHContainer(self.cfg, GGUI=True)
            self.solver = DFSPHSolver(self.container)
        elif method == "iisph":
            self.container = IISPHContainer(self.cfg, GGUI=True)
            self.solver = IISPHSolver(self.container)
        elif method == "dfsph_l":
            self.container = DFSPHContainer(self.cfg, GGUI=True)
            self.solver = DFSPH_LSolver(self.container)
        elif method == "wcsph":
            self.container = WCSPHContainer(self.cfg, GGUI=True)
            self.solver = WCSPHSolver(self.container)
        else:
            raise ValueError(f"不支持的模拟方法: {method}")
            
    def _handle_keyboard_input(self):
        """处理键盘输入"""
        if not self.window.running:
            return False
            
        if self.window.get_event(ti.ui.PRESS):
            if self.window.event.key == ti.ui.ESCAPE:
                return False
                
        velocity_change = 0.5
        for i in range(self.container.max_object_num):
            if not self.container.rigid_body_is_dynamic[i]:
                continue
                
            if self.window.is_pressed(ti.ui.LEFT):
                self.container.rigid_body_velocities[i][0] -= velocity_change
            if self.window.is_pressed(ti.ui.RIGHT):
                self.container.rigid_body_velocities[i][0] += velocity_change
            if self.window.is_pressed(ti.ui.UP):
                self.container.rigid_body_velocities[i][1] += velocity_change
            if self.window.is_pressed(ti.ui.DOWN):
                self.container.rigid_body_velocities[i][1] -= velocity_change
                
        return True
        
    def _update_visualization(self):
        """更新可视化"""
        self.container.copy_to_vis_buffer()
        
        if self.dim == 2:
            canvas = self.window.get_canvas()
            canvas.set_background_color((0, 0, 0))
            canvas.circles(self.container.x_vis_buffer, 
                         radius=self.container.dx/80.0, 
                         color=(1, 1, 1))
        else:
            self.camera.track_user_inputs(self.window, 
                                        movement_speed=0.02, 
                                        hold_key=ti.ui.LMB)
            self.scene.set_camera(self.camera)
            
            # 设置光照和渲染粒子
            self.scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
            self.scene.particles(self.container.x_vis_buffer, 
                               radius=self.container.radius,
                               per_vertex_color=self.container.color_vis_buffer)
            
            # 渲染边界框
            self.scene.lines(self.box_points,
                           indices=self.box_edges,
                           color=(0.99, 0.68, 0.28),
                           width=1.0)
            
            # 更新画布
            canvas = self.window.get_canvas()
            canvas.scene(self.scene)
            
    def run(self):
        """运行模拟"""
        print(f"开始模拟: {self.scene_name}")
        self.solver.prepare()
        step = 0
        
        while step < self.max_steps:
            if not self._handle_keyboard_input():
                break
                
            self.solver.step()
            self._update_visualization()
            self.window.show()
            step += 1
            
        print("模拟完成")


def main():
    parser = argparse.ArgumentParser(description="交互式流体模拟")
    parser.add_argument('--scene_file', default='', help='场景配置文件路径')
    args = parser.parse_args()
    
    simulator = InteractiveFluidSimulator(args.scene_file)
    simulator.run()

if __name__ == "__main__":
    main()