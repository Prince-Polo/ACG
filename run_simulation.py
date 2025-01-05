import os
import argparse
import taichi as ti
import numpy as np
from SPH.utils import SimConfig
from SPH.containers import DFSPHContainer, IISPHContainer, BaseContainer
from SPH.fluid_solvers import DFSPHSolver, IISPHSolver, DFSPH_LSolver, WCSPHSolver

ti.init(arch=ti.gpu, device_memory_fraction=0.8)

class PhysicsSimulator:
    """物理模拟系统"""
    
    def __init__(self, scene_path: str):
        self.scene_path = scene_path
        self.scene_name = os.path.splitext(os.path.basename(scene_path))[0]
        self.config = SimConfig(scene_file_path=scene_path)
        self._prepare_simulation()
        
    def _prepare_simulation(self):
        """准备模拟环境"""
        self._init_timing()
        self._init_output()
        self._init_domain()
        self._init_solver()
        self._init_render_window()
        
    def _init_timing(self):
        """初始化时间参数"""
        self.time_step = self.config.get_cfg("timeStepSize")
        self.fps = self.config.get_cfg("fps") or 60
        self.total_time = self.config.get_cfg("totalTime") or 10.0
        self.output_interval = self.config.get_cfg("outputInterval") or int(1.0 / (self.fps * self.time_step))
        self.max_steps = int(self.total_time / self.time_step)
        
    def _init_output(self):
        """初始化输出设置"""
        self.output_root = f"output_{self.scene_name}"
        os.makedirs(self.output_root, exist_ok=True)
        
        self.save_frame = self.config.get_cfg("exportFrame")
        self.save_ply = self.config.get_cfg("exportPly")
        self.save_obj = self.config.get_cfg("exportObj")
        
    def _init_domain(self):
        """初始化模拟域"""
        self.domain_end = self.config.get_cfg("domainEnd")
        self.dim = len(self.domain_end)
        self._setup_boundary()
        
    def _setup_boundary(self):
        """设置边界数据"""
        if self.dim == 3:
            self._setup_3d_boundary()
        else:
            self._setup_2d_boundary()
            
    def _setup_3d_boundary(self):
        """设置3D边界"""
        x, y, z = self.domain_end
        vertices = []
        indices = []
        
        # 定义立方体的8个顶点
        corners = [
            (0,0,0), (x,0,0), (x,y,0), (0,y,0),
            (0,0,z), (x,0,z), (x,y,z), (0,y,z)
        ]
        
        # 定义12条边的连接关系
        edges = [
            (0,1), (1,2), (2,3), (3,0),  # 底面
            (4,5), (5,6), (6,7), (7,4),  # 顶面
            (0,4), (1,5), (2,6), (3,7)   # 连接线
        ]
        
        # 创建顶点和索引数组
        self.box_vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(corners))
        self.box_indices = ti.field(int, shape=len(edges) * 2)
        
        # 填充数据
        for i, pos in enumerate(corners):
            self.box_vertices[i] = pos
            
        for i, (start, end) in enumerate(edges):
            self.box_indices[i*2] = start
            self.box_indices[i*2+1] = end
            
    def _setup_2d_boundary(self):
        """设置2D边界"""
        x, y = self.domain_end
        corners = [(0,0), (x,0), (x,y), (0,y)]
        
        self.box_vertices = ti.Vector.field(2, dtype=ti.f32, shape=4)
        self.box_indices = ti.field(int, shape=8)
        
        for i, pos in enumerate(corners):
            self.box_vertices[i] = pos
            self.box_indices[i*2] = i
            self.box_indices[i*2+1] = (i + 1) % 4
            
    def _init_solver(self):
        """初始化求解器"""
        method = self.config.get_cfg("simulationMethod")
        solver_map = {
            "dfsph": (DFSPHContainer, DFSPHSolver),
            "iisph": (IISPHContainer, IISPHSolver),
            "dfsph_L": (DFSPHContainer, DFSPH_LSolver),
            "wcsph": (BaseContainer, WCSPHSolver)
        }
        
        if method not in solver_map:
            raise ValueError(f"未支持的模拟方法: {method}")
            
        container_cls, solver_cls = solver_map[method]
        self.container = container_cls(self.config, GGUI=True)
        self.solver = solver_cls(self.container)
        
    def _init_render_window(self):
        """初始化渲染窗口"""
        self.window = ti.ui.Window("物理模拟", (1024, 1024), 
                                 show_window=False, vsync=False)
        self.scene = self.window.get_scene()
        self._setup_camera()
        
    def _setup_camera(self):
        """设置相机参数"""
        camera = ti.ui.Camera()
        camera.position(4.0, 3.0, 3.0)
        camera.lookat(0.0, 1.0, 0.0)
        camera.fov(65)
        self.scene.set_camera(camera)
        
    def _save_outputs(self, step):
        """保存输出文件"""
        if step % self.output_interval != 0:
            return
            
        frame_dir = f"{self.output_root}/{step:06d}"
        os.makedirs(frame_dir, exist_ok=True)
        
        if self.save_frame:
            self.window.save_image(f"{frame_dir}/frame.png")
            
        if self.save_ply:
            self._save_particles(frame_dir)
            
        if self.save_obj:
            self._save_meshes(frame_dir)
            
    def _save_particles(self, directory):
        """保存粒子数据"""
        for body_id in self.container.object_id_fluid_body:
            data = self.container.dump(obj_id=body_id)
            pos = data["position"]
            
            writer = ti.tools.PLYWriter(num_vertices=len(pos))
            writer.add_vertex_pos(pos[:, 0], pos[:, 1], 
                                pos[:, 2] if self.dim == 3 else None)
            writer.export_ascii(f"{directory}/fluid_{body_id}.ply")
            
    def _save_meshes(self, directory):
        """保存网格数据"""
        for body_id in self.container.object_id_rigid_body:
            mesh = self.container.object_collection[body_id]["mesh"]
            with open(f"{directory}/rigid_{body_id}.obj", 'w') as f:
                f.write(mesh.export(file_type='obj'))
                
    def _update_visualization(self):
        """更新可视化"""
        self.container.copy_to_vis_buffer(
            invisible_objects=self.config.get_cfg("invisibleObjects", []),
            dim=self.dim
        )
        
        # 设置场景
        self.scene.ambient_light((0.1, 0.1, 0.15))
        self.scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))
        
        # 渲染粒子
        self.scene.particles(self.container.x_vis_buffer,
                            radius=self.container.radius,
                            per_vertex_color=self.container.color_vis_buffer)
        
        # 渲染边界
        self.scene.lines(self.box_vertices,
                        indices=self.box_indices,
                        color=(0.6, 0.6, 0.8),
                        width=2.0)
        
        # 更新画布
        canvas = self.window.get_canvas()
        canvas.scene(self.scene)
        
    def execute(self):
        """执行模拟"""
        self.solver.prepare()
        
        for step in range(self.max_steps):
            if not self.window.running:
                break
                
            self.solver.step()
            self._update_visualization()
            self._save_outputs(step)
            
        print("模拟完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="物理模拟系统")
    parser.add_argument('--scene_file', required=True, help='场景配置文件路径')
    args = parser.parse_args()
    
    simulator = PhysicsSimulator(args.scene_file)
    simulator.execute()