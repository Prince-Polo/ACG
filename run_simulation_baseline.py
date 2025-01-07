import os
import argparse
import numpy as np
from SPH_baseline.utils import SimConfig
from SPH_baseline.containers import DFSPHContainerBaseline
from SPH_baseline.fluid_solvers import DFSPHSolverBaseline

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
        
        self.save_ply = self.config.get_cfg("exportPly")
        self.save_obj = self.config.get_cfg("exportObj")
        
    def _init_domain(self):
        """初始化模拟域"""
        self.domain_end = self.config.get_cfg("domainEnd")
        self.dim = len(self.domain_end)
            
    def _init_solver(self):
        """初始化求解器"""
        method = self.config.get_cfg("simulationMethod")
        solver_map = {
            "dfsph": (DFSPHContainerBaseline, DFSPHSolverBaseline)
        }
        
        if method not in solver_map:
            raise ValueError(f"未支持的模拟方法: {method}")
            
        container_cls, solver_cls = solver_map[method]
        self.container = container_cls(self.config)
        self.solver = solver_cls(self.container)
            
    def _save_outputs(self, step):
        """保存输出文件"""
        if step % self.output_interval != 0:
            return
            
        frame_dir = f"{self.output_root}/{step:06d}"
        os.makedirs(frame_dir, exist_ok=True)
            
        if self.save_ply:
            self._save_particles(frame_dir)
            
        if self.save_obj:
            self._save_meshes(frame_dir)
            
    def _save_particles(self, directory):
        """保存粒子数据"""
        for body_id in self.container.object_id_fluid_body:
            data = self.container.dump(obj_id=body_id)
            pos = data["position"]
            
            writer = PLYWriter(num_vertices=len(pos))
            writer.add_vertex_pos(pos[:, 0], pos[:, 1], 
                                pos[:, 2] if self.dim == 3 else None)
            writer.export_ascii(f"{directory}/fluid_{body_id}.ply")
            
    def _save_meshes(self, directory):
        """保存网格数据"""
        for body_id in self.container.object_id_rigid_body:
            mesh = self.container.object_collection[body_id]["mesh"]
            with open(f"{directory}/rigid_{body_id}.obj", 'w') as f:
                f.write(mesh.export(file_type='obj'))
                
    def execute(self):
        """执行模拟"""
        self.solver.prepare()
        
        for step in range(self.max_steps):
            self.solver.step()
            self._save_outputs(step)
            
        print("模拟完成")


class PLYWriter:
    """PLY文件写入器"""
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.vertex_pos = []
        
    def add_vertex_pos(self, x, y, z=None):
        if z is not None:
            self.vertex_pos = np.column_stack((x, y, z))
        else:
            self.vertex_pos = np.column_stack((x, y))
            
    def export_ascii(self, filename):
        with open(filename, 'w') as f:
            # 写入头部
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {self.num_vertices}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            if self.vertex_pos.shape[1] > 2:
                f.write("property float z\n")
            f.write("end_header\n")
            
            # 写入顶点数据
            np.savetxt(f, self.vertex_pos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="物理模拟系统")
    parser.add_argument('--scene_file', required=True, help='场景配置文件路径')
    args = parser.parse_args()
    
    simulator = PhysicsSimulator(args.scene_file)
    simulator.execute()