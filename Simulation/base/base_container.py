import taichi as ti
import numpy as np
import trimesh as tm
from functools import reduce
from utils import SimConfig

@ti.data_oriented
class BaseContainer:
    def __init__(self, config: SimConfig, GGUI=False):
        self.cfg = config
        self.GGUI = GGUI
        self.total_time = 0.0

        self.domain_start = np.array([0.0, 0.0, 0.0])
        self.domain_start = np.array(self.cfg.get_cfg("domainStart"))

        assert self.domain_start[1] >= 0.0, "domain start y should be greater than 0"

        self.domain_end = np.array([1.0, 1.0, 1.0])
        self.domain_end = np.array(self.cfg.get_cfg("domainEnd"))
        
        self.domain_size = self.domain_end - self.domain_start

        self.dim = len(self.domain_size)
        print(f"Dimension: {self.dim}")

        # 材料类型
        self.material_rigid = 2
        self.material_fluid = 1

        # 粒子半径和间距
        self.dx = 0.01  # 粒子半径
        self.dx = self.cfg.get_cfg("particleRadius")

        self.particle_diameter = 2 * self.dx
        if self.dim == 3:
            self.dh = self.dx * 4.0  # 支持半径
        else:
            self.dh = self.dx * 3.0  # 支持半径

        if self.cfg.get_cfg("supportRadius"):
            self.dh = self.cfg.get_cfg("supportRadius")
        
        self.particle_spacing = self.particle_diameter
        if self.cfg.get_cfg("particleSpacing"):
            self.particle_spacing = self.cfg.get_cfg("particleSpacing")

        self.V0 = 0.8 * self.particle_diameter ** self.dim
        self.particle_num = ti.field(int, shape=())

        self.max_num_object = 20

        # 网格相关参数
        self.grid_size = self.dh
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        print("grid size: ", self.grid_num)
        self.padding = self.grid_size

        self.add_domain_box = self.cfg.get_cfg("addDomainBox")
        if self.add_domain_box:
            self.domain_box_start = [self.domain_start[i] + self.padding for i in range(self.dim)]
            self.domain_box_size = [self.domain_size[i] - 2 * self.padding for i in range(self.dim)]
            self.domain_box_thickness = 0.03
        else:
            self.domain_box_thickness = 0.0

        # 对象集合和标识
        self.object_collection = dict()
        self.object_id_rigid_body = set()
        self.object_id_fluid_body = set()
        self.present_object = []

        # 计算粒子数量
        fluid_particle_num = 0
        rigid_body_particle_num = 0

        self.fluid_bodies = self.cfg.get_fluid_bodies()
        for fluid_body in self.fluid_bodies:
            voxelized_points_np = self.load_fluid_body(fluid_body, pitch=self.particle_spacing)
            fluid_body["particleNum"] = voxelized_points_np.shape[0]
            fluid_body["voxelizedPoints"] = voxelized_points_np
            fluid_particle_num += voxelized_points_np.shape[0]

        self.fluid_blocks = self.cfg.get_fluid_blocks()
        for fluid in self.fluid_blocks:
            particle_num = self.compute_cube_particle_num(fluid["start"], fluid["end"], space=self.particle_spacing)
            fluid["particleNum"] = particle_num
            fluid_particle_num += particle_num

        num_fluid_object = len(self.fluid_blocks) + len(self.fluid_bodies)

        self.rigid_bodies = self.cfg.get_rigid_bodies()
        for rigid_body in self.rigid_bodies:
            voxelized_points_np = self.load_rigid_body(rigid_body, pitch=self.particle_spacing)
            rigid_body["particleNum"] = voxelized_points_np.shape[0]
            rigid_body["voxelizedPoints"] = voxelized_points_np
            rigid_body_particle_num += voxelized_points_np.shape[0]

        self.rigid_blocks = self.cfg.get_rigid_blocks()
        for rigid_block in self.rigid_blocks:
            raise NotImplementedError

        num_rigid_object = len(self.rigid_blocks) + len(self.rigid_bodies)
        print(f"Number of rigid bodies and rigid blocks: {num_rigid_object}")

        self.fluid_particle_num = fluid_particle_num
        self.rigid_body_particle_num = rigid_body_particle_num
        self.particle_max_num = (
            fluid_particle_num 
            + rigid_body_particle_num 
            + (self.compute_box_particle_num(self.domain_box_start, self.domain_box_size, space=self.particle_spacing, thickness=self.domain_box_thickness) if self.add_domain_box else 0)
        )
        
        print(f"Fluid particle num: {self.fluid_particle_num}, Rigid body particle num: {self.rigid_body_particle_num}")

        self.fluid_particle_num = ti.field(int, shape=())

@ti.data_oriented
class DFSPHContainer(BaseContainer):
    def __init__(self, config: SimConfig, GGUI=False):
        super().__init__(config, GGUI)
        # additional dfsph related property
        self.particle_dfsph_alphas = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_dfsph_kappa = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_dfsph_kappa_v = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_densities_star = ti.field(dtype=float, shape=self.particle_max_num)
        self.particle_densities_derivatives = ti.field(dtype=float, shape=self.particle_max_num)