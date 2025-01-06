import taichi as ti
import numpy as np
from ..containers import BaseContainer
from ..utils.kernel import *
from ..utils.boundary import Boundary
from ..rigid_solver import RigidSolver


@ti.data_oriented
class BaseSolver():
    def __init__(self, container: BaseContainer):
        self.container = container
        self.cfg = container.cfg

        # Gravity
        self.g = np.array(self.cfg.get_cfg("gravitation"))

        # density
        self.density = 1000.0
        self.density_0 = self.cfg.get_cfg("density0")

        # surface tension
        if self.cfg.get_cfg("surface_tension"):
            self.surface_tension = self.cfg.get_cfg("surface_tension")
        else:
            self.surface_tension = 0.01

        # viscosity
        self.viscosity = self.cfg.get_cfg("viscosity")
        if self.cfg.get_cfg("viscosity_b"):
            self.viscosity_b = self.cfg.get_cfg("viscosity_b")
        else:
            self.viscosity_b = self.viscosity

        # time step
        self.dt = ti.field(float, shape=())
        self.dt[None] = self.cfg.get_cfg("timeStepSize")

        # kernel
        self.kernel = CubicKernel()

        # boundary
        self.boundary = Boundary(self.container)

        # others
        if self.cfg.get_cfg("gravitationUpper"):
            self.g_upper = self.cfg.get_cfg("gravitationUpper")
        else:
            self.g_upper = 10000.0

        # rigid solver
        self.rigid_solver = RigidSolver(self.container, gravity=self.g, dt=self.dt[None])

        # boundary
        self.boundary = Boundary(self.container)

    @ti.kernel
    def compute_rigid_particle_volume(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid:
                if self.container.particle_positions[p_i][1] <= self.g_upper:
                    # Initialize the volume with the kernel weight at zero distance
                    volume = self.kernel.weight(0.0, self.container.dh)
                    # Iterate over all neighbors to compute the volume
                    self.container.for_all_neighbors(p_i, self.compute_rigid_particle_volume_task, volume)
                    # Compute the rest volume and mass of the particle
                    self.container.particle_rest_volumes[p_i] = 1.0 / volume
                    self.container.particle_masses[p_i] = self.density_0 * self.container.particle_rest_volumes[p_i]

    @ti.func
    def compute_rigid_particle_volume_task(self, p_i, p_j, volume: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        if self.container.particle_object_ids[p_j] == self.container.particle_object_ids[p_i]:
            pos_j = self.container.particle_positions[p_j]
            distance = (pos_i - pos_j).norm()
            # Accumulate the kernel weight based on the distance between particles
            volume += self.kernel.weight(distance, self.container.dh)

    def compute_non_pressure_acceleration(self):
        """计算非压力加速度项"""
        self._process_gravity()
        self._process_surface()
        self.compute_viscosity_acceleration()
    
    @ti.kernel
    def _process_gravity(self):
        for idx in range(self.container.particle_num[None]):
            if self.container.particle_materials[idx] == self.container.material_fluid:
                self.container.particle_accelerations[idx] = ti.Vector(self.g)
    
    @ti.kernel
    def _process_surface(self):
        for p_i in range(self.container.particle_num[None]):
            self._handle_particle_tension(p_i)
                
    @ti.func
    def _handle_particle_tension(self, p_i: int):
        material = self.container.particle_materials[p_i]
        if material == self.container.material_fluid:
            acc = ti.Vector([0.0 for _ in range(self.container.dim)])
            self._accumulate_tension_force(p_i, acc)
            self._apply_tension_acc(p_i, acc)
                
    @ti.func
    def _accumulate_tension_force(self, p_i: int, acc: ti.template()):
        self.container.for_all_neighbors(p_i, self.compute_surface_tension_acceleration_task, acc)
                
    @ti.func
    def _apply_tension_acc(self, p_i: int, acc: ti.template()):
        self.container.particle_accelerations[p_i] += acc

    @ti.func
    def compute_surface_tension_acceleration_task(self, p_i, p_j, acc: ti.template()):
        """计算单个粒子对的表面张力"""
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            self._compute_tension_contribution(p_i, p_j, acc)
                
    @ti.func
    def _compute_tension_contribution(self, p_i: int, p_j: int, acc: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        rel_pos = pos_i - pos_j
        
        dist2 = rel_pos.norm_sqr()
        weight = self._calculate_weight(dist2, rel_pos)
        mass_r = self.container.particle_masses[p_j] / self.container.particle_masses[p_i]
        
        acc -= self.surface_tension * mass_r * rel_pos * weight
                
    @ti.func
    def _calculate_weight(self, dist2: float, rel_pos: ti.template()) -> float:
        diam2 = self.container.diameter * self.container.diameter
        dist = rel_pos.norm()
        return (self.kernel.weight(dist, self.container.dh) 
                if dist2 > diam2 
                else self.kernel.weight(self.container.diameter, self.container.dh))

    @ti.kernel
    def compute_viscosity_acceleration(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                a_i = ti.Vector([0.0 for _ in range(self.container.dim)])
                self.container.for_all_neighbors(p_i, self.compute_viscosity_acceleration_task, a_i)
                self.container.particle_accelerations[p_i] += (a_i / self.container.particle_rest_densities[p_i])

    @ti.func
    def compute_viscosity_acceleration_task(self, p_i, p_j, a_i: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        nabla_ij = self.kernel.gradient(R, self.container.dh)
        v_xy = ti.math.dot(self.container.particle_velocities[p_i] - self.container.particle_velocities[p_j], R)

        if self.container.particle_materials[p_j] == self.container.material_fluid:
            avg_mass = (self.container.particle_masses[p_i] + self.container.particle_masses[p_j])/2
            regular_viscosity = self.viscosity / self.container.particle_densities[p_j]
            a_i +=  2 * (self.container.dim + 2) * regular_viscosity * avg_mass * v_xy * nabla_ij / (R.norm()**2 + 0.01 * self.container.dh**2)

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            rigid_mass = self.density_0 * self.container.particle_rest_volumes[p_j]
            regular_viscosity_b = self.viscosity_b / self.container.particle_densities[p_i]
            acc = 2 * (self.container.dim + 2) * regular_viscosity_b * rigid_mass * v_xy * nabla_ij / (R.norm()**2 + 0.01 * self.container.dh**2)

            a_i += acc

            if self.container.particle_is_dynamic[p_j]:
                # If the rigid particle is dynamic, compute the force and torque
                object_j = self.container.particle_object_ids[p_j]
                self.container.rigid_body_forces[object_j] += -acc * self.container.particle_masses[p_i] / self.density_0
                self.container.rigid_body_torques[object_j] += ti.math.cross(pos_j - self.container.rigid_body_com[object_j], -acc * self.container.particle_masses[p_i] / self.density_0)

    @ti.kernel
    def compute_density(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                # Initialize density with the self-contribution
                density = self.container.particle_masses[p_i] * self.kernel.weight(0.0, self.container.dh)
                # Iterate over all neighbors to compute the density
                self.container.for_all_neighbors(p_i, self.compute_density_task, density)
                # Scale the density by the reference density
                self.container.particle_densities[p_i] = density

    @ti.func
    def compute_density_task(self, p_i, p_j, density: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R_mod = (pos_i - pos_j).norm()
        # Accumulate the density contribution from particle j
        density += self.container.particle_masses[p_j] * self.kernel.weight(R_mod, self.container.dh)

    @ti.kernel
    def _renew_rigid_particle_state(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid and self.container.particle_is_dynamic[p_i]:
                object_id = self.container.particle_object_ids[p_i]
                if self.container.rigid_body_is_dynamic[object_id]:
                    # Calculate the rotated position
                    p = self.container.rigid_body_rotations[object_id] @ (self.container.rigid_particle_original_positions[p_i] - self.container.rigid_body_original_com[object_id])
                    # Update particle position
                    self.container.particle_positions[p_i] = self.container.rigid_body_com[object_id] + p
                    # Update particle velocity
                    self.container.particle_velocities[p_i] = self.container.rigid_body_velocities[object_id] + ti.math.cross(self.container.rigid_body_angular_velocities[object_id], p)

    def renew_rigid_particle_state(self):
        self._renew_rigid_particle_state()
    
        if self.cfg.get_cfg("exportObj"):
            for obj_i in range(self.container.object_num[None]):
                if self.container.rigid_body_is_dynamic[obj_i] and self.container.object_materials[obj_i] == self.container.material_rigid:
                    # Update the mesh vertices based on the new rigid body state
                    self.container.object_collection[obj_i]["mesh"].vertices = (self.container.rigid_body_rotations[obj_i].to_numpy() @ (self.container.object_collection[obj_i]["restPosition"] - self.container.object_collection[obj_i]["restCenterOfMass"]).T).T + self.container.rigid_body_com[obj_i].to_numpy()

    @ti.kernel
    def update_fluid_velocity(self):
        """更新流体粒子速度"""
        for p_i in range(self.container.particle_num[None]):
            self._process_velocity_update(p_i)
                
    @ti.func
    def _process_velocity_update(self, p_i: int):
        if self._is_fluid_particle(p_i):
            self._apply_acceleration(p_i)
            
    @ti.func
    def _apply_acceleration(self, idx: int):
        acc = self.container.particle_accelerations[idx]
        vel = self.container.particle_velocities[idx]
        self.container.particle_velocities[idx] = vel + self.dt[None] * acc

    @ti.kernel
    def update_fluid_position(self):
        """更新流体粒子位置"""
        for p_i in range(self.container.particle_num[None]):
            self._handle_position_update(p_i)
            
    @ti.func
    def _handle_position_update(self, p_i: int):
        pos = self.container.particle_positions[p_i]
        if self._is_fluid_particle(p_i):
            self._update_fluid_pos(p_i)
        elif pos[1] > self.g_upper:
            self._handle_emitter_particle(p_i)
            
    @ti.func
    def _update_fluid_pos(self, idx: int):
        vel = self.container.particle_velocities[idx]
        pos = self.container.particle_positions[idx]
        self.container.particle_positions[idx] = pos + self.dt[None] * vel
            
    @ti.func
    def _handle_emitter_particle(self, p_i: int):
        obj_id = self.container.particle_object_ids[p_i]
        if self.container.object_materials[obj_id] == self.container.material_fluid:
            self._process_emitter_movement(p_i)
            
    @ti.func
    def _process_emitter_movement(self, idx: int):
        pos = self.container.particle_positions[idx]
        vel = self.container.particle_velocities[idx]
        new_pos = pos + self.dt[None] * vel
        self.container.particle_positions[idx] = new_pos
        if new_pos[1] <= self.g_upper:
            self.container.particle_materials[idx] = self.container.material_fluid

    @ti.kernel
    def prepare_emitter(self):
        for p_i in range(self.container.particle_num[None]):
            if self._should_convert_to_rigid(p_i):
                self.container.particle_materials[p_i] = self.container.material_rigid
            
    @ti.func
    def _should_convert_to_rigid(self, idx: int) -> ti.i32:
        pos = self.container.particle_positions[idx]
        return self.container.particle_materials[idx] == self.container.material_fluid and pos[1] > self.g_upper

    @ti.kernel
    def init_object_id(self):
        """初始化对象ID"""
        for i in range(self.container.particle_num[None]):
            self.container.particle_object_ids[i] = -1
            
    def prepare(self):
        print("initializing object id")
        self.init_object_id()
        print("inserting object")
        self.container.insert_object()
        print("inserting emitter")
        self.prepare_emitter()
        print("inserting rigid object")
        self.rigid_solver.insert_rigid_object()
        print("renewing rigid particle state")
        self.renew_rigid_particle_state()
        print("preparing neighborhood search")
        self.container.prepare_neighbor_search()
        print("computing volume")
        self.compute_rigid_particle_volume()
        print("preparing finished")
        self.container.object_num[None] = self.container.fluid_object_num[None] + self.container.rigid_object_num[None] + (1 if self.container.add_boundary else 0)
        print(f"Total object num: {self.container.object_num[None]}")
    
    def step(self):
        self._step()
        self.container.total_time += self.dt[None]
        self.rigid_solver.total_time += self.dt[None]
        self.compute_rigid_particle_volume()
