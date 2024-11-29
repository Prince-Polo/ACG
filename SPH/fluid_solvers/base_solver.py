import taichi as ti
import numpy as np
from ..containers import BaseContainer
from .utils import *
from .boundary import Boundary
from ..rigid_solver import RigidSolver


@ti.data_oriented
class BaseSolver():
    def __init__(self, container: BaseContainer):
        self.container = container
        self.cfg = container.cfg

        # Gravity
        self.g = np.array(self.container.cfg.get_cfg("gravitation"))

        # density
        self.density = 1000.0
        self.density_0 = self.container.cfg.get_cfg("density0")

        # surface tension
        if self.container.cfg.get_cfg("surface_tension"):
            self.surface_tension = self.container.cfg.get_cfg("surface_tension")
        else:
            self.surface_tension = 0.01

        # viscosity
        self.viscosity = self.container.cfg.get_cfg("viscosity")
        if self.container.cfg.get_cfg("viscosity_b"):
            self.viscosity_b = self.container.cfg.get_cfg("viscosity_b")
        else:
            self.viscosity_b = self.viscosity

        # time step
        self.dt = ti.field(float, shape=())
        self.dt[None] = 1e-4
        self.dt[None] = self.container.cfg.get_cfg("timeStepSize")

        # kernel
        self.kernel = CubicKernel()

        # boundary
        self.boundary = Boundary(self.container)

        # others
        if self.container.cfg.get_cfg("gravitationUpper"):
            self.g_upper = self.container.cfg.get_cfg("gravitationUpper")
        else:
            self.g_upper = 10000.0

        # rigid solver
        self.rigid_solver = RigidSolver(self.container, gravity=self.g, dt=self.dt[None])

        # boundary
        self.boundary = Boundary(self.container)

    @ti.kernel
    def compute_rigid_particle_volume(self):
        # implemented as paper "Versatile Rigid-Fluid Coupling for Incompressible SPH"
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid:
                if self.container.particle_positions[p_i][1] <= self.g_upper:
                    ret = self.kernel.weight(0.0, self.container.dh)
                    self.container.for_all_neighbors(p_i, self.compute_rigid_particle_volumn_task, ret)
                    self.container.particle_rest_volumes[p_i] = 1.0 / ret 
                    self.container.particle_masses[p_i] = self.density_0 * self.container.particle_rest_volumes[p_i]
            else:
                self.container.particle_masses[p_i] = self.density * self.container.particle_rest_volumes[p_i]

    @ti.func
    def compute_rigid_particle_volumn_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        if self.container.particle_object_ids[p_j] == self.container.particle_object_ids[p_i]:
            pos_j = self.container.particle_positions[p_j]
            R = pos_i - pos_j
            R_mod = R.norm()
            ret += self.kernel.weight(R_mod, self.container.dh)

    @ti.kernel
    def init_acceleration(self):
        self.container.particle_accelerations.fill(0.0)

    @ti.kernel
    def init_rigid_body_force_and_torque(self):
        self.container.rigid_body_forces.fill(0.0)
        self.container.rigid_body_torques.fill(0.0)

    @ti.kernel
    def compute_pressure_acceleration(self):
        self.container.particle_accelerations.fill(0.0)
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_is_dynamic[p_i]:
                self.container.particle_accelerations[p_i] = ti.Vector([0.0 for _ in range(self.container.dim)])
                if self.container.particle_materials[p_i] == self.container.material_fluid:
                    ret_i = ti.Vector([0.0 for _ in range(self.container.dim)])
                    self.container.for_all_neighbors(p_i, self.compute_pressure_acceleration_task, ret_i)
                    self.container.particle_accelerations[p_i] = ret_i

    @ti.func
    def compute_pressure_acceleration_task(self, p_i, p_j, ret: ti.template()):
        # compute pressure acceleration from the gradient of pressure
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        den_i = self.container.particle_densities[p_i]
        R = pos_i - pos_j
        nabla_ij = self.kernel.gradient(R, self.container.dh)

        if self.container.particle_materials[p_j] == self.container.material_fluid:
            den_j = self.container.particle_densities[p_j]
            pressure_term = (self.container.particle_pressures[p_i] / (den_i * den_i) + self.container.particle_pressures[p_j] / (den_j * den_j))
            ret += -self.container.particle_masses[p_j] * pressure_term * nabla_ij

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            # use fluid particle pressure, density as rigid particle pressure, density
            den_j = self.density_0
            pressure_term = self.container.particle_pressures[p_i] / (den_i * den_i)
            ret += -self.density_0 * self.container.particle_rest_volumes[p_j] * pressure_term * nabla_ij

            if self.container.particle_is_dynamic[p_j]:
                # add force and torque to rigid body
                object_j = self.container.particle_object_ids[p_j]
                center_of_mass_j = self.container.rigid_body_com[object_j]
                force_j = self.density_0 * self.container.particle_rest_volumes[p_j] * pressure_term * nabla_ij * self.density_0 * self.container.particle_rest_volumes[p_i]
                torque_j = ti.math.cross(pos_i - center_of_mass_j, force_j)
                self.container.rigid_body_forces[object_j] += force_j
                self.container.rigid_body_torques[object_j] += torque_j

    def compute_non_pressure_acceleration(self):
        # compute acceleration from gravity, surface tension and viscosity
        self.compute_gravity_acceleration()
        self.compute_surface_tension_acceleration()
        self.compute_viscosity_acceleration()
        
    @ti.kernel
    def compute_gravity_acceleration(self):
        # assign g to all fluid particles, not +=
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_accelerations[p_i] =  ti.Vector(self.g)

    @ti.kernel
    def compute_surface_tension_acceleration(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                a_i = ti.Vector([0.0 for _ in range(self.container.dim)])
                self.container.for_all_neighbors(p_i, self.compute_surface_tension_acceleration_task, a_i)
                self.container.particle_accelerations[p_i] += a_i

    @ti.func
    def compute_surface_tension_acceleration_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Fluid neighbors
            diameter2 = self.container.diameter * self.container.diameter
            pos_j = self.container.particle_positions[p_j]
            R = pos_i - pos_j
            R2 = R.norm_sqr()
            mass_ratio = self.container.particle_masses[p_j] / self.container.particle_masses[p_i]
            weight = 0.0
            if R2 > diameter2:
                weight = self.kernel.weight(R.norm(), self.container.dh)
            else:
                weight = self.kernel.weight(self.container.diameter, self.container.dh)
            ret -= self.surface_tension * mass_ratio * R * weight

    @ti.kernel
    def compute_viscosity_acceleration(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                a_i = ti.Vector([0.0 for _ in range(self.container.dim)])
                self.container.for_all_neighbors(p_i, self.compute_viscosity_acceleration_task, a_i)
                self.container.particle_accelerations[p_i] += (a_i / self.density_0)

    @ti.func
    def compute_viscosity_acceleration_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        nabla_ij = self.kernel.gradient(R, self.container.dh)
        v_xy = ti.math.dot(self.container.particle_velocities[p_i] - self.container.particle_velocities[p_j], R)

        if self.container.particle_materials[p_j] == self.container.material_fluid:
            m_ij = (self.container.particle_masses[p_i] + self.container.particle_masses[p_j]) / 2
            ret += (
                2 * (self.container.dim + 2) * self.viscosity * m_ij 
                / self.container.particle_densities[p_j]
                / (R.norm()**2 + 0.01 * self.container.dh**2)  
                * v_xy 
                * nabla_ij
            )

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            m_ij = (self.density_0 * self.container.particle_rest_volumes[p_j])
            acc = (
                2 * (self.container.dim + 2) * self.viscosity_b * m_ij
                / self.container.particle_densities[p_i]
                / (R.norm()**2 + 0.01 * self.container.dh**2)
                * v_xy
                * nabla_ij
            )

            ret += acc

            if self.container.particle_is_dynamic[p_j]:
                object_j = self.container.particle_object_ids[p_j]
                center_of_mass_j = self.container.rigid_body_com[object_j]
                force_j =  - acc * self.container.particle_masses[p_i] / self.density_0
                self.container.rigid_body_forces[object_j] += force_j
                self.container.rigid_body_torques[object_j] += ti.math.cross(pos_j - center_of_mass_j, force_j)

    @ti.kernel
    def compute_density(self):
        """
        compute density for each particle from mass of neighbors in SPH standard way.
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_densities[p_i] = self.container.particle_rest_volumes[p_i] * self.kernel.weight(0.0, self.container.dh)
                ret_i = 0.0
                self.container.for_all_neighbors(p_i, self.compute_density_task, ret_i)
                self.container.particle_densities[p_i] += ret_i
                self.container.particle_densities[p_i] *= self.density_0

    @ti.func
    def compute_density_task(self, p_i, p_j, ret: ti.template()):
        # Fluid neighbors and rigid neighbors are treated the same
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        R_mod = R.norm()
        ret += self.container.particle_rest_volumes[p_j] * self.kernel.weight(R_mod, self.container.dh)

    @ti.kernel
    def _renew_rigid_particle_state(self):
        # update rigid particle state from rigid body state updated by the rigid solver
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid and self.container.particle_is_dynamic[p_i]:
                object_id = self.container.particle_object_ids[p_i]
                if self.container.rigid_body_is_dynamic[object_id]:
                    center_of_mass = self.container.rigid_body_com[object_id]
                    rotation = self.container.rigid_body_rotations[object_id]
                    velocity = self.container.rigid_body_velocities[object_id]
                    angular_velocity = self.container.rigid_body_angular_velocities[object_id]
                    q = self.container.rigid_particle_original_positions[p_i] - self.container.rigid_body_original_com[object_id]
                    p = rotation @ q
                    self.container.particle_positions[p_i] = center_of_mass + p
                    self.container.particle_velocities[p_i] = velocity + ti.math.cross(angular_velocity, p)

    def renew_rigid_particle_state(self):
        self._renew_rigid_particle_state()

        self.container.object_num[None] = self.container.fluid_object_num[None] + self.container.rigid_object_num[None] + (1 if self.container.add_boundary else 0)
        
        if self.cfg.get_cfg("exportObj"):
            for obj_i in range(self.container.object_num[None]):
                if self.container.rigid_body_is_dynamic[obj_i] and self.container.object_materials[obj_i] == self.container.material_rigid:
                    center_of_mass = self.container.rigid_body_com[obj_i]
                    rotation = self.container.rigid_body_rotations[obj_i]
                    ret = rotation.to_numpy() @ (self.container.object_collection[obj_i]["restPosition"] - self.container.object_collection[obj_i]["restCenterOfMass"]).T
                    self.container.object_collection[obj_i]["mesh"].vertices = ret.T + center_of_mass.to_numpy()

    @ti.kernel
    def update_fluid_velocity(self):
        """
        update velocity for each particle from acceleration
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_velocities[p_i] += self.dt[None] * self.container.particle_accelerations[p_i]

    @ti.kernel
    def update_fluid_position(self):
        """
        update position for each particle from velocity
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_positions[p_i] += self.dt[None] * self.container.particle_velocities[p_i]

            elif self.container.particle_positions[p_i][1] > self.g_upper:
                # the emitter part
                obj_id = self.container.particle_object_ids[p_i]
                if self.container.object_materials[obj_id] == self.container.material_fluid:
                    self.container.particle_positions[p_i] += self.dt[None] * self.container.particle_velocities[p_i]
                    if self.container.particle_positions[p_i][1] <= self.g_upper:
                        self.container.particle_materials[p_i] = self.container.material_fluid
        
            
    @ti.kernel
    def prepare_emitter(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                if self.container.particle_positions[p_i][1] > self.g_upper:
                    # an awful hack to realize emitter
                    # not elegant but works
                    # feel free to implement your own emitter
                    self.container.particle_materials[p_i] = self.container.material_rigid

    @ti.kernel
    def init_object_id(self):
        self.container.particle_object_ids.fill(-1)

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

    def step(self):
        self._step()
        self.container.total_time += self.dt[None]
        self.rigid_solver.total_time += self.dt[None]
        self.compute_rigid_particle_volume()
