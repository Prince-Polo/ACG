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
        """
        Calculate the volume of rigid particles based on the method described in 
        the paper "Versatile Rigid-Fluid Coupling for Incompressible SPH".
        """
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
        """
        Calculate the contribution to the volume of a rigid particle from its neighbors.
        """
        pos_i = self.container.particle_positions[p_i]
        if self.container.particle_object_ids[p_j] == self.container.particle_object_ids[p_i]:
            pos_j = self.container.particle_positions[p_j]
            distance = (pos_i - pos_j).norm()
            # Accumulate the kernel weight based on the distance between particles
            volume += self.kernel.weight(distance, self.container.dh)

    @ti.kernel
    def init_acceleration(self):
        self.container.particle_accelerations.fill(0.0)

    @ti.kernel
    def init_rigid_body_force_and_torque(self):
        self.container.rigid_body_forces.fill(0.0)
        self.container.rigid_body_torques.fill(0.0)

    @ti.kernel
    def compute_pressure_acceleration(self):
        """
        Compute the pressure acceleration for each particle.
        """
        self.container.particle_accelerations.fill(0.0)
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_is_dynamic[p_i]:
                if self.container.particle_materials[p_i] == self.container.material_fluid:
                    # Initialize acceleration vector
                    acceleration = ti.Vector([0.0 for _ in range(self.container.dim)])
                    # Iterate over all neighbors to compute the pressure acceleration
                    self.container.for_all_neighbors(p_i, self.compute_pressure_acceleration_task, acceleration)
                    # Assign computed acceleration to the particle
                    self.container.particle_accelerations[p_i] = acceleration

    @ti.func
    def compute_pressure_acceleration_task(self, p_i, p_j, acceleration: ti.template()):
        """
        Compute the pressure acceleration contribution from neighboring particles.
        """
        # Get positions of particles i and j
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        # Compute the distance vector between particles i and j
        R = pos_i - pos_j
        # Compute the gradient of the kernel function
        nabla_ij = self.kernel.gradient(R, self.container.dh)

        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Update the acceleration with the pressure contribution from particle j
            acceleration += -self.container.particle_masses[p_j] * (
                (self.container.particle_pressures[p_i] / (self.container.particle_densities[p_i] ** 2) +
                self.container.particle_pressures[p_j] / (self.container.particle_densities[p_j] ** 2)) * nabla_ij)

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            # Update the acceleration with the pressure contribution from particle j
            acceleration += -self.container.particle_masses[p_j] * (
                self.container.particle_pressures[p_i] / (self.container.particle_densities[p_i] ** 2)) * nabla_ij

            if self.container.particle_is_dynamic[p_j]:
                # If the rigid particle is dynamic, compute the force and torque
                object_j = self.container.particle_object_ids[p_j]
                force_j = self.container.particle_masses[p_j] * (
                    self.container.particle_pressures[p_i] / (self.container.particle_densities[p_i] ** 2)) * nabla_ij * self.container.particle_masses[p_i]
                torque_j = ti.math.cross(pos_i - self.container.rigid_body_com[object_j], force_j)
                # Update the force and torque of the rigid body
                self.container.rigid_body_forces[object_j] += force_j
                self.container.rigid_body_torques[object_j] += torque_j

    def compute_non_pressure_acceleration(self):
        # computing acceleration from gravity, surface tension and viscosity
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
        """
        Compute the surface tension acceleration for each fluid particle.
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                # Initialize acceleration vector
                a_i = ti.Vector([0.0 for _ in range(self.container.dim)])
                # Iterate over all neighbors to compute the surface tension acceleration
                self.container.for_all_neighbors(p_i, self.compute_surface_tension_acceleration_task, a_i)
                # Add computed acceleration to the particle's acceleration
                self.container.particle_accelerations[p_i] += a_i

    @ti.func
    def compute_surface_tension_acceleration_task(self, p_i, p_j, a_i: ti.template()):
        """
        Compute the surface tension acceleration contribution from neighboring particles.
        """
        pos_i = self.container.particle_positions[p_i]
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Fluid neighbors
            pos_j = self.container.particle_positions[p_j]
            R = pos_i - pos_j
            R2 = R.norm_sqr()
            # Compute the mass ratio
            mass_ratio = self.container.particle_masses[p_j] / self.container.particle_masses[p_i]
            # Compute the weight based on the distance between particles
            weight = self.kernel.weight(R.norm(), self.container.dh) if R2 > self.container.diameter * self.container.diameter else self.kernel.weight(self.container.diameter, self.container.dh)
            # Update the acceleration with the surface tension contribution from particle j
            a_i -= self.surface_tension * mass_ratio * R * weight

    @ti.kernel
    def compute_viscosity_acceleration(self):
        """
        Compute the viscosity acceleration for each fluid particle.
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                # Initialize acceleration vector
                a_i = ti.Vector([0.0 for _ in range(self.container.dim)])
                # Iterate over all neighbors to compute the viscosity acceleration
                self.container.for_all_neighbors(p_i, self.compute_viscosity_acceleration_task, a_i)
                # Add computed acceleration to the particle's acceleration
                self.container.particle_accelerations[p_i] += (a_i / self.container.particle_rest_densities[p_i])

    @ti.func
    def compute_viscosity_acceleration_task(self, p_i, p_j, a_i: ti.template()):
        """
        Compute the viscosity acceleration contribution from neighboring particles.
        """
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R = pos_i - pos_j
        nabla_ij = self.kernel.gradient(R, self.container.dh)
        v_xy = ti.math.dot(self.container.particle_velocities[p_i] - self.container.particle_velocities[p_j], R)

        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Compute the viscosity acceleration for fluid neighbors
            a_i += (
                2 * (self.container.dim + 2) * self.viscosity * (self.container.particle_masses[p_i] + self.container.particle_masses[p_j]) / 2
                / self.container.particle_densities[p_j]
                / (R.norm()**2 + 0.01 * self.container.dh**2)
                * v_xy
                * nabla_ij
            )

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            # Compute the viscosity acceleration for rigid neighbors
            acc = (
                2 * (self.container.dim + 2) * self.viscosity_b * (self.density_0 * self.container.particle_rest_volumes[p_j])
                / self.container.particle_densities[p_i]
                / (R.norm()**2 + 0.01 * self.container.dh**2)
                * v_xy
                * nabla_ij
            )

            a_i += acc

            if self.container.particle_is_dynamic[p_j]:
                # If the rigid particle is dynamic, compute the force and torque
                object_j = self.container.particle_object_ids[p_j]
                self.container.rigid_body_forces[object_j] += -acc * self.container.particle_masses[p_i] / self.density_0
                self.container.rigid_body_torques[object_j] += ti.math.cross(pos_j - self.container.rigid_body_com[object_j], -acc * self.container.particle_masses[p_i] / self.density_0)

    @ti.kernel
    def compute_density(self):
        """
        Compute density for each particle from the mass of neighbors in the standard SPH way.
        """
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
        """
        Compute the density contribution from neighboring particles.
        """
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        R_mod = (pos_i - pos_j).norm()
        # Accumulate the density contribution from particle j
        density += self.container.particle_masses[p_j] * self.kernel.weight(R_mod, self.container.dh)

    @ti.kernel
    def _renew_rigid_particle_state(self):
        """
        Update rigid particle state from rigid body state updated by the rigid solver.
        """
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
        """
        Renew the state of rigid particles and update the mesh if necessary.
        """
        self._renew_rigid_particle_state()
    
        if self.cfg.get_cfg("exportObj"):
            for obj_i in range(self.container.object_num[None]):
                if self.container.rigid_body_is_dynamic[obj_i] and self.container.object_materials[obj_i] == self.container.material_rigid:
                    # Update the mesh vertices based on the new rigid body state
                    self.container.object_collection[obj_i]["mesh"].vertices = (self.container.rigid_body_rotations[obj_i].to_numpy() @ (self.container.object_collection[obj_i]["restPosition"] - self.container.object_collection[obj_i]["restCenterOfMass"]).T).T + self.container.rigid_body_com[obj_i].to_numpy()

    @ti.kernel
    def update_fluid_velocity(self):
        """
        Update velocity for each fluid particle based on its acceleration.
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                # Update velocity using the acceleration and time step
                self.container.particle_velocities[p_i] += self.dt[None] * self.container.particle_accelerations[p_i]

    @ti.kernel
    def update_fluid_position(self):
        """
        Update position for each fluid particle based on its velocity.
        """
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                # Update position using the velocity and time step
                self.container.particle_positions[p_i] += self.dt[None] * self.container.particle_velocities[p_i]
            elif self.container.particle_positions[p_i][1] > self.g_upper:
                # Handle the emitter part
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
        self.container.object_num[None] = self.container.fluid_object_num[None] + self.container.rigid_object_num[None] + (1 if self.container.add_boundary else 0)
        print(f"Total object num: {self.container.object_num[None]}")
    
    def step(self):
        self._step()
        self.container.total_time += self.dt[None]
        self.rigid_solver.total_time += self.dt[None]
        self.compute_rigid_particle_volume()
