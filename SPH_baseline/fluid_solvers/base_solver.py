import numpy as np
from ..containers import BaseContainerBaseline
from .utils import *
from .boundary import BoundaryBaseline
from ..rigid_solver import RigidSolverBaseline

class BaseSolverBaseline():
    def __init__(self, container: BaseContainerBaseline):
        self.container = container
        self.cfg = container.cfg

        # Gravity
        self.g = np.array(self.cfg.get_cfg("gravitation"))

        # density
        self.density = 1000.0
        self.density_0 = self.cfg.get_cfg("density0")

        # surface tension
        if self.container.cfg.get_cfg("surface_tension"):
            self.surface_tension = self.container.cfg.get_cfg("surface_tension")
        else:
            self.surface_tension = 0.01

        # viscosity
        self.viscosity = self.cfg.get_cfg("viscosity")
        if self.cfg.get_cfg("viscosity_b"):
            self.viscosity_b = self.cfg.get_cfg("viscosity_b")
        else:
            self.viscosity_b = self.viscosity

        # time step
        self.dt = self.container.cfg.get_cfg("timeStepSize")

        # kernel
        self.kernel = CubicKernel()

        # boundary
        self.boundary = BoundaryBaseline(self.container)

        # others
        if self.container.cfg.get_cfg("gravitationUpper"):
            self.g_upper = self.container.cfg.get_cfg("gravitationUpper")
        else:
            self.g_upper = 10000.0

        # rigid solver
        self.rigid_solver = RigidSolverBaseline(self.container, gravity=self.g, dt=self.dt)

        # boundary
        self.boundary = BoundaryBaseline(self.container)

    def compute_rigid_particle_volume(self):
        """
        Calculate the volume of rigid particles based on the method described in 
        the paper "Versatile Rigid-Fluid Coupling for Incompressible SPH".
        """
        for p_i in range(self.container.particle_num):
            if self.container.particle_materials[p_i] == self.container.material_rigid:
                if self.container.particle_positions[p_i][1] <= self.g_upper:
                    # Initialize the volume with the kernel weight at zero distance
                    volume = self.kernel.weight(0.0, self.container.dh)
                    
                    def volume_task(p_j, ret):
                        nonlocal volume
                        pos_i = self.container.particle_positions[p_i]
                        if self.container.particle_object_ids[p_j] == self.container.particle_object_ids[p_i]:
                            pos_j = self.container.particle_positions[p_j]
                            distance = np.linalg.norm(pos_i - pos_j)
                            volume += self.kernel.weight(distance, self.container.dh)
                    
                    # Iterate over all neighbors to compute the volume
                    self.container.for_all_neighbors(p_i, volume_task)
                    
                    # Compute the rest volume and mass of the particle
                    self.container.particle_rest_volumes[p_i] = 1.0 / volume
                    self.container.particle_masses[p_i] = self.density_0 * self.container.particle_rest_volumes[p_i]

    def compute_non_pressure_acceleration(self):
        # computing acceleration from gravity, surface tension and viscosity
        self.compute_gravity_acceleration()
        self.compute_surface_tension_acceleration()
        self.compute_viscosity_acceleration()
        
    def compute_gravity_acceleration(self):
        # assign g to all fluid particles, not +=
        for p_i in range(self.container.particle_num):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_accelerations[p_i] = self.g

    def compute_surface_tension_acceleration(self):
        """
        Compute the surface tension acceleration for each fluid particle.
        """
        for p_i in range(self.container.particle_num):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                # Initialize acceleration vector
                a_i = np.zeros(self.container.dim)
                
                def tension_task(p_j, ret):
                    nonlocal a_i
                    pos_i = self.container.particle_positions[p_i]
                    if self.container.particle_materials[p_j] == self.container.material_fluid:
                        pos_j = self.container.particle_positions[p_j]
                        R = pos_i - pos_j
                        R2 = np.sum(R * R)
                        mass_ratio = self.container.particle_masses[p_j] / self.container.particle_masses[p_i]
                        weight = self.kernel.weight(np.sqrt(R2), self.container.dh) if R2 > self.container.diameter * self.container.diameter else self.kernel.weight(self.container.diameter, self.container.dh)
                        a_i -= self.surface_tension * mass_ratio * R * weight
                
                # Iterate over all neighbors to compute the surface tension acceleration
                self.container.for_all_neighbors(p_i, tension_task)
                
                # Add computed acceleration to the particle's acceleration
                self.container.particle_accelerations[p_i] += a_i

    def compute_viscosity_acceleration(self):
        """
        Compute the viscosity acceleration for each fluid particle.
        """
        for p_i in range(self.container.particle_num):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                # Initialize acceleration vector
                a_i = np.zeros(self.container.dim)
                
                def viscosity_task(p_j, ret):
                    nonlocal a_i
                    pos_i = self.container.particle_positions[p_i]
                    pos_j = self.container.particle_positions[p_j]
                    R = pos_i - pos_j
                    nabla_ij = self.kernel.gradient(R, self.container.dh)
                    v_xy = np.dot(self.container.particle_velocities[p_i] - self.container.particle_velocities[p_j], R)

                    if self.container.particle_materials[p_j] == self.container.material_fluid:
                        # Compute the viscosity acceleration for fluid neighbors
                        a_i += (
                            2 * (self.container.dim + 2) * self.viscosity * (self.container.particle_masses[p_i] + self.container.particle_masses[p_j]) / 2
                            / self.container.particle_densities[p_j]
                            / (np.linalg.norm(R)**2 + 0.01 * self.container.dh**2)
                            * v_xy
                            * nabla_ij
                        )

                    elif self.container.particle_materials[p_j] == self.container.material_rigid:
                        # Compute the viscosity acceleration for rigid neighbors
                        acc = (
                            2 * (self.container.dim + 2) * self.viscosity_b * (self.density_0 * self.container.particle_rest_volumes[p_j])
                            / self.container.particle_densities[p_i]
                            / (np.linalg.norm(R)**2 + 0.01 * self.container.dh**2)
                            * v_xy
                            * nabla_ij
                        )

                        a_i += acc

                        if self.container.particle_is_dynamic[p_j]:
                            # If the rigid particle is dynamic, compute the force and torque
                            object_j = self.container.particle_object_ids[p_j]
                            self.container.rigid_body_forces[object_j] += -acc * self.container.particle_masses[p_i] / self.density_0
                            self.container.rigid_body_torques[object_j] += np.cross(pos_j - self.container.rigid_body_com[object_j], 
                                                                                  -acc * self.container.particle_masses[p_i] / self.density_0)
                
                # Iterate over all neighbors to compute the viscosity acceleration
                self.container.for_all_neighbors(p_i, viscosity_task)
                
                # Add computed acceleration to the particle's acceleration
                self.container.particle_accelerations[p_i] += (a_i / self.container.particle_rest_densities[p_i])

    def compute_density(self):
        """
        Compute density for each particle from the mass of neighbors in the standard SPH way.
        """
        for p_i in range(self.container.particle_num):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                # Initialize density with the self-contribution
                density = self.container.particle_masses[p_i] * self.kernel.weight(0.0, self.container.dh)
                
                def density_task(p_j, ret):
                    nonlocal density
                    pos_i = self.container.particle_positions[p_i]
                    pos_j = self.container.particle_positions[p_j]
                    R_mod = np.linalg.norm(pos_i - pos_j)
                    density += self.container.particle_masses[p_j] * self.kernel.weight(R_mod, self.container.dh)
                
                # Iterate over all neighbors to compute the density
                self.container.for_all_neighbors(p_i, density_task)
                
                # Scale the density by the reference density
                self.container.particle_densities[p_i] = density

    def renew_rigid_particle_state(self):
        """
        Update rigid particle state from rigid body state updated by the rigid solver.
        """
        for p_i in range(self.container.particle_num):
            if self.container.particle_materials[p_i] == self.container.material_rigid and self.container.particle_is_dynamic[p_i]:
                object_id = self.container.particle_object_ids[p_i]
                if self.container.rigid_body_is_dynamic[object_id]:
                    # Calculate the rotated position
                    p = self.container.rigid_body_rotations[object_id] @ (
                        self.container.rigid_particle_original_positions[p_i] - 
                        self.container.rigid_body_original_com[object_id]
                    )
                    # Update particle position
                    self.container.particle_positions[p_i] = self.container.rigid_body_com[object_id] + p
                    # Update particle velocity
                    self.container.particle_velocities[p_i] = (
                        self.container.rigid_body_velocities[object_id] + 
                        np.cross(self.container.rigid_body_angular_velocities[object_id], p)
                    )
    
        if self.cfg.get_cfg("exportObj"):
            for obj_i in range(self.container.object_num):
                if self.container.rigid_body_is_dynamic[obj_i] and self.container.object_materials[obj_i] == self.container.material_rigid:
                    # Update the mesh vertices based on the new rigid body state
                    self.container.object_collection[obj_i]["mesh"].vertices = (
                        self.container.rigid_body_rotations[obj_i] @ 
                        (self.container.object_collection[obj_i]["restPosition"] - 
                         self.container.object_collection[obj_i]["restCenterOfMass"]).T
                    ).T + self.container.rigid_body_com[obj_i]

    def update_fluid_velocity(self):
        """
        Update velocity for each fluid particle based on its acceleration.
        """
        for p_i in range(self.container.particle_num):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                # Update velocity using the acceleration and time step
                self.container.particle_velocities[p_i] += self.dt * self.container.particle_accelerations[p_i]

    def update_fluid_position(self):
        """
        Update position for each fluid particle based on its velocity.
        """
        for p_i in range(self.container.particle_num):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                # Update position using the velocity and time step
                self.container.particle_positions[p_i] += self.dt * self.container.particle_velocities[p_i]
            elif self.container.particle_positions[p_i][1] > self.g_upper:
                # Handle the emitter part
                obj_id = self.container.particle_object_ids[p_i]
                if self.container.object_materials[obj_id] == self.container.material_fluid:
                    self.container.particle_positions[p_i] += self.dt * self.container.particle_velocities[p_i]
                    if self.container.particle_positions[p_i][1] <= self.g_upper:
                        self.container.particle_materials[p_i] = self.container.material_fluid

    def prepare_emitter(self):
        for p_i in range(self.container.particle_num):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                if self.container.particle_positions[p_i][1] > self.g_upper:
                    self.container.particle_materials[p_i] = self.container.material_rigid

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
        self.container.object_num = (
            self.container.fluid_object_num + 
            self.container.rigid_object_num + 
            (1 if self.container.add_boundary else 0)
        )
        print(f"Total object num: {self.container.object_num}")
    
    def step(self):
        self._step()
        self.container.total_time += self.dt
        self.rigid_solver.total_time += self.dt
        self.compute_rigid_particle_volume()
