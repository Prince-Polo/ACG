import taichi as ti
from .kernel_functions import KernelFunctions

@ti.data_oriented
class AccelerationUtils:
    def __init__(self, container):
        self.container = container
        self.kernel_functions = KernelFunctions(container.dh, container.dim)

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
                    self.container.for_all_neighbors(p_i, lambda p_i, p_j, ret=ret_i: self.compute_pressure_acceleration_task(p_i, p_j, ret, self.container.density_0))
                    self.container.particle_accelerations[p_i] = ret_i

    @ti.func
    def compute_pressure_acceleration_task(self, p_i, p_j, ret: ti.template(), density_0):
        # compute pressure acceleration from the gradient of pressure
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        den_i = self.container.particle_densities[p_i]
        R = pos_i - pos_j
        nabla_ij = self.kernel_functions.kernel_gradient(R)

        if self.container.particle_materials[p_j] == self.container.material_fluid:
            den_j = self.container.particle_densities[p_j]

            ret += (
                - self.container.particle_masses[p_j] 
                * (self.container.particle_pressures[p_i] / (den_i * den_i) + self.container.particle_pressures[p_j] / (den_j * den_j)) 
                * nabla_ij
            )

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            # use fluid particle pressure, density as rigid particle pressure, density
            acc = (
                - density_0 * self.container.particle_rest_volumes[p_j] 
                * self.container.particle_pressures[p_i] / (den_i * den_i)
                * nabla_ij
            )
            ret += acc

            if self.container.particle_is_dynamic[p_j]:
                # add force and torque to rigid body
                object_j = self.container.particle_object_ids[p_j]
                center_of_mass_j = self.container.rigid_body_centers_of_mass[object_j]
                force_j = (
                    density_0 * self.container.particle_rest_volumes[p_j] 
                    * self.container.particle_pressures[p_i] / (den_i * den_i)
                    * nabla_ij
                    * (density_0 * self.container.particle_rest_volumes[p_i])
                )

                torque_j = ti.math.cross(pos_i - center_of_mass_j, force_j)
                self.container.rigid_body_forces[object_j] += force_j
                self.container.rigid_body_torques[object_j] += torque_j

    def compute_non_pressure_acceleration(self):
        # compute acceleration from gravity, surface tension and viscosity
        self.compute_gravity_acceleration()
        self.compute_surface_tension_acceleration()

        if self.container.viscosity_method == "standard":
            self.compute_viscosity_acceleration_standard()
        elif self.container.viscosity_method == "implicit":
            self.container.implicit_viscosity_solve()
        else:
            raise NotImplementedError(f"viscosity method {self.container.viscosity_method} not implemented")

    @ti.kernel
    def compute_gravity_acceleration(self):
        # assign g to all fluid particles, not +=
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_accelerations[p_i] =  ti.Vector(self.container.g)

    @ti.kernel
    def compute_surface_tension_acceleration(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                a_i = ti.Vector([0.0 for _ in range(self.container.dim)])
                self.container.for_all_neighbors(p_i, lambda p_i, p_j, ret=a_i: self.compute_surface_tension_acceleration_task(p_i, p_j, ret))
                self.container.particle_accelerations[p_i] += a_i

    @ti.func
    def compute_surface_tension_acceleration_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            # Fluid neighbors
            diameter2 = self.container.particle_diameter * self.container.particle_diameter
            pos_j = self.container.particle_positions[p_j]
            R = pos_i - pos_j
            R2 = ti.math.dot(R, R)
            if R2 > diameter2:
                ret -= self.container.surface_tension / self.container.particle_masses[p_i] * self.container.particle_masses[p_j] * R * self.kernel_functions.kernel_W(R.norm())
            else:
                ret -= self.container.surface_tension / self.container.particle_masses[p_i] * self.container.particle_masses[p_j] * R * self.kernel_functions.kernel_W(ti.Vector([self.container.particle_diameter, 0.0, 0.0]).norm())

    @ti.kernel
    def compute_viscosity_acceleration_standard(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                a_i = ti.Vector([0.0 for _ in range(self.container.dim)])
                self.container.for_all_neighbors(p_i, lambda p_i, p_j, ret=a_i: self.compute_viscosity_acceleration_standard_task(p_i, p_j, ret))
                self.container.particle_accelerations[p_i] += (a_i / self.container.density_0)

    @ti.func
    def compute_viscosity_acceleration_standard_task(self, p_i, p_j, ret: ti.template()):
        # we leave / self.container.density_0 to the caller. so we should do this when we compute rigid torque and force
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        # Compute the viscosity force contribution
        R = pos_i - pos_j
        nabla_ij = self.kernel_functions.kernel_gradient(R)
        v_xy = ti.math.dot(self.container.particle_velocities[p_i] - self.container.particle_velocities[p_j], R)
        
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            m_ij = (self.container.particle_masses[p_i] + self.container.particle_masses[p_j]) / 2
            acc = (
                2 * (self.container.dim + 2) * self.container.viscosity * m_ij 
                / self.container.particle_densities[p_j]
                / (R.norm()**2 + 0.01 * self.container.dh**2)  
                * v_xy 
                * nabla_ij
            )
            ret += acc

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            m_ij = (self.container.density_0 * self.container.particle_rest_volumes[p_j])
            acc = (
                2 * (self.container.dim + 2) * self.container.viscosity_b * m_ij
                / self.container.particle_densities[p_i]
                / (R.norm()**2 + 0.01 * self.container.dh**2)
                * v_xy
                * nabla_ij
            )

            ret += acc

            if self.container.particle_is_dynamic[p_j]:
                object_j = self.container.particle_object_ids[p_j]
                center_of_mass_j = self.container.rigid_body_centers_of_mass[object_j]
                force_j =  - acc * self.container.particle_masses[p_i] / self.container.density_0
                torque_j = ti.math.cross(pos_j - center_of_mass_j, force_j)
                self.container.rigid_body_forces[object_j] += force_j
                self.container.rigid_body_torques[object_j] += torque_j