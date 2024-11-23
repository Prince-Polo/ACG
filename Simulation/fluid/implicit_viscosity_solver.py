import taichi as ti

@ti.data_oriented
class ImplicitViscositySolver:
    def __init__(self, container):
        self.container = container
        self.dt = container.dt
        self.density_0 = container.density_0
        self.viscosity = container.viscosity
        self.viscosity_b = container.viscosity_b
        self.cg_tol = 1e-6

        self.cg_p = ti.Vector.field(container.dim, dtype=ti.f32, shape=container.particle_max_num)
        self.original_velocity = ti.Vector.field(container.dim, dtype=ti.f32, shape=container.particle_max_num)
        self.cg_Ap = ti.Vector.field(container.dim, dtype=ti.f32, shape=container.particle_max_num)
        self.cg_x = ti.Vector.field(container.dim, dtype=ti.f32, shape=container.particle_max_num)
        self.cg_b = ti.Vector.field(container.dim, dtype=ti.f32, shape=container.particle_max_num)
        self.cg_alpha = ti.field(dtype=ti.f32, shape=())
        self.cg_beta = ti.field(dtype=ti.f32, shape=())
        self.cg_r = ti.Vector.field(container.dim, dtype=ti.f32, shape=container.particle_max_num)
        self.cg_error = ti.field(dtype=ti.f32, shape=())
        self.cg_diagnol_ii_inv = ti.Matrix.field(container.dim, container.dim, dtype=ti.f32, shape=container.particle_max_num)

    @ti.kernel
    def prepare_conjugate_gradient_solver1(self):
        # initialize conjugate gradient solver phase 1
        self.cg_r.fill(0.0)
        self.cg_p.fill(0.0)
        self.original_velocity.fill(0.0)
        self.cg_b.fill(0.0)
        self.cg_Ap.fill(0.0)

        # initial guess for x. We use v^{df} + v(t) - v^{df}(t - dt) as initial guess. we assume v(t) - v^{df}(t - dt) is already in x
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.cg_x[p_i] += self.container.particle_velocities[p_i]

        # storing the original velocity
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.original_velocity[p_i] = self.container.particle_velocities[p_i]

        # prepare b
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret = ti.Matrix.zero(ti.f32, self.container.dim, self.container.dim)
                self.container.for_all_neighbors(p_i, self.compute_A_ii_task, ret)

                # preconditioner
                diag_ii = ti.Matrix.identity(ti.f32, self.container.dim) - ret * self.dt[None] / self.density_0
                self.cg_diagnol_ii_inv[p_i] = ti.math.inverse(diag_ii)

                ret1 = ti.Vector([0.0 for _ in range(self.container.dim)])
                self.container.for_all_neighbors(p_i, self.compute_b_i_task, ret1)
                self.cg_b[p_i] = self.container.particle_velocities[p_i] - self.dt[None] * ret1 / self.density_0

                # copy x into p to calculate Ax
                self.cg_p[p_i] = self.cg_x[p_i]

    @ti.kernel
    def prepare_conjugate_gradient_solver2(self):
        # initialize conjugate gradient solver phase 2
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.cg_r[p_i] = self.cg_diagnol_ii_inv[p_i] @ self.cg_b[p_i] - self.cg_Ap[p_i]
                self.cg_p[p_i] = self.cg_r[p_i]

    @ti.func
    def compute_A_ii_task(self, p_i, p_j, ret: ti.template()):
        # there is left densities[p_i] to be divided
        # we assume p_i is a fluid particle
        # p_j can either be a fluid particle or a rigid particle
        A_ij = self.compute_A_ij(p_i, p_j)
        ret -= A_ij

    @ti.func
    def compute_b_i_task(self, p_i, p_j, ret: ti.template()):
        # we assume p_i is a fluid particle
        if self.container.particle_materials[p_j] == self.container.material_rigid:
            R = self.container.particle_positions[p_i] - self.container.particle_positions[p_j]
            nabla_ij = self.kernel_gradient(R)
            ret += (
                2 * (self.container.dim + 2) * self.viscosity_b
                * self.density_0 * self.container.particle_rest_volumes[p_j]
                / self.container.particle_densities[p_i]
                * ti.math.dot(self.container.particle_velocities[p_j], R)
                / (R.norm_sqr() + 0.01 * self.container.dh**2)
                * nabla_ij
            )

    @ti.func
    def compute_A_ij(self, p_i, p_j):
        # we do not divide densities[p_i] here.
        # we assume p_i is a fluid particle
        A_ij = ti.Matrix.zero(ti.f32, self.container.dim, self.container.dim)
        R = self.container.particle_positions[p_i] - self.container.particle_positions[p_j]
        nabla_ij = self.kernel_gradient(R)
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            m_ij = (self.container.particle_masses[p_i] + self.container.particle_masses[p_j]) / 2
            A_ij = (- 2 * (self.container.dim + 2) * self.viscosity * m_ij
                    / self.container.particle_densities[p_j]
                    / (R.norm_sqr() + 0.01 * self.container.dh**2) 
                    * nabla_ij.outer_product(R) 
            )

        elif self.container.particle_materials[p_j] == self.container.material_rigid:
            m_ij = (self.density_0 * self.container.particle_rest_volumes[p_j])
            A_ij = (- 2 * (self.container.dim + 2) * self.viscosity_b * m_ij
                    / self.container.particle_densities[p_i]
                    / (R.norm_sqr() + 0.01 * self.container.dh**2) 
                    * nabla_ij.outer_product(R) 
            )

        return A_ij

    @ti.kernel
    def compute_Ap(self):
        # the linear operator
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret = ti.Vector([0.0 for _ in range(self.container.dim)])
                self.container.for_all_neighbors(p_i, self.compute_Ap_task, ret)
                ret *= self.dt[None]
                ret /= self.density_0
                ret += self.cg_p[p_i]
                self.cg_Ap[p_i] = ret    

    @ti.func
    def compute_Ap_task(self, p_i, p_j, ret: ti.template()):
        if self.container.particle_materials[p_j] == self.container.material_fluid:
            A_ij = self.compute_A_ij(p_i, p_j)

            # preconditioner
            ret += self.cg_diagnol_ii_inv[p_i] @ (-A_ij) @ self.cg_p[p_j]

    @ti.kernel
    def compute_cg_alpha(self):
        self.cg_alpha[None] = 0.0
        numerator = 0.0
        denominator = 0.0
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                numerator += self.cg_r[p_i].norm_sqr()
                denominator += self.cg_p[p_i].dot(self.cg_Ap[p_i])

        if denominator > 1e-18:
            self.cg_alpha[None] = numerator / denominator
        else:
            self.cg_alpha[None] = 0.0

    @ti.kernel
    def update_cg_x(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.cg_x[p_i] += self.cg_alpha[None] * self.cg_p[p_i]

    @ti.kernel
    def update_cg_r_and_beta(self):
        self.cg_error[None] = 0.0
        numerator = 0.0
        denominator = 0.0
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                new_r_i = self.cg_r[p_i] - self.cg_alpha[None] * self.cg_Ap[p_i]
                numerator += new_r_i.norm_sqr()
                denominator += self.cg_r[p_i].norm_sqr()
                self.cg_error[None] += new_r_i.norm_sqr()
                self.cg_r[p_i] = new_r_i

        self.cg_error[None] = ti.sqrt(self.cg_error[None])
        if denominator > 1e-18:
            self.cg_beta[None] = numerator / denominator
        else:
            self.cg_beta[None] = 0.0

    @ti.kernel
    def update_p(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.cg_p[p_i] =  self.cg_r[p_i] + self.cg_beta[None] * self.cg_p[p_i]

    @ti.kernel
    def prepare_guess(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.cg_x[p_i] -= self.original_velocity[p_i]

    def conjugate_gradient_loop(self):
        tol = 1000.0
        num_itr = 0

        while tol > self.cg_tol and num_itr < 1000:
            self.compute_Ap()
            self.compute_cg_alpha()
            self.update_cg_x()

            self.update_cg_r_and_beta()
            self.update_p()
            tol = self.cg_error[None]
            num_itr += 1

        print("CG iteration: ", num_itr, " error: ", tol)

    @ti.kernel
    def viscosity_update_velocity(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_velocities[p_i] = self.cg_x[p_i]

    @ti.kernel
    def copy_back_original_velocity(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                self.container.particle_velocities[p_i] = self.original_velocity[p_i]

    @ti.kernel
    def add_viscosity_force_to_rigid(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_fluid:
                ret = ti.Vector([0.0 for _ in range(self.container.dim)]) # dummy variable
                self.container.for_all_neighbors(p_i, self.add_viscosity_force_to_rigid_task, ret)

    @ti.func
    def add_viscosity_force_to_rigid_task(self, p_i, p_j, ret: ti.template()):
        if self.container.particle_materials[p_j] == self.container.material_rigid and self.container.particle_is_dynamic[p_j]:
            pos_i = self.container.particle_positions[p_i]
            pos_j = self.container.particle_positions[p_j]
            # Compute the viscosity force contribution
            R = pos_i - pos_j
            v_ij = self.container.particle_velocities[p_i] - self.container.particle_velocities[p_j]
            nabla_ij = self.kernel_gradient(R)

            acc = (
                2 * (self.container.dim + 2) * self.viscosity_b
                * (self.density_0 * self.container.particle_rest_volumes[p_j])
                / self.container.particle_densities[p_i] /  self.density_0
                * ti.math.dot(v_ij, R)
                / (R.norm_sqr() + 0.01 * self.container.dh**2)
                * nabla_ij
            )

            object_j = self.container.particle_object_ids[p_j]
            center_of_mass_j = self.container.rigid_body_centers_of_mass[object_j]
            force_j =  - acc * self.container.particle_rest_volumes[p_i] * self.density_0
            torque_j = ti.math.cross(pos_j - center_of_mass_j, force_j)
            self.container.rigid_body_forces[object_j] += force_j
            self.container.rigid_body_torques[object_j] += torque_j

    def implicit_viscosity_solve(self):
        self.prepare_conjugate_gradient_solver1()
        self.compute_Ap()
        self.prepare_conjugate_gradient_solver2()
        self.conjugate_gradient_loop()
        self.viscosity_update_velocity()
        self.compute_viscosity_acceleration_standard() # we use this function to update acceleration
        self.copy_back_original_velocity() # copy back original velocity
        self.prepare_guess()