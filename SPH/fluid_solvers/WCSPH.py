import taichi as ti
from .base_solver import BaseSolver
from ..containers.base_container import BaseContainer
from ..utils.kernel import *

cnt = 0

@ti.data_oriented
class WCSPHSolver(BaseSolver):
    @ti.kernel
    def compute_pressure(self):
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] != self.container.material_rigid:
                rho_i = self.container.particle_densities[p_i]
                if rho_i < self.density_0:
                    rho_i = self.density_0
                
                self.container.particle_densities[p_i] = rho_i
                rho_ratio = rho_i / self.density_0
                pressure_term = ti.pow(rho_ratio, self.container.gamma) - 1.0
                self.container.particle_pressures[p_i] = self.container.stiffness * pressure_term
    
    @ti.kernel
    def compute_pressure_acceleration(self):
        cnt2 = 0
        for i in range(self.container.particle_num[None]):
            ret_i = ti.Vector([0.0, 0.0, 0.0])
            if self.container.particle_materials[i]!= self.container.material_rigid:
                self.container.for_all_neighbors(i, self.compute_pressure_acceleration_task, ret_i)
                if cnt == 1 and cnt2 == 0:
                    print("ret_i", ret_i)
                self.container.particle_accelerations[i] = ret_i
                cnt2 += 1

    @ti.func
    def compute_pressure_acceleration_task(self, p_i, p_j, ret: ti.template()):
        pos_i = self.container.particle_positions[p_i]
        pos_j = self.container.particle_positions[p_j]
        den_i = self.container.particle_densities[p_i]
        nabla_ij = self.kernel.gradient(pos_i - pos_j, self.container.dh)

        if self.container.particle_materials[p_j] != self.container.material_rigid:
            ret += self._compute_fluid_pressure_acc(p_i, p_j, den_i, self.container.particle_densities[p_j], nabla_ij)
        else:
            ret += self._compute_rigid_pressure_acc(p_i, p_j, den_i, nabla_ij)

    @ti.kernel
    def update_rigid_body_force(self):
        cnt2 = 0
        for i in range(self.container.particle_num[None]):
            if cnt == 1 and cnt2 == 0:
                print("更新刚体力")
            if self.container.particle_materials[i] != self.container.material_rigid:
                self.container.for_all_neighbors(i, self.update_rigid_body_task, None)
            cnt2 += 1

    @ti.func
    def update_rigid_body_task(self, p_i, p_j, ret: ti.template()):
        if (self.container.particle_materials[p_j] != self.container.material_rigid and self.container.particle_is_dynamic[p_j]):
            pos_i = self.container.particle_positions[p_i]
            pos_j = self.container.particle_positions[p_j]
            den_i = self.container.particle_densities[p_i]
            nabla_ij = self.kernel.gradient(pos_i - pos_j, self.container.dh)
            
            object_j = self.container.particle_object_ids[p_j]
            center_of_mass_j = self.container.rigid_body_com[object_j]
            regular_pressure_i = self.container.particle_pressures[p_i] / den_i
            
            force_j = self.container.particle_masses[p_i] * regular_pressure_i * nabla_ij * self.container.particle_masses[p_i] / den_i
            self.container.rigid_body_forces[object_j] += force_j
            
            torque_j = ti.math.cross(pos_i - center_of_mass_j, force_j)
            self.container.rigid_body_torques[object_j] += torque_j

    @ti.func
    def _compute_fluid_pressure_acc(self, p_i, p_j, den_i, den_j, nabla_ij):
        regular_pressure_i = self.container.particle_pressures[p_i] / den_i
        regular_pressure_j = self.container.particle_pressures[p_j] / den_j
        pressure_term = regular_pressure_i / den_i + regular_pressure_j / den_j
        return -self.container.particle_masses[p_j] * pressure_term * nabla_ij

    @ti.func
    def _compute_rigid_pressure_acc(self, p_i, p_j, den_i, nabla_ij):
        regular_pressure_i = self.container.particle_pressures[p_i] / den_i
        return (-self.container.particle_masses[p_j] * regular_pressure_i * nabla_ij/ den_i)

    def _step(self):
        global cnt
        cnt += 1
        if cnt % 100 == 1:
            print("邻居搜索")
        self.container.prepare_neighbor_search()
        if cnt % 100 == 1:
            print("计算密度")
        self.compute_density()
        if cnt % 100 == 1:
            print("计算非压力加速度")
        self.compute_non_pressure_acceleration()
        if cnt % 100 == 1:
            print("更新速度")
        self.update_fluid_velocity()

        if cnt % 100 == 1:
            print("计算压力")
        self.compute_pressure()

        if cnt % 100 == 1:
            print("计算压力加速度")
        self.compute_pressure_acceleration()
        if cnt % 100 == 1:
            print("更新位置")
        self.update_rigid_body_force()
        if cnt % 100 == 1:
            print("更新刚体力")
        self.update_fluid_velocity()
        if cnt % 100 == 1:
            print("更新刚体位置")
        self.update_fluid_position()

        if cnt % 100 == 1:
            print("更新刚体位置")
        self.rigid_solver.step()
        
        if cnt % 100 == 1:
            print("更新刚体位置")
        self.container.insert_object()
        if cnt % 100 == 1:
            print("更新刚体位置")
        self.rigid_solver.insert_rigid_object()
        if cnt % 100 == 1:
            print("更新刚体位置")
        self.renew_rigid_particle_state()
    
        if cnt % 100 == 1:
            print("更新刚体位置")
        self.boundary.enforce_domain_boundary(self.container.material_fluid)
