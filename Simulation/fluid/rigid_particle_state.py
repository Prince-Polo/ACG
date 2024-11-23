import taichi as ti

@ti.data_oriented
class RigidParticleState:
    def __init__(self, container):
        self.container = container

    @ti.kernel
    def _renew_rigid_particle_state(self):
        # update rigid particle state from rigid body state updated by the rigid solver
        for p_i in range(self.container.particle_num[None]):
            if self.container.particle_materials[p_i] == self.container.material_rigid and self.container.particle_is_dynamic[p_i]:
                object_id = self.container.particle_object_ids[p_i]
                if self.container.rigid_body_is_dynamic[object_id]:
                    center_of_mass = self.container.rigid_body_centers_of_mass[object_id]
                    rotation = self.container.rigid_body_rotations[object_id]
                    velocity = self.container.rigid_body_velocities[object_id]
                    angular_velocity = self.container.rigid_body_angular_velocities[object_id]
                    q = self.container.rigid_particle_original_positions[p_i] - self.container.rigid_body_original_centers_of_mass[object_id]
                    p = rotation @ q
                    self.container.particle_positions[p_i] = center_of_mass + p
                    self.container.particle_velocities[p_i] = velocity + ti.math.cross(angular_velocity, p)

    def renew_rigid_particle_state(self):
        self._renew_rigid_particle_state()
        
        if self.container.cfg.get_cfg("exportObj"):
            for obj_i in range(self.container.object_num[None]):
                if self.container.rigid_body_is_dynamic[obj_i] and self.container.object_materials[obj_i] == self.container.material_rigid:
                    center_of_mass = self.container.rigid_body_centers_of_mass[obj_i]
                    rotation = self.container.rigid_body_rotations[obj_i]
                    ret = rotation.to_numpy() @ (self.container.object_collection[obj_i]["restPosition"] - self.container.object_collection[obj_i]["restCenterOfMass"]).T
                    self.container.object_collection[obj_i]["mesh"].vertices = ret.T + center_of_mass.to_numpy()