import taichi as ti

@ti.data_oriented
class InitializationUtils:
    def __init__(self, container):
        self.container = container

    @ti.kernel
    def init_object_id(self):
        self.container.particle_object_ids.fill(-1)

    def prepare(self):
        self.init_object_id()
        self.container.insert_object()
        self.container.prepare_emitter()
        self.container.rigid_solver.insert_rigid_object()
        self.container.renew_rigid_particle_state()
        self.container.prepare_neighborhood_search()
        self.container.compute_rigid_particle_volume()