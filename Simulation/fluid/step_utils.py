import taichi as ti

@ti.data_oriented
class StepUtils:
    def __init__(self, container):
        self.container = container

    def step(self):
        self._step()
        self.container.total_time += self.container.dt[None]
        self.container.rigid_solver.total_time += self.container.dt[None]
        self.container.compute_rigid_particle_volume()