# implicit viscosity solver implemented as paper "A Physically Consistent Implicit Viscosity Solver for SPH Fluids"
import taichi as ti
import numpy as np
from ..containers import BaseContainer
from ..rigid_solver import RigidSolver

@ti.data_oriented
class BaseSolver():
    def __init__(self, container: BaseContainer):
        self.container = container
        self.cfg = container.cfg
        self.g = ti.Vector([0.0, -9.81, 0.0])  # Gravity
        if self.container.dim == 2:
            self.g = ti.Vector([0.0, -9.81])
        
        self.g = np.array(self.container.cfg.get_cfg("gravitation"))

        # this is used to realize emitter. If a fluid particle is above this height,
        # we shall make it a rigid particle and fix its speed.
        # it's an awful hack, but it works. feel free to change it.
        self.g_upper = self.container.cfg.get_cfg("gravitationUpper")
        if self.g_upper == None:
            self.g_upper = 10000.0 # a large number

        self.viscosity_method = self.container.cfg.get_cfg("viscosityMethod")
        self.viscosity = self.container.cfg.get_cfg("viscosity")
        self.viscosity_b = self.container.cfg.get_cfg("viscosity_b")
        if self.viscosity_b == None:
            self.viscosity_b = self.viscosity
        self.density_0 = 1000.0  
        self.density_0 = self.container.cfg.get_cfg("density0")
        self.surface_tension = 0.01

        self.dt = ti.field(float, shape=())
        self.dt[None] = 1e-4
        self.dt[None] = self.container.cfg.get_cfg("timeStepSize")

        self.rigid_solver = RigidSolver(container, gravity=self.g,  dt=self.dt[None])

        if self.viscosity_method == "implicit":
            # initialize things needed for conjugate gradient solver
            # conjugate gradient solver implemented following https://en.wikipedia.org/wiki/Conjugate_gradient_method
            self.cg_p = ti.Vector.field(self.container.dim, dtype=ti.f32, shape=self.container.particle_max_num)
            self.original_velocity = ti.Vector.field(self.container.dim, dtype=ti.f32, shape=self.container.particle_max_num)
            self.cg_Ap = ti.Vector.field(self.container.dim, dtype=ti.f32, shape=self.container.particle_max_num)
            self.cg_x = ti.Vector.field(self.container.dim, dtype=ti.f32, shape=self.container.particle_max_num)
            self.cg_b = ti.Vector.field(self.container.dim, dtype=ti.f32, shape=self.container.particle_max_num)
            self.cg_alpha = ti.field(dtype=ti.f32, shape=())
            self.cg_beta = ti.field(dtype=ti.f32, shape=())
            self.cg_r = ti.Vector.field(self.container.dim, dtype=ti.f32, shape=self.container.particle_max_num)
            self.cg_error = ti.field(dtype=ti.f32, shape=())
            self.cg_diagnol_ii_inv = ti.Matrix.field(self.container.dim, self.container.dim, dtype=ti.f32, shape=self.container.particle_max_num)

            self.cg_tol = 1e-6