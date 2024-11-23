# implicit viscosity solver implemented as paper "A Physically Consistent Implicit Viscosity Solver for SPH Fluids"
import taichi as ti
import numpy as np
from base import BaseContainer
from rigid import RigidSolver
from .kernel_functions import KernelFunctions
from .rigid_particle_utils import RigidParticleUtils
from .acceleration_utils import AccelerationUtils
from .implicit_viscosity_solver import ImplicitViscositySolver
from .density_utils import DensityUtils
from .boundary_utils import BoundaryUtils
from .rigid_particle_state import RigidParticleState
from .fluid_particle_utils import FluidParticleUtils
from .initialization_utils import InitializationUtils
from .step_utils import StepUtils

@ti.data_oriented
class BaseSolver:
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

        self.kernel_functions = KernelFunctions(container.dh, container.dim)
        self.rigid_particle_utils = RigidParticleUtils(container, self.density_0, self.g_upper)
        self.acceleration_utils = AccelerationUtils(container)
        self.implicit_viscosity_solver = ImplicitViscositySolver(container)
        self.density_utils = DensityUtils(container)
        self.boundary_utils = BoundaryUtils(container)
        self.rigid_particle_state = RigidParticleState(container)
        self.fluid_particle_utils = FluidParticleUtils(container)
        self.initialization_utils = InitializationUtils(container)
        self.step_utils = StepUtils(container)

    def step(self):
        self.step_utils.step()