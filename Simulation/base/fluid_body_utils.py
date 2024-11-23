import taichi as ti
import numpy as np
import trimesh as tm
from tqdm import tqdm
from functools import reduce

@ti.data_oriented
class FluidBodyUtils:
    def __init__(self, particle_diameter, dim):
        self.particle_diameter = particle_diameter
        self.dim = dim

    def load_fluid_body(self, fluid_body, pitch=None):
        if pitch is None:
            pitch = self.particle_diameter
        obj_id = fluid_body["objectId"]
        mesh = tm.load(fluid_body["geometryFile"])
        mesh.apply_scale(fluid_body["scale"])
        offset = np.array(fluid_body["translation"])

        angle = fluid_body["rotationAngle"] / 360 * 2 * 3.1415926
        direction = fluid_body["rotationAxis"]
        rot_matrix = tm.transformations.rotation_matrix(angle, direction, mesh.vertices.mean(axis=0))
        mesh.apply_transform(rot_matrix)
        mesh.vertices += offset

        min_point, max_point = mesh.bounding_box.bounds
        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(min_point[i], max_point[i], pitch))
        
        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        
        print(f"processing {len(new_positions)} points to decide whether they are inside the mesh. This might take a while.")
        inside = [False for _ in range(len(new_positions))]

        # decide whether the points are inside the mesh or not
        # TODO: make it parallel or precompute and store
        pbar = tqdm(total=len(new_positions))
        for i in range(len(new_positions)):
            if mesh.contains([new_positions[i]])[0]:
                inside[i] = True
            pbar.update(1)

        pbar.close()

        new_positions = new_positions[inside]
        return new_positions

    def insert_fluid_objects(self, fluid_blocks, fluid_bodies, present_object, total_time, particle_spacing, add_cube, add_particles, fluid_particle_num, object_visibility, object_materials, material_fluid, object_id_fluid_body, object_collection):
        # Fluid block
        for fluid in fluid_blocks:
            obj_id = fluid["objectId"]

            if obj_id in present_object:
                continue
            if fluid["entryTime"] > total_time:
                continue

            offset = np.array(fluid["translation"])
            start = np.array(fluid["start"]) + offset
            end = np.array(fluid["end"]) + offset
            scale = np.array(fluid["scale"])
            velocity = fluid["velocity"]
            density = fluid["density"]
            color = fluid["color"]
            object_id_fluid_body.add(obj_id)

            if "visible" in fluid:
                object_visibility[obj_id] = fluid["visible"]
            else:
                object_visibility[obj_id] = 1

            object_materials[obj_id] = material_fluid
            object_collection[obj_id] = fluid

            add_cube(object_id=obj_id,
                     lower_corner=start,
                     cube_size=(end-start)*scale,
                     velocity=velocity,
                     density=density, 
                     is_dynamic=1, 
                     color=color,
                     material=material_fluid,
                     space=particle_spacing)
            
            present_object.append(obj_id)

        # Fluid body
        for fluid_body in fluid_bodies:
            obj_id = fluid_body["objectId"]

            if obj_id in present_object:
                continue
            if fluid_body["entryTime"] > total_time:
                continue

            num_particles_obj = fluid_body["particleNum"]
            voxelized_points_np = fluid_body["voxelizedPoints"]
            velocity = np.array(fluid_body["velocity"], dtype=np.float32)
   
            density = fluid_body["density"]
            color = np.array(fluid_body["color"], dtype=np.int32)

            if "visible" in fluid_body:
                object_visibility[obj_id] = fluid_body["visible"]
            else:
                object_visibility[obj_id] = 1

            object_materials[obj_id] = material_fluid
            object_id_fluid_body.add(obj_id)
            object_collection[obj_id] = fluid_body

            add_particles(obj_id,
                          num_particles_obj,
                          np.array(voxelized_points_np, dtype=np.float32), # position
                          np.stack([velocity for _ in range(num_particles_obj)]), # velocity
                          density * np.ones(num_particles_obj, dtype=np.float32), # density
                          np.zeros(num_particles_obj, dtype=np.float32), # pressure
                          np.array([material_fluid for _ in range(num_particles_obj)], dtype=np.int32), 
                          1 * np.ones(num_particles_obj, dtype=np.int32), # dynamic
                          np.stack([color for _ in range(num_particles_obj)]))

            present_object.append(obj_id)
            fluid_particle_num[None] += num_particles_obj