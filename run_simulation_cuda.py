import os
import argparse
import numpy as np
from SPH.utils import SimConfig
from SPH.containers.dfsph_container_numba import DFSPHContainer
from SPH.fluid_solvers.dfsph_solver_numba import DFSPHSolver

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_file',
                        default='',
                        help='scene file')
    args = parser.parse_args()
    scene_path = args.scene_file
    config = SimConfig(scene_file_path=scene_path)
    scene_name = scene_path.split("/")[-1].split(".")[0]

    output_frames = config.get_cfg("exportFrame")

    fps = config.get_cfg("fps")
    if fps == None:
        fps = 60

    frame_time = 1.0 / fps

    output_interval = int(frame_time / config.get_cfg("timeStepSize"))

    total_time = config.get_cfg("totalTime")
    if total_time == None:
        total_time = 10.0

    total_rounds = int(total_time / config.get_cfg("timeStepSize"))
    
    if config.get_cfg("outputInterval"):
        output_interval = config.get_cfg("outputInterval")

    output_ply = config.get_cfg("exportPly")
    output_obj = config.get_cfg("exportObj")

    os.makedirs(f"{scene_name}_output", exist_ok=True)

    simulation_method = config.get_cfg("simulationMethod")
    if simulation_method == "dfsph":
        container = DFSPHContainer(config)
        solver = DFSPHSolver(container)
    else:
        raise NotImplementedError(f"Simulation method {simulation_method} not implemented")

    solver.prepare()

    cnt = 0
    cnt_ply = 0

    while cnt < total_rounds:
        solver.step()
        
        if output_frames:
            if cnt % output_interval == 0:
                os.makedirs(f"{scene_name}_output/{cnt:06}", exist_ok=True)
                # 保存粒子位置到文件
                positions = container.particle_positions.copy_to_host()
                np.save(f"{scene_name}_output/{cnt:06}/positions.npy", positions)
        
        if cnt % output_interval == 0:
            if output_ply:
                os.makedirs(f"{scene_name}_output/{cnt:06}", exist_ok=True)
                for f_body_id in container.object_id_fluid_body:
                    obj_data = container.dump(obj_id=f_body_id)
                    np_pos = obj_data["position"]
                    with open(f"{scene_name}_output/{cnt:06}/particle_object_{f_body_id}.ply", 'w') as f:
                        f.write("ply\n")
                        f.write("format ascii 1.0\n")
                        f.write(f"element vertex {len(np_pos)}\n")
                        f.write("property float x\n")
                        f.write("property float y\n")
                        f.write("property float z\n")
                        f.write("end_header\n")
                        for pos in np_pos:
                            f.write(f"{pos[0]} {pos[1]} {pos[2]}\n")
            
            if output_obj:
                os.makedirs(f"{scene_name}_output/{cnt:06}", exist_ok=True)
                for r_body_id in container.object_id_rigid_body:
                    with open(f"{scene_name}_output/{cnt:06}/mesh_object_{r_body_id}.obj", "w") as f:
                        e = container.object_collection[r_body_id]["mesh"].export(file_type='obj')
                        f.write(e)

        cnt += 1

    print(f"Simulation Finished")