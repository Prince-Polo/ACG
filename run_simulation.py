import taichi as ti
import os
import argparse
from SPH.utils import SimConfig
from SPH.containers import DFSPHContainer, IISPHContainer
from SPH.fluid_solvers import DFSPHSolver, IISPHSolver

ti.init(arch=ti.gpu, device_memory_fraction=0.8)

def create_box_lines(domain_end):
    """创建边界框线段"""
    x_max, y_max, z_max = domain_end
    box_anchors = ti.Vector.field(3, dtype=ti.f32, shape=8)
    points = [
        [0.0, 0.0, 0.0], [0.0, y_max, 0.0],
        [x_max, 0.0, 0.0], [x_max, y_max, 0.0],
        [0.0, 0.0, z_max], [0.0, y_max, z_max],
        [x_max, 0.0, z_max], [x_max, y_max, z_max]
    ]
    for i, point in enumerate(points):
        box_anchors[i] = ti.Vector(point)
    
    box_lines_indices = ti.field(int, shape=(2 * 12))
    lines = [0,1, 0,2, 1,3, 2,3, 4,5, 4,6, 5,7, 6,7, 0,4, 1,5, 2,6, 3,7]
    for i, val in enumerate(lines):
        box_lines_indices[i] = val
    
    return box_anchors, box_lines_indices

def handle_output(cnt, output_interval, scene_name, container, output_frames, output_ply, output_obj, window):
    """处理输出逻辑"""
    if cnt % output_interval != 0:
        return
        
    output_path = f"{scene_name}_output/{cnt:06}"
    os.makedirs(output_path, exist_ok=True)
    
    if output_frames:
        window.save_image(f"{output_path}/raw_view.png")
    
    if output_ply:
        for f_body_id in container.object_id_fluid_body:
            obj_data = container.dump(obj_id=f_body_id)
            np_pos = obj_data["position"]
            writer = ti.tools.PLYWriter(num_vertices=container.object_collection[f_body_id]["particleNum"])
            writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
            writer.export_ascii(f"{output_path}/particle_object_{f_body_id}.ply")
    
    if output_obj:
        for r_body_id in container.object_id_rigid_body:
            with open(f"{output_path}/mesh_object_{r_body_id}.obj", "w") as f:
                e = container.object_collection[r_body_id]["mesh"].export(file_type='obj')
                f.write(e)

def render_scene(scene, container, box_anchors, box_lines_indices, dim):
    """渲染场景"""
    scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
    scene.particles(container.x_vis_buffer, radius=container.radius, per_vertex_color=container.color_vis_buffer)
    if dim == 3:
        scene.lines(box_anchors, indices=box_lines_indices, color=(0.99, 0.68, 0.28), width=1.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_file', default='', help='场景文件')
    args = parser.parse_args()
    
    config = SimConfig(scene_file_path=args.scene_file)
    scene_name = args.scene_file.split("/")[-1].split(".")[0]

    # 初始化模拟参数
    fps = config.get_cfg("fps") or 60
    frame_time = 1.0 / fps
    time_step = config.get_cfg("timeStepSize")
    output_interval = config.get_cfg("outputInterval") or int(frame_time / time_step)
    total_time = config.get_cfg("totalTime") or 10.0
    total_rounds = int(total_time / time_step)

    os.makedirs(f"{scene_name}_output", exist_ok=True)

    # 初始化求解器
    simulation_method = config.get_cfg("simulationMethod")
    if simulation_method == "dfsph":
        container = DFSPHContainer(config, GGUI=True)
        solver = DFSPHSolver(container)
    elif simulation_method == "iisph":
        container = IISPHContainer(config, GGUI=True)
        solver = IISPHSolver(container)
    else:
        raise NotImplementedError(f"未实现的模拟方法 {simulation_method}")

    solver.prepare()

    # 设置窗口和相机
    window = ti.ui.Window('SPH', (1024, 1024), show_window=False, vsync=False)
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(5.5, 2.5, 4.0)
    camera.up(0.0, 1.0, 0.0)
    camera.lookat(-1.0, 0.0, 0.0)
    camera.fov(70)
    scene.set_camera(camera)
    canvas = window.get_canvas()

    # 设置渲染参数
    invisible_objects = config.get_cfg("invisibleObjects") or []
    domain_end = config.get_cfg("domainEnd")
    dim = len(domain_end)
    box_anchors, box_lines_indices = create_box_lines(domain_end) if dim == 3 else (None, None)

    # 主循环
    cnt = 0
    while window.running:
        solver.step()
        container.copy_to_vis_buffer(invisible_objects=invisible_objects, dim=dim)
        
        if container.dim == 2:
            canvas.set_background_color((0, 0, 0))
            canvas.circles(container.x_vis_buffer, radius=container.dx/80.0, color=(1, 1, 1))
        elif container.dim == 3:
            render_scene(scene, container, box_anchors, box_lines_indices, dim)
            canvas.scene(scene)
        
        handle_output(
            cnt, output_interval, scene_name, container,
            config.get_cfg("exportFrame"),
            config.get_cfg("exportPly"),
            config.get_cfg("exportObj"),
            window
        )
        
        cnt += 1
        if cnt >= total_rounds:
            break

    print("模拟完成")