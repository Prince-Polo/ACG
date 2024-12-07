import bpy
import os

# 渲染输出文件夹
output_path = "E:/Documents/Homework/graphics/ACG/output_frames"
os.makedirs(output_path, exist_ok=True)

# 设置场景
blend_file_path = "E:/Documents/Homework/graphics/ACG/scene.blend"

# 加载 scene.blend 文件
bpy.ops.wm.open_mainfile(filepath=blend_file_path)

# 获取当前场景
scene = bpy.context.scene

# 获取场景中的相机对象
camera_object = scene.camera  # 从加载的文件中获取相机对象

# 查找光源对象
light_object = None
for obj in scene.objects:
    if obj.type == 'LIGHT' and obj.name == "Light":
        light_object = obj
        break

if not light_object:
    raise ValueError("Light object 'Light' not found in the scene.")

# 设置相机和光源位置
scene.camera = camera_object
light_object.location = (2.0, 2.0, 2.0)  # 适当位置调整光源

# 处理Plane对象
plane_object = None
for obj in scene.objects:
    if obj.name == "Plane":
        plane_object = obj
        break

if not plane_object:
    raise ValueError("Plane object 'Plane' not found in the scene.")

# 设置渲染引擎为 Cycles (启用光追)
scene.render.engine = 'CYCLES'

# 强制启用 OptiX 渲染
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPTIX'

# 设置光线追踪的采样数量
scene.cycles.samples = 32  # 调整采样数量，默认 128，越高质量越好，时间越长

# 设置渲染设置
scene.render.image_settings.file_format = 'PNG'
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080

# 输入文件夹路径（每帧对应的 .obj 文件）
input_dir = "fluid_rigid_coupling1_output"
frames = sorted(os.listdir(input_dir))

# 检查当前使用的渲染设备
cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
render_device_type = cycles_prefs.compute_device_type
print("当前渲染设备类型:", render_device_type)

if render_device_type == 'OPTIX':
    print("OptiX 渲染已启用")
else:
    print("未启用 OptiX 渲染，使用默认设备")

# 遍历每一帧，导入对象并渲染
for frame_idx, frame_dir in enumerate(frames):
    frame_path = os.path.join(input_dir, frame_dir)
    obj_files = sorted([f for f in os.listdir(frame_path) if f.endswith(".obj")])

    # 删除场景中现有的对象，只保留相机、光源和plane
    for obj in scene.objects:
        # 删除非相机、非光源、非Plane对象
        if obj.type not in {'CAMERA', 'LIGHT'} and obj.name != "Plane":
            bpy.data.objects.remove(obj, do_unlink=True)

    # 确保plane仍然存在
    if "Plane" not in scene.objects:
        scene.collection.objects.link(plane_object)

    # 批量导入 .obj 文件并应用材质
    for obj_idx, obj_file in enumerate(obj_files):
        obj_path = os.path.join(frame_path, obj_file)
        bpy.ops.wm.obj_import(filepath=obj_path)

        # 查找导入的对象，并设置材质
        for obj in scene.objects:
            if obj.type == 'MESH':
                if obj_idx == 0:
                    # 为第一个对象设置特定材质
                    obj.data.materials.append(bpy.data.materials["AR3DMat Procedural Realistic Mirror"])
                elif obj_idx == 1:
                    # 为第二个对象设置特定材质
                    obj.data.materials.append(bpy.data.materials["AR3DMat Procedural Realistic Mirror"])
                elif obj_idx == 2:
                    # 为第三个对象设置水材质
                    obj.data.materials.append(bpy.data.materials["Water"])

    # 设置输出路径
    scene.render.filepath = os.path.join(output_path, f"frame_{frame_idx:06d}.png")
    bpy.ops.render.render(write_still=True)

    # 清理未使用的内存
    bpy.ops.wm.save_as_mainfile(filepath=blend_file_path)  # 保存场景，以清理内存

print("所有帧已渲染完成。")
