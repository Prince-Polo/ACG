import imageio.v2 as imageio
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help="Directory containing the rendered frames (output_frames).")
parser.add_argument('--output_path', type=str, required=True, help="Output video file path.")
parser.add_argument('--fps', type=int, default=20, help="Frames per second for the video.")

args = parser.parse_args()

# 获取输入目录中的所有帧文件
frame_list = os.listdir(args.input_dir)
# 按照帧的顺序对文件进行排序（假设帧文件的命名格式是类似 frame_000001.png）
frame_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # 例如 frame_000001.png -> 000001

# 准备图像列表
images = []
for frame in frame_list:
    file_path = os.path.join(args.input_dir, frame)
    try:
        # 读取图像并添加到列表中
        images.append(imageio.imread(file_path))
    except Exception as e:
        print(f"Failed to load image from {frame}: {e}")

# 使用 imageio 保存为视频文件
imageio.mimsave(args.output_path, images, fps=args.fps)

print(f"视频生成完成，输出路径为：{args.output_path}")
