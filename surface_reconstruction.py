import os
import multiprocessing as mp
from tqdm import tqdm
import argparse

class MeshReconstructor:
    """网格重建器"""
    
    def __init__(self, input_dir, num_workers=4, radius=0.01, smoothing_length=3.5):
        self.input_path = input_dir
        self.worker_count = num_workers
        self.mesh_params = {
            'radius': radius,
            'smoothing': smoothing_length
        }
        self.reconstruction_cmd = (
            "splashsurf reconstruct {input} -o {output} -q "
            "-r={params[radius]} -l={params[smoothing]} -c=0.5 -t=0.6 "
            "--subdomain-grid=on --mesh-cleanup=on "
            "--mesh-smoothing-weights=on --mesh-smoothing-iters=25 "
            "--normals=on --normals-smoothing-iters=10"
        )
        
    def _get_frame_list(self):
        """获取并排序帧列表"""
        frames = os.listdir(self.input_path)
        frames.sort(key=lambda x: int(x))
        return frames
        
    def _process_single_frame(self, frame_path):
        """处理单个帧"""
        try:
            # 获取当前帧中的所有PLY文件
            ply_files = [f for f in os.listdir(frame_path) if f.endswith('.ply')]
            
            # 处理每个PLY文件
            for ply_name in ply_files:
                input_file = os.path.join(frame_path, ply_name)
                output_file = input_file.replace('.ply', '.obj')
                
                # 构建并执行重建命令
                cmd = self.reconstruction_cmd.format(
                    input=input_file,
                    output=output_file,
                    params=self.mesh_params
                )
                os.system(cmd)
                
            return True
            
        except Exception as e:
            print(f"处理失败: {frame_path}")
            print(f"错误信息: {str(e)}")
            return False
            
    def execute(self):
        """执行重建过程"""
        frames = self._get_frame_list()
        total_frames = len(frames)
        
        # 创建进度条
        progress = tqdm(total=total_frames, desc="重建进度")
        
        # 创建进程池
        with mp.Pool(self.worker_count) as pool:
            # 提交所有任务
            tasks = []
            for frame in frames:
                frame_path = os.path.join(self.input_path, frame)
                task = pool.apply_async(
                    self._process_single_frame,
                    args=(frame_path,),
                    callback=lambda _: progress.update(1)
                )
                tasks.append(task)
            
            # 等待所有任务完成
            for task in tasks:
                task.wait()
                
        # 关闭进度条
        progress.close()
        
        # 统计处理结果
        success_count = sum(1 for task in tasks if task.get())
        print(f"\n重建完成: {success_count}/{total_frames} 帧成功处理")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="表面重建工具")
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入数据目录路径')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='并行处理的工作进程数')
    parser.add_argument('--radius', type=float, default=0.01,
                       help='重建半径参数')
    parser.add_argument('--smoothing-length', type=float, default=3.5,
                       help='平滑长度参数')
    
    args = parser.parse_args()
    
    # 创建重建器并执行
    reconstructor = MeshReconstructor(
        input_dir=args.input_dir,
        num_workers=args.num_workers,
        radius=args.radius,
        smoothing_length=args.smoothing_length
    )
    reconstructor.execute()


if __name__ == "__main__":
    main()
