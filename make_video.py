import os
import sys
import argparse
from pathlib import Path
import imageio.v2 as imageio
from typing import List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class FrameSequenceProcessor:
    """帧序列处理器 - 用于将图像序列转换为视频"""
    
    def __init__(self):
        self.args = self._parse_arguments()
        self.frame_paths = []
        self.image_data = []
        
    def _parse_arguments(self) -> argparse.Namespace:
        """解析命令行参数"""
        parser = argparse.ArgumentParser(description="序列帧转视频工具")
        parser.add_argument('--input_dir', type=str, required=True,
                          help="输入序列帧目录")
        parser.add_argument('--image_name', type=str, default='raw_view.png',
                          help="图像文件名")
        parser.add_argument('--output_path', type=str, required=True,
                          help="输出视频路径")
        parser.add_argument('--fps', type=int, default=20,
                          help="输出视频帧率")
        return parser.parse_args()
        
    def _scan_directory(self) -> None:
        """扫描目录获取所有帧路径"""
        try:
            base_path = Path(self.args.input_dir)
            # 获取所有子目录并按数字排序
            subdirs = sorted(
                [d for d in base_path.iterdir() if d.is_dir()],
                key=lambda x: int(x.name)
            )
            
            # 构建完整的图像路径
            self.frame_paths = [
                d / self.args.image_name for d in subdirs
            ]
            
            if not self.frame_paths:
                print("错误：未找到任何图像文件")
                sys.exit(1)
                
        except Exception as e:
            print(f"目录扫描失败: {e}")
            sys.exit(1)
            
    def _load_single_frame(self, path: Path) -> Optional[Any]:
        """加载单个图像帧"""
        try:
            if not path.exists():
                print(f"文件不存在: {path}")
                return None
            return imageio.imread(str(path))
        except Exception as e:
            print(f"无法加载图像 {path.parent.name}: {e}")
            return None
            
    def _load_frames(self) -> None:
        """并行加载所有图像帧"""
        print("正在加载图像序列...")
        with ThreadPoolExecutor() as executor:
            futures = []
            for path in self.frame_paths:
                future = executor.submit(self._load_single_frame, path)
                futures.append(future)
            
            # 收集结果
            for future in tqdm(futures):
                result = future.result()
                if result is not None:
                    self.image_data.append(result)
                    
        if not self.image_data:
            print("错误：未能成功加载任何图像")
            sys.exit(1)
            
    def _create_video(self) -> None:
        """生成视频文件"""
        try:
            print(f"正在生成视频: {self.args.output_path}")
            imageio.mimsave(
                self.args.output_path,
                self.image_data,
                fps=self.args.fps
            )
            print(f"视频生成成功！")
            print(f"- 总帧数: {len(self.image_data)}")
            print(f"- 帧率: {self.args.fps} FPS")
            print(f"- 输出路径: {self.args.output_path}")
            
        except Exception as e:
            print(f"视频生成失败: {e}")
            sys.exit(1)
            
    def process(self) -> None:
        """执行完整的处理流程"""
        self._scan_directory()
        self._load_frames()
        self._create_video()


def main():
    """主函数"""
    try:
        processor = FrameSequenceProcessor()
        processor.process()
    except KeyboardInterrupt:
        print("\n处理被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"发生未预期的错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
