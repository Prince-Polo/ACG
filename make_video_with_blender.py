import os
import argparse
import imageio.v2 as imageio
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

class BlenderFrameCompiler:
    """Blender渲染帧编译器"""
    
    def __init__(self, input_dir: str, output_path: str, fps: int = 20):
        self.input_dir = Path(input_dir)
        self.output_path = Path(output_path)
        self.fps = fps
        self.frames: List[imageio.core.Array] = []
        
    def _validate_paths(self) -> None:
        """验证输入输出路径"""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"输入目录不存在: {self.input_dir}")
            
        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def _collect_frame_files(self) -> List[Path]:
        """收集并排序帧文件"""
        frame_files = [f for f in self.input_dir.iterdir() if f.is_file()]
        
        # 按帧号排序（从文件名中提取数字）
        frame_files.sort(key=lambda x: int(x.stem.split('_')[1]))
        
        if not frame_files:
            raise RuntimeError(f"未在 {self.input_dir} 中找到任何帧文件")
            
        return frame_files
        
    def _load_frames(self, frame_files: List[Path]) -> None:
        """加载所有帧"""
        print("正在加载渲染帧...")
        for frame_path in tqdm(frame_files):
            try:
                frame = imageio.imread(str(frame_path))
                self.frames.append(frame)
            except Exception as e:
                print(f"警告: 无法加载帧 {frame_path.name}: {e}")
                
        if not self.frames:
            raise RuntimeError("未能成功加载任何帧")
            
    def _generate_video(self) -> None:
        """生成视频文件"""
        print(f"正在生成视频: {self.output_path}")
        try:
            imageio.mimsave(
                str(self.output_path),
                self.frames,
                fps=self.fps
            )
            self._print_summary()
        except Exception as e:
            raise RuntimeError(f"视频生成失败: {e}")
            
    def _print_summary(self) -> None:
        """打印处理摘要"""
        print("\n视频生成完成!")
        print(f"- 总帧数: {len(self.frames)}")
        print(f"- 帧率: {self.fps} FPS")
        print(f"- 输出文件: {self.output_path}")
        
    def compile(self) -> None:
        """执行完整的编译流程"""
        try:
            self._validate_paths()
            frame_files = self._collect_frame_files()
            self._load_frames(frame_files)
            self._generate_video()
        except Exception as e:
            print(f"错误: {str(e)}")
            raise


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Blender渲染帧视频生成工具")
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help="渲染帧输入目录"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help="输出视频文件路径"
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=20,
        help="视频帧率"
    )
    return parser.parse_args()


def main() -> None:
    """主函数"""
    try:
        args = parse_arguments()
        compiler = BlenderFrameCompiler(
            input_dir=args.input_dir,
            output_path=args.output_path,
            fps=args.fps
        )
        compiler.compile()
    except KeyboardInterrupt:
        print("\n处理被用户中断")
        exit(1)
    except Exception as e:
        print(f"处理失败: {e}")
        exit(1)


if __name__ == "__main__":
    main()
