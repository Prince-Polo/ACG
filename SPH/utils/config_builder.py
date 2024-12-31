import json
from typing import Dict, List, Optional, Any

class SimConfig:
    """模拟配置管理器"""
    
    def __init__(self, scene_file_path: str) -> None:
        """从JSON文件加载配置"""
        with open(scene_file_path, "r") as f:
            self.config: Dict[str, Any] = json.load(f)
        print(self.config)
    
    def get_cfg(self, name: str, enforce_exist: bool = False) -> Optional[Any]:
        """获取指定名称的配置值"""
        if enforce_exist:
            assert name in self.config["Configuration"]
        return self.config["Configuration"].get(name)

    def _get_bodies(self, key: str) -> List[Dict[str, Any]]:
        """通用的物体配置获取方法"""
        return self.config.get(key, [])

    def get_rigid_bodies(self) -> List[Dict[str, Any]]:
        return self._get_bodies("RigidBodies")
    
    def get_rigid_blocks(self) -> List[Dict[str, Any]]:
        return self._get_bodies("RigidBlocks")

    def get_fluid_bodies(self) -> List[Dict[str, Any]]:
        return self._get_bodies("FluidBodies")
    
    def get_fluid_blocks(self) -> List[Dict[str, Any]]:
        return self._get_bodies("FluidBlocks")