import json

class SimConfig:
    def __init__(self, scene_file_path) -> None:
        try:
            f = open(scene_file_path, "r")
            self.config = json.load(f)
            f.close()
            print(f"已加载场景配置:{scene_file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"未找到场景文件: {scene_file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"场景文件格式错误: {scene_file_path}")
    
    def get_cfg(self, name, enforce_exist=False):
        config_dict = self.config.get("Configuration", {})
        if enforce_exist and name not in config_dict:
            raise KeyError(f"Required configuration '{name}' not found")
        return config_dict.get(name)

    def _get_config_section(self, section_name):
        """统一的配置获取方法"""
        return self.config.get(section_name, [])
    
    def get_rigid_bodies(self):
        return self._get_config_section("RigidBodies")
    
    def get_rigid_blocks(self):
        return self._get_config_section("RigidBlocks")

    def get_fluid_bodies(self):
        return self._get_config_section("FluidBodies")
    
    def get_fluid_blocks(self):
        return self._get_config_section("FluidBlocks")