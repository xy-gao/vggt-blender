bl_info = {
    "name": "VGGT Addon",
    "author": "Xiangyi Gao",
    "version": (1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > VGGT",
    "description": "Generate point clouds from images using VGGT",
    "category": "3D View",
}

import bpy
from .dependencies import Dependencies


def register():
    if not Dependencies.check():
        Dependencies.install()
    if Dependencies.check():
        from . import operators, panels
        bpy.utils.register_class(operators.DownloadModelOperator)
        bpy.utils.register_class(operators.GeneratePointCloudOperator)
        bpy.utils.register_class(panels.VGGTPanel)
        bpy.types.Scene.vggt_input_folder = bpy.props.StringProperty(subtype='DIR_PATH')
    else:
        raise ValueError("installation failed.")

def unregister():
    if Dependencies.check():
        from . import operators, panels
        bpy.utils.unregister_class(operators.DownloadModelOperator)
        bpy.utils.unregister_class(operators.GeneratePointCloudOperator)
        bpy.utils.unregister_class(panels.VGGTPanel)
        del bpy.types.Scene.vggt_input_folder

if __name__ == "__main__":
    register()