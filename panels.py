import bpy
from .operators import MODEL_PATH
import os

class VGGTPanel(bpy.types.Panel):
    bl_label = "VGGT"
    bl_idname = "VIEW3D_PT_vggt"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "VGGT"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        row = layout.row()
        if os.path.exists(MODEL_PATH):
            row.label(text="Model already downloaded")
        else:
            row.operator("vggt.download_model")
        layout.prop(scene, "vggt_input_folder", text="Input Folder")
        row = layout.row()
        row.operator("vggt.generate_point_cloud")