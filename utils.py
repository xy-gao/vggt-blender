import torch
import glob
import os
import numpy as np
import bpy
import math
from mathutils import Matrix
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

def run_model(target_dir, model):
    print(f"Processing images from {target_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")
    model = model.to(device)
    model.eval()
    image_names = glob.glob(os.path.join(target_dir, "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")
    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions["images"] = images.cpu().numpy()
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points
    torch.cuda.empty_cache()
    return predictions

def import_point_cloud(d):
    points = d["world_points_from_depth"]
    images = d["images"]
    points_batch = points.reshape(-1, 3)
    reordered_points_batch = points_batch.copy()
    reordered_points_batch[:, [0, 1, 2]] = points_batch[:, [0, 2, 1]]
    reordered_points_batch[:, 2] = -reordered_points_batch[:, 2]
    points_batch = reordered_points_batch
    colors_batch = images.transpose(0, 2, 3, 1).reshape(-1, 3)
    colors_batch = np.hstack((colors_batch, np.ones((colors_batch.shape[0], 1))))
    mesh = bpy.data.meshes.new(name="Points")
    vertices = points_batch.tolist()
    mesh.from_pydata(vertices, [], [])
    attribute = mesh.attributes.new(name="point_color", type="FLOAT_COLOR", domain="POINT")
    color_values = colors_batch.flatten().tolist()
    attribute.data.foreach_set("color", color_values)
    obj = bpy.data.objects.new("Points", mesh)
    bpy.context.collection.objects.link(obj)
    mat = bpy.data.materials.new(name="PointMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in nodes:
        nodes.remove(node)
    attr_node = nodes.new('ShaderNodeAttribute')
    attr_node.attribute_name = "point_color"
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    links.new(attr_node.outputs['Color'], bsdf.inputs['Base Color'])
    output_node_material = nodes.new('ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], output_node_material.inputs['Surface'])
    geo_mod = obj.modifiers.new(name="GeometryNodes", type='NODES')
    node_group = bpy.data.node_groups.new(name="PointCloud", type='GeometryNodeTree')
    node_group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    node_group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    geo_mod.node_group = node_group
    input_node = node_group.nodes.new('NodeGroupInput')
    output_node = node_group.nodes.new('NodeGroupOutput')
    mesh_to_points = node_group.nodes.new('GeometryNodeMeshToPoints')
    mesh_to_points.inputs['Radius'].default_value = 0.002
    set_material_node = node_group.nodes.new('GeometryNodeSetMaterial')
    set_material_node.inputs['Material'].default_value = mat
    node_group.links.new(input_node.outputs['Geometry'], mesh_to_points.inputs['Mesh'])
    node_group.links.new(mesh_to_points.outputs['Points'], set_material_node.inputs['Geometry'])
    node_group.links.new(set_material_node.outputs['Geometry'], output_node.inputs['Geometry'])

def create_cameras(predictions, image_width=None, image_height=None):
    """
    Create Blender cameras from extrinsic and intrinsic parameters.
    
    Args:
        predictions (dict): Dictionary with 'extrinsic' and 'intrinsic' keys, each containing
                           lists of NumPy arrays (3x4 for extrinsic, 3x3 for intrinsic).
        image_width (int, optional): Image width in pixels. Defaults to scene render resolution.
        image_height (int, optional): Image height in pixels. Defaults to scene render resolution.
    """
    # Get the current scene
    scene = bpy.context.scene
    
    # Use scene render resolution if image_width or image_height not provided
    if image_width is None:
        image_width = scene.render.resolution_x
    if image_height is None:
        image_height = scene.render.resolution_y
    
    # Set pixel aspect ratio based on the first camera's intrinsic matrix
    K0 = predictions["intrinsic"][0]
    pixel_aspect_y = K0[1,1] / K0[0,0]
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = float(pixel_aspect_y)
    
    # Ensure equal length of extrinsic and intrinsic lists
    num_cameras = len(predictions["extrinsic"])
    if len(predictions["intrinsic"]) != num_cameras:
        raise ValueError("Extrinsic and intrinsic lists must have the same length")
    
    # Define coordinate system transformation matrix (OpenCV to Blender)
    T = np.diag([1.0, -1.0, -1.0, 1.0])  # 180-degree rotation around X-axis
    
    # Create cameras
    for i in range(num_cameras):
        # Create new camera data
        cam_data = bpy.data.cameras.new(name=f"Camera_{i}")
        
        # Set intrinsic parameters
        K = predictions["intrinsic"][i]
        f_x = K[0,0]
        c_x = K[0,2]
        c_y = K[1,2]
        
        # Set sensor width (standard full-frame sensor size)
        sensor_width = 36.0  # in mm
        cam_data.sensor_width = sensor_width
        
        # Compute focal length in mm
        cam_data.lens = (f_x / image_width) * sensor_width
        
        # Set principal point shifts
        cam_data.shift_x = 0
        cam_data.shift_y = 0
        
        # Create camera object and link to scene
        cam_obj = bpy.data.objects.new(name=f"Camera_{i}", object_data=cam_data)
        scene.collection.objects.link(cam_obj)
        
        # Set extrinsic parameters
        ext = predictions["extrinsic"][i]
        # Convert 3x4 extrinsic matrix to 4x4
        E = np.vstack((ext, [0, 0, 0, 1]))
        # Compute inverse
        E_inv = np.linalg.inv(E)
        # Compute camera-to-world matrix
        M = np.dot(E_inv, T)
        # Convert to Blender Matrix
        cam_obj.matrix_world = Matrix(M.tolist())
        # Create a 90-degree rotation matrix around the Z-axis (counter-clockwise)
        R = Matrix.Rotation(math.radians(-90), 4, 'X')

        # Apply the rotation to the camera's world matrix
        cam_obj.matrix_world = R @ cam_obj.matrix_world
