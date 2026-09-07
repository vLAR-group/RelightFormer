"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

"""
# import bpy
import argparse
import json
import math
import os
import random
import sys
sys.path.append('..')
import time
import urllib.request
import uuid
from typing import Tuple
from mathutils import Vector, Matrix
import numpy as np
import bpy
from mathutils import Vector
from glob import glob
import traceback
import objaverse
import shutil
import torch
import cv2
import copy
import hashlib

LIGHTING_INFO = './info/laval/full_[split]_lighting.json'
VIEW_INFO = './info/view/full_[split]_view.json'
ALL_VIEWS = './info/view/all_view.json'

DEPTH_SCALE = 1

context = bpy.context
scene = context.scene
render = scene.render

LIGHTING_DIR = "./laval/preprocessed"
VIEW_DIR = "./info/view"


N_LIGHTINGS = 16
N_VIEWS = {
    "training": 16,
    "testing": 200,
    "validation": 200,
}

def find_key_by_value(d, item):
    for key, lst in d.items():
        if item in lst:
            return key
    return None  # Item not found

def get_lighting(seed, json_path, N=16):
    """
    Select N environments deterministically seeded by `name`.
    
    Guarantees:
    - Exactly N//2 items from "Indoor" and N//2 from "Outdoor"
    - First 4 items contain exactly 2 Indoor and 2 Outdoor
    
    Args:
        name (str): String used as seed for deterministic selection
        N (int): Total number of environments to select (must be even)
        json_path (str): Path to JSON file with {"Indoor": [...], "Outdoor": [...]}
    
    Returns:
        list: Selected environment names
    """
    if N % 2 != 0:
        raise ValueError("N must be even to split equally between Indoor/Outdoor")
    
    # Load environment lists
    with open(json_path, 'r') as f:
        env_data = json.load(f)
    
    indoor_list = env_data["Indoor"]
    outdoor_list = env_data["Outdoor"]
    

    # Calculate selection counts
    half_n = N // 2
    first_half_indoor = 2  # Required for first 4 items constraint
    first_half_outdoor = 2
    remaining_indoor = half_n - first_half_indoor
    remaining_outdoor = half_n - first_half_outdoor
    
    if len(indoor_list) < half_n or len(outdoor_list) < half_n:
        raise ValueError(f"Need at least {half_n} items in both Indoor and Outdoor lists")
    
    # Select items deterministically
    shuffled_indoor = indoor_list[:]
    random.Random(seed).shuffle(shuffled_indoor)

    shuffled_outdoor = outdoor_list[:]
    random.Random(seed).shuffle(shuffled_outdoor)
    selected_indoor = shuffled_indoor[:half_n]
    selected_outdoor = shuffled_outdoor[:N - half_n]
    
    # Construct result with constraint on first 4 items
    result = []
    # First 4: 2 Indoor + 2 Outdoor (interleaved for balance)
    result.extend(selected_indoor[:first_half_indoor])
    result.extend(selected_outdoor[:first_half_outdoor])
    
    # Remaining items: append rest of selections
    result.extend(selected_indoor[first_half_indoor:])
    result.extend(selected_outdoor[first_half_outdoor:])
    
    # Optional: shuffle remaining items (positions 4+) while preserving first 4 constraint
    if len(result) > 4:
        tail = result[4:]
        random.Random(seed).shuffle(tail)
        result = result[:4] + tail
    
    return result

def get_view(seed, json_path, N=16):
    """
    Load environment list from JSON and optionally sample N items deterministically.
    
    Args:
        name (str): String used as seed for deterministic sampling
        json_path (str): Path to JSON file containing a flat list of views
        N (int, optional): Number of items to sample. If None, return full list.
    
    Returns:
        list: Full list (if N=None) or sampled list of N items
    """

    # Load JSON list
    with open(json_path, 'r') as f:
        view_list = json.load(f)
    
    # Validate JSON structure
    if not isinstance(view_list, list):
        raise ValueError(f"JSON file must contain a flat list, got {type(view_list)}")
    
    # Return full list if N not specified
    if N is None or N >= len(view_list):
        return view_list.copy()  # Return copy to avoid external mutation
    
    # Sample N items deterministically
    shuffled= view_list[:]
    random.Random(seed).shuffle(shuffled)
    selected = shuffled[:N]
    return selected

# add environment map as the lighting condition
def add_light_env(env=(1, 1, 1, 1), strength=1, rot_vec_rad=(0, 0, 0), scale=(1, 1, 1)):
    r"""Adds environment lighting.
    Args:
        env (tuple(float) or str, optional): Environment map. If tuple,
            it's RGB or RGBA, each element of which :math:`\in [0,1]`.
            Otherwise, it's the path to an image.
        strength (float, optional): Light intensity.
        rot_vec_rad (tuple(float), optional): Rotations in radians around x, y and z.
        scale (tuple(float), optional): If all changed simultaneously, then no effects.
    """
    engine = bpy.context.scene.render.engine
    assert engine == "CYCLES", "Rendering engine is not Cycles"

    if isinstance(env, str):
        bpy.data.images.load(env, check_existing=True)
        env = bpy.data.images[os.path.basename(env)]
    else:
        if len(env) == 3:
            env += (1,)
        assert len(env) == 4, "If tuple, env must be of length 3 or 4"

    world = bpy.context.scene.world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    bg_node = nodes.new("ShaderNodeBackground")
    links.new(bg_node.outputs["Background"], nodes["World Output"].inputs["Surface"])

    if isinstance(env, tuple):
        # Color
        bg_node.inputs["Color"].default_value = env
        print(("Environment is pure color, " "so rotation and scale have no effect"))
    else:
        # Environment map
        texcoord_node = nodes.new("ShaderNodeTexCoord")
        env_node = nodes.new("ShaderNodeTexEnvironment")
        env_node.image = env
        mapping_node = nodes.new("ShaderNodeMapping")
        mapping_node.inputs["Rotation"].default_value = rot_vec_rad
        mapping_node.inputs["Scale"].default_value = scale
        links.new(texcoord_node.outputs["Generated"], mapping_node.inputs["Vector"])
        links.new(mapping_node.outputs["Vector"], env_node.inputs["Vector"])
        links.new(env_node.outputs["Color"], bg_node.inputs["Color"])

    bg_node.inputs["Strength"].default_value = strength

    return env

def bpy_image_2_torch(env, size):
    # Get the pixel data as a flat array
    pixels = np.array(env.pixels[:])  # Get the pixel data
    width = env.size[0]
    height = env.size[1]

    image_array = pixels.reshape((height, width, 4))[:,:,3]

    # Convert to RGB by taking the first three channels
    rgb_resized = cv2.resize(image_array, size, interpolation=cv2.INTER_LINEAR)
    # Resize the image using Pillow

    # Convert the resized image back to a NumPy array
    rgb_tensor = torch.from_numpy(rgb_resized).to(torch.float32) 
    return rgb_tensor

def remove_unwanted_objects():
    """
    Remove unwanted objects from the scene, such as lights and background plane objects.
    """
    # Remove undesired objects and existing lights
    objs = []
    for o in bpy.data.objects:
        if o.name == 'BackgroundPlane':
            objs.append(o)
        elif o.type == 'LIGHT':
            objs.append(o)
        elif o.active_material is not None:
            for node in o.active_material.node_tree.nodes:
                if node.type == 'EMISSION':
                    objs.append(o)
    bpy.ops.object.delete({'selected_objects': objs})

def reset_scene():
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

    scene.use_nodes = True

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    # Create input render layer node
    render_layers = nodes.new("CompositorNodeRLayers")
    render_layers.label = 'Custom Outputs'
    render_layers.name = 'Custom Outputs'

    bpy.context.view_layer.use_pass_normal = True
    bpy.context.view_layer.use_pass_diffuse_color = True
    bpy.context.view_layer.use_pass_z = True

    depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    map = nodes.new(type="CompositorNodeMapRange")
    depth_file_output.label = 'Depth Output'
    depth_file_output.name = 'Depth Output'
    depth_file_output.format.file_format = 'OPEN_EXR'
    depth_file_output.format.color_depth = '32'
    depth_file_output.format.exr_codec = 'ZIP'
    depth_file_output.base_path = "/"

    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    map.inputs['From Min'].default_value = 0
    map.inputs['From Max'].default_value = DEPTH_SCALE
    map.inputs['To Min'].default_value = 0
    map.inputs['To Max'].default_value = 1
    links.new(render_layers.outputs['Depth'], map.inputs[0])
    links.new(map.outputs[0], depth_file_output.inputs[0])

    # Create normal output nodes
    scale_node = nodes.new(type="CompositorNodeMixRGB")
    scale_node.blend_type = "MULTIPLY"
    # scale_node.use_alpha = True
    scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs["Normal"], scale_node.inputs[1])

    bias_node = nodes.new(type="CompositorNodeMixRGB")
    bias_node.blend_type = "ADD"
    # bias_node.use_alpha = True
    bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_node.outputs[0], bias_node.inputs[1])

    alpha_normal = nodes.new(type="CompositorNodeSetAlpha")
    links.new(bias_node.outputs[0], alpha_normal.inputs["Image"])
    links.new(render_layers.outputs["Alpha"], alpha_normal.inputs["Alpha"])
    
    normal_file_output = nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = "Normal Output"
    normal_file_output.base_path = "/"
    normal_file_output.file_slots[0].use_node_format = True
    normal_file_output.format.file_format = "PNG"    
    links.new(alpha_normal.outputs["Image"], normal_file_output.inputs[0])

    # Create albedo output nodes
    alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
    links.new(render_layers.outputs["DiffCol"], alpha_albedo.inputs["Image"])
    links.new(render_layers.outputs["Alpha"], alpha_albedo.inputs["Alpha"])

    albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = "Albedo Output"
    albedo_file_output.base_path = "/"
    # albedo_file_output.file_slots[0].use_node_format = True
    albedo_file_output.format.file_format = "PNG"
    albedo_file_output.format.color_mode = "RGB"
    albedo_file_output.format.color_depth = "8"
    links.new(alpha_albedo.outputs["Image"], albedo_file_output.inputs[0])
    
    # scene.view_settings.view_transform = 'Raw'
  
    return depth_file_output, normal_file_output, albedo_file_output

# load the glb model
def load_object(object_path: str) -> None:
    try:
        """Loads a glb model into the scene."""
        if object_path.endswith(".glb"):
            bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
        elif object_path.endswith(".fbx"):
            bpy.ops.import_scene.fbx(filepath=object_path)
        else:
            raise ValueError(f"Unsupported file type: {object_path}")
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    except:
        os.system(f'echo "{object_path}" >> {args.output_dir}/bug.txt')
    return mesh_objects

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        # Avoid that it scale CAMERA
        # In Neural Gaffer, they use decompose to let CAMERA World Pose to Rotation Euler and Location, then transfer to Matrix form
        if not obj.parent and obj.type not in {"CAMERA", "LIGHT"}:
            # print(obj)
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    
    
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    # return True

def pattern_file_exists(pattern: str) -> bool:
    """
    Check if any albedo file exists for index `j` in `save_path`.
    
    Matches files like: {save_path}/{j:03d}_albedo_*.png, .exr, .jpg, etc.
    
    Args:
        save_path (str): Directory path to search in.
        j (int): Index (will be formatted as 3-digit zero-padded number).
    
    Returns:
        bool: True if at least one matching file exists, False otherwise.
    """
    return bool(glob(pattern))

def main(args):
    if args.timing:
        start_time = time.perf_counter()
    depth_file_output, normal_file_output, albedo_file_output = reset_scene()
    object_name = args.object_name
    # with open(args.objaverse_info, 'r') as file:
    #     object_info = json.load(file)[object_name]
    os.makedirs(os.path.join(args.output_dir, 'temp'),exist_ok=True)
    TEMP_PATH = os.path.join(args.output_dir, 'temp', f'temp_{object_name}.glb')
    # print(0)
    download(object_name, TEMP_PATH)
    # print(0)
    load_object(TEMP_PATH)
    remove_temp_file(TEMP_PATH)
    normalize_scene()

    camera = bpy.context.scene.camera
    # cam_constraint = camera.constraints.new(type="TRACK_TO")
    # cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    # cam_constraint.up_axis = "UP_Y"

    seed = int(hashlib.sha256(object_name.encode()).hexdigest(), 16) % (2**32)
    lightings = {}
    for split in args.lighting_split:
        lightings[split] = get_lighting(seed, LIGHTING_INFO.replace("[split]", split), N=N_LIGHTINGS)
    views = {}
    for split in args.view_split:
        views[split] = get_view(seed, VIEW_INFO.replace("[split]", split), N=N_VIEWS[split])
    infos = {}
    infos['basic'] = \
        {
            "object_name": object_name,
            "focal": camera.data.lens,
            "sensor_size": [camera.data.sensor_width, camera.data.sensor_width] ,
            "image_size": [render.resolution_x, render.resolution_y],
            "lighting": lightings,
            "view": views,
            "depth_scale": DEPTH_SCALE,
            "object_scale": args.scale,
            "camera_angle_x": 2.0 * math.atan(camera.data.sensor_width / (2.0 * camera.data.lens))
        }
    infos['images'] = []
    
    with open(ALL_VIEWS, 'r') as f:
        all_views = json.load(f)

    for i, lighting in enumerate([item for sublist in lightings.values() for item in sublist]):
        add_light_env(os.path.join(LIGHTING_DIR, lighting))
        for j, view in enumerate([item for sublist in views.values() for item in sublist]):
            save_path = os.path.join(args.output_dir, object_name)

            # transform = np.load(os.path.join(VIEW_DIR, view))
            # We used to load the view transform from a .npy file, but now we load it from the all_views JSON.
            # view = "V{view:03d}.npy".format(view=j)
            view_index = int(view.replace("V", "").replace(".npy", ""))
            transform = all_views[view_index]["transform"]

            # Validate matrix shape
            if transform.shape != (4, 4):
                raise ValueError(f"Expected 4x4 pose matrix, got shape {transform.shape}")

            camera.matrix_world = Matrix(transform.tolist())

            # Force scene update to propagate transform to camera
            bpy.context.view_layer.update()

            assert np.allclose(transform, np.array(camera.matrix_world), atol=1e-6), \
                f"Pose mismatch:\nComputed:\n{transform}\nCamera:\n{np.array(camera.matrix_world)}"
            image_stem = view.split('.')[0] + '-'+ lighting.replace('/', '_').split('.')[0]

            infos['images'].append(
                        {   
                            "object_name": args.object_name,
                            "lighting_split": find_key_by_value(lightings, lighting),
                            "view_split": find_key_by_value(views, view),
                            "image_stem": image_stem,
                            'view': view,
                            'lighting': lighting,
                            "transform": copy.deepcopy(transform.tolist()),
                        }
                    )
            
            # image: 001_003_image.png
            if args.no_rgb:
                scene.render.filepath = ''  
                if i > 0: # not write depth or albedo anymore
                    continue
            else:
                scene.render.filepath = os.path.join(save_path, f'{image_stem}_image')

            if args.skip_exist and os.path.exists(scene.render.filepath+'.png'):
                img = cv2.imread(scene.render.filepath+'.png')
                if img is not None:
                    height, width, _ = img.shape
                    if (width, height) == (render.resolution_x, render.resolution_y):
                        if i == 0:
                            normal_exist = args.normal and pattern_file_exists(os.path.join(save_path, f'{view}_normal_*.png'))
                            albedo_exist = args.albedo and pattern_file_exists(os.path.join(save_path, f'{view}_albedo_*.png'))
                            depth_exist = args.depth and pattern_file_exists(os.path.join(save_path, f'{view}_depth_*.png'))
                            if normal_exist and albedo_exist and depth_exist:
                                continue                               
                        else:
                            continue

            # normal: V1_normal_*.png
            # albedo: V1_albedo_*.png
            # depth: V1_depth_*.exr
            if i == 0:
                view_stem = view.split('.')[0]
                if args.normal:
                    normal_file_output.file_slots[0].use_node_format = True
                    normal_file_output.file_slots[0].path = \
                        os.path.join(save_path, f'{view_stem}_normal_')
                if args.albedo:
                    albedo_file_output.file_slots[0].use_node_format = True
                    albedo_file_output.file_slots[0].path = \
                        os.path.join(save_path, f'{view_stem}_albedo_')
                if args.depth:
                    depth_file_output.file_slots[0].use_node_format = True
                    depth_file_output.file_slots[0].path = \
                        os.path.join(save_path, f'{view_stem}_depth_')
            else:
                bpy.context.view_layer.use_pass_normal = False
                bpy.context.view_layer.use_pass_diffuse_color = False
                bpy.context.view_layer.use_pass_z = False

                depth_file_output.file_slots[0].use_node_format = False
                depth_file_output.file_slots[0].path = ''

                normal_file_output.file_slots[0].use_node_format = False
                normal_file_output.file_slots[0].path = ''

                albedo_file_output.file_slots[0].use_node_format = False
                albedo_file_output.file_slots[0].path = ''

                nodes = bpy.context.scene.node_tree.nodes
            bpy.ops.render.render(write_still=True)
    
    json_output = json.dumps(infos, indent=4)
    os.makedirs(os.path.join(args.output_dir, object_name), exist_ok=True)
    output_file_path = os.path.join(args.output_dir, object_name, "info.json")  # 修改为你的文件路径
    with open(output_file_path, 'w') as json_file:
        json_file.write(json_output)
    print("Dataset info. have been save to", output_file_path)
    if args.timing:
        return time.perf_counter() - start_time
    else:
        return None

def download(uid, tmp):
    objects = objaverse.load_objects(
        uids=[uid],
        download_processes=1
        )
    
    file_path = objects[uid]
    if os.path.isfile(file_path): 
        shutil.copy(file_path, tmp)
        os.remove(file_path)
        

def remove_temp_file(tmp):
    if os.path.isfile(tmp): 
        os.remove(tmp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_name",
        type=str,
        required=True,
        help="Object UID",
    )
    parser.add_argument(
        "--objaverse_info",
        type=str,
        default='../../dataset/Objaverse/objects.json',
        help="Path to the Objaverse info json",
    )
    # parsser.add_argument("--output_dir", type=str, default="{args.output_dir}/views_whole_sphere")
    parser.add_argument("--output_dir", type=str, default="./rendered")

    parser.add_argument(
        "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
    )
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--image_size", type=tuple, default=(512, 512))
    
    parser.add_argument("--timing", action='store_true')
    parser.add_argument("--depth", action='store_true')
    parser.add_argument("--normal", action='store_true')
    parser.add_argument("--albedo", action='store_true')
    parser.add_argument("--no_rgb", action='store_true')

    parser.add_argument(
        "--lighting_split",
        type=str,
        nargs="+",  # Accept 1+ space-separated values
        default=["training"],  # Default as list for consistency
        help="Lighting split(s) to use (e.g., 'training', 'validation', 'test')"
    )
    parser.add_argument(
        "--view_split",
        type=str,
        nargs="+",
        default=["training"],
        help="View split(s) to use (e.g., 'training', 'validation', 'test')"
    )


    parser.add_argument("--skip_exist", action='store_true')

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    print('===================', args.engine, '===================')

    # cam = scene.objects["Camera"]
    # cam.location = (0, 1.2, 0)
    # cam.data.lens = 35
    # cam.data.sensor_width = 32

    # cam_constraint = cam.constraints.new(type="TRACK_TO")
    # cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    # cam_constraint.up_axis = "UP_Y"

    # camera = bpy.context.scene.camera
    # camera.location = (0,1.5,0)

    # cam_constraint = camera.constraints.new(type="TRACK_TO")
    # cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    # cam_constraint.up_axis = "UP_Y"


    render.engine = args.engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = args.image_size[0]
    render.resolution_y = args.image_size[0]
    render.resolution_percentage = 100

    scene.cycles.device = "GPU"
    scene.cycles.samples = 128
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True

    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    # Set the device_type

    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" # or "OPENCL"
    args.output_dir = os.path.abspath(args.output_dir)

    timing = main(args)
    if timing is not None:
        print(timing)

# CUDA_VISIBLE_DEVICES=0 ./blender/blender-4.3.2-linux-x64/blender -b -P ./blender/blender_script.py -- --object_name 2dca2c61c96b40bd885e46eae324fc62 --output_dir '/home/user/Documents/research/LavalObjaverseDataset/rendered/training/subset_0' --depth --lighting_split training --view_split training  --skip_exist