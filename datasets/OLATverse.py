"""
OLATverse relighting evaluation dataset (rewritten).

Changes vs. the previous version
--------------------------------
1. Works with the NEW mapping JSON produced by gen_test_mapping.py:
   for every object, ALL (source, target) lighting combinations are
   enumerated (src may equal tgt), each entry carrying a fixed list of
   32 views:
       {
           "object":          "data-xxxx-Cxxx",
           "source_lighting": "red-bedroom",
           "target_lighting": "sunset",
           "view":            ["Cam01", "Cam06", ...],   # 32 views
           "idx":             0
       }

2. Camera parameters (extrinsics + intrinsics) are now read from the
   per-object / per-lighting transforms_test.json instead of all_cam.json:

       {data_dir}/preprocessed/{PREPROCESS_SUBFOLDER}/
           {object_name}/{lighting_name}/transforms_test.json

   Each frame in that file looks like:
       {
           "file_path": "./{object}_{lighting}_{CamXX}.png",
           "transform_matrix":  [[...], ...],   # 4x4 c2w, Blender convention
           "camera_intrinsics": [cx, cy, fx, fy]
       }

   The intrinsics are ALREADY expressed in the preprocessed image space
   (the images stored next to transforms_test.json), so the elaborate
   crop_and_pad_to_square replication of the old code is no longer
   needed — we only rescale K for the final resize to self.resolution.
"""

import os
import json
import math
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import OpenEXR
import Imath
import cv2
from pathlib import Path
import imagecodecs
from .utils import *

SCALE = 1_000
PREPROCESS_SUBFOLDER = "relight_static_0.1"


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB → linear conversion (vectorised, float32)."""
    return np.where(x <= 0.04045,
                    x / 12.92,
                    ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def resize(image, resolution, mode='bilinear'):
    """Resize a [C, H, W] tensor to the given (W, H) resolution tuple."""
    h, w = resolution[1], resolution[0]
    if mode == 'nearest':
        return TF.resize(image, [h, w], interpolation=TF.InterpolationMode.NEAREST)
    return TF.resize(image, [h, w], interpolation=TF.InterpolationMode.BILINEAR)


class OLATverseEvalDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir='/media/SSD1/hejun/Relighting/OLATverse',
                 pair_info='/media/SSD1/hejun/Relighting/OLATverse/data/experimental_pair/32_to_32_mapping.json',
                 black_background=True,
                 resolution=(256, 256),
                 seed=180,
                 is_train=False,
                 **kwargs):
        super().__init__()

        self.data_dir = data_dir
        self.pair_info = pair_info
        self.black_background = black_background
        self.resolution = resolution
        self.seed = seed
        self.is_train = is_train
        self.scale = kwargs.pop("scale", 2.0)
        self.background = (torch.tensor([0.0, 0.0, 0.0]) if black_background
                           else torch.tensor([1.0, 1.0, 1.0]))

        with open(pair_info) as f:
            self.data_pairs = json.load(f)

        self._cam_cache = {}

    def __len__(self):
        return len(self.data_pairs)

    # ------------------------------------------------------------------
    # Camera loading from transforms_test.json
    # ------------------------------------------------------------------
    def _load_transforms(self, object_name):
        """
        Load and cache the per-object / per-lighting camera file:

            {data_dir}/preprocessed/{PREPROCESS_SUBFOLDER}/
                {object_name}/{lighting_name}/transforms_test.json

        Returns
        -------
        dict : {view_name: frame_dict}
            view_name (e.g. "Cam01") is parsed from the trailing token of
            file_path: "./{object}_{lighting}_{CamXX}.png" -> "CamXX".
        """
        if object_name in self._cam_cache:
            return self._cam_cache[object_name]

        transforms_path = os.path.join(
            self.data_dir, 'data/preprocessed', PREPROCESS_SUBFOLDER,
            object_name, 'transforms_test.json')

        with open(transforms_path) as f:
            data = json.load(f)

        cam_dict = {}
        for frame in data["frames"]:
            # "./data-040325-C028_red-bedroom_Cam01.png" -> "Cam01"
            stem = os.path.splitext(os.path.basename(frame["file_path"]))[0]
            view_name = stem.split('_')[-1]
            cam_dict[view_name] = frame

        self._cam_cache[object_name] = cam_dict
        return cam_dict

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        data_pair = self.data_pairs[idx]
        object_name     = data_pair["object"]
        source_lighting = data_pair["source_lighting"]
        target_lighting = data_pair["target_lighting"]
        views           = data_pair["view"]          # list of 32 view names

        item = self._fetch_one_pair(object_name,
                                    source_lighting, target_lighting,
                                    views)
        item["idx"] = idx  # record data pair index
        return item

    # ------------------------------------------------------------------
    def _fetch_one_pair(self, object_name, source_lighting, target_lighting,
                        selected_views):

        # The mapping JSON stores a single shared view list used for both
        # source and target lightings.
        source_view_name = target_view_name = selected_views

        source_lighting_name = source_lighting
        target_lighting_name = target_lighting

        # ==============================================================
        # Inner helper 1: Load HDR environment map from .exr
        # ==============================================================
        def _fetch_lightings(lighting_name, addition_rotation=None):
            """
            Load an HDR environment map from:
                {data_dir}/lightings/{lighting_name}.exr

            Returns
            -------
            lightings : torch.Tensor  [1, 3, H_env, W_env]
            rays      : torch.Tensor  [6, 1, H_env, W_env]
                        Per-pixel unit-sphere directions (dx, dy, dz) and
                        their squares, packed for network consumption.
            """
            exr_path = os.path.join(self.data_dir, 'data/lightings',
                                    f"{lighting_name}.exr")

            raw = read_hdr(exr_path, self.resolution)
            if not isinstance(raw, torch.Tensor):
                raw = torch.from_numpy(np.array(raw, dtype=np.float32))

            lightings = resize(raw.permute(2, 0, 1).contiguous(),
                               self.resolution).unsqueeze(0)

            return lightings

        # ==============================================================
        # Inner helper 2: Load images, masks, intrinsics, extrinsics
        # ==============================================================

        def _fetch_images_and_masks(lighting_name, view_name_list):
            """
            For each camera view, load the preprocessed RGBA image and its mask,
            resize both to self.resolution, and build the camera intrinsic
            matrix K from the aggregated transforms_test.json.

            Image path  : {data_dir}/preprocessed/{PREPROCESS_SUBFOLDER}/
                        {object_name}/{lighting_name}/
                        {object_name}_{lighting_name}_{view_name}.png
                        
            Camera info : {data_dir}/preprocessed/{PREPROCESS_SUBFOLDER}/
                        {object_name}/transforms_test.json  <-- UPDATED PATH
                        (c2w in Blender convention; intrinsics
                        [cx, cy, fx, fy] in the preprocessed image space)

            Returns
            -------
            images : [N, 3, H, W]
            masks  : [N, 1, H, W]
            Ks     : [N, 3, 3]   intrinsics rescaled to self.resolution
            views  : [N, 4, 4]   c2w extrinsics (converted to OpenCV in view_preprocess)
            """
            
            # 1. Load the AGGREGATED transforms_test.json from the object level
            json_path = os.path.join(
                self.data_dir, 'data/preprocessed', PREPROCESS_SUBFOLDER, 
                object_name, "transforms_test.json"
            )
            
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Transforms JSON not found at: {json_path}")
                
            with open(json_path, 'r') as f:
                transforms_data = json.load(f)
            
            # Build a dictionary mapping view_name to its camera data
            cam_dict = {}
            for frame in transforms_data.get("frames", []):
                file_path = frame.get("file_path", "")
                # Extract view_name from path like "./street/obj_street_Cam01.png" -> "Cam01"
                view_name = os.path.basename(file_path).split('.')[0].split('_')[-1]
                cam_dict[view_name] = {
                    "transform_matrix": frame.get("transform_matrix"),
                    "camera_intrinsics": frame.get("camera_intrinsics")
                }

            images, masks, Ks, views = [], [], [], []

            for view_name in view_name_list:
                # 2. Load preprocessed RGBA image ----------------------------
                img_path = os.path.join(
                    self.data_dir, 'data/preprocessed', PREPROCESS_SUBFOLDER,
                    object_name, lighting_name,
                    f"{object_name}_{lighting_name}_{view_name}.png"
                )

                # USE CV2 to read RGBA (IMREAD_UNCHANGED preserves the 4th channel)
                raw_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2RGBA)
                if raw_img is None:
                    raise FileNotFoundError(f"Failed to load image: {img_path}")

                # Split into RGB and Alpha channels
                if raw_img.shape[2] == 4:
                    rgb_np = raw_img[:, :, :3]
                    alpha_np = raw_img[:, :, 3:]
                else:
                    raise ValueError("Image does not have 4 channels (BGRA)")
                    # Fallback: if somehow not 4 channels, assume RGB and full opacity mask
                    # rgb_np = raw_img
                    # alpha_np = np.ones((*raw_img.shape[:2], 1), dtype=raw_img.dtype) * 255

                # Normalize to [0, 1] float32
                rgb_np = rgb_np.astype(np.float32) / 255.0
                alpha_np = alpha_np.astype(np.float32) / 255.0

                # Convert to PyTorch tensors: [3, H, W] for image, [1, H, W] for mask
                image = torch.from_numpy(rgb_np).permute(2, 0, 1)
                mask = torch.from_numpy(alpha_np).permute(2, 0, 1)

                # 3. Resize to target resolution ------------------------
                # Note: If your `resize` is torch.nn.functional.interpolate, 
                # you may need to do: resize(image.unsqueeze(0), self.resolution).squeeze(0)
                image = resize(image.contiguous(), self.resolution)
                mask  = resize(mask.contiguous(), self.resolution, mode='nearest')

                # 4. Camera extrinsics (c2w, Blender convention) ---------
                if view_name not in cam_dict:
                    raise KeyError(
                        f"View '{view_name}' not found in transforms_test.json "
                        f"for object={object_name}, lighting={lighting_name}")
                
                cam_data = cam_dict[view_name]
                c2w = torch.tensor(cam_data["transform_matrix"], dtype=torch.float32)
                views.append(c2w)

                # 5. Camera intrinsics -----------------------------------
                cx, cy, fx, fy = cam_data["camera_intrinsics"]

                K = torch.tensor([
                    [fx, 0,  cx],
                    [0,  fy, cy],
                    [0,  0,  1 ]
                ], dtype=torch.float32)

                images.append(image)
                masks.append(mask)
                Ks.append(K)

            # Stack into batches
            images = torch.stack(images, dim=0)   # [N, 3, H, W]
            masks  = torch.stack(masks,  dim=0)   # [N, 1, H, W]
            Ks     = torch.stack(Ks,     dim=0)   # [N, 3, 3]
            views  = torch.stack(views,  dim=0)   # [N, 4, 4]

            # Alpha compositing over background
            images = images * masks + self.background.reshape(1, 3, 1, 1) * (1 - masks)
            
            return images, masks, Ks, views

        # ==============================================================
        # Rotation augmentation (training only)
        # ==============================================================
        if self.is_train:
            # addition_rotation = torch.tensor(
            #     self.get_random_rotation()[0].as_matrix(), dtype=torch.float32)
            addition_rotation = torch.eye(3, dtype=torch.float32)  # fallback
        else:
            addition_rotation = torch.eye(3, dtype=torch.float32)

        # ==============================================================
        # Fetch all data
        # ==============================================================
        source_images, source_mask, source_Ks, source_view = \
            _fetch_images_and_masks(source_lighting_name, source_view_name)
        target_images, target_mask, target_Ks, target_view = \
            _fetch_images_and_masks(target_lighting_name, target_view_name)

        # fetch environment map
        source_lighting = _fetch_lightings(source_lighting_name, addition_rotation)
        target_lighting = _fetch_lightings(target_lighting_name, addition_rotation)

        # Preprocess views (coordinate conversion + scaling)
        all_views = self.view_preprocess(torch.cat([source_view, target_view], dim=0))
        num_src_views = source_view.size(0)
        source_view = all_views[:num_src_views]
        target_view = all_views[num_src_views:]
        return_dict = {
            "source_lighting": source_lighting,
            "target_lighting": target_lighting,

            "source_images":   source_images,
            "target_images":   target_images,

            "source_view":     source_view,
            "target_view":     target_view,

            "source_depths":   torch.zeros_like(source_mask),
            "target_depths":   torch.zeros_like(target_mask),

            "source_mask":     source_mask,
            "target_mask":     target_mask,

            "source_Ks":       source_Ks,
            "target_Ks":       target_Ks,

            "addition_rotation": addition_rotation,
        }

        return return_dict

    # ------------------------------------------------------------------
    def view_preprocess(self, view):
        # view shape: (N, 4, 4)

        # 1. Blender -> CV/OpenCV coordinate transformation matrix
        #    (transforms_test.json stores c2w in Blender convention)
        blender_to_cv = torch.tensor([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ], dtype=view.dtype, device=view.device)

        # Batched transformation: (N, 4, 4) @ (4, 4) -> (N, 4, 4)
        view = view @ blender_to_cv

        # 2. Normalize translations so the average length equals self.scale
        translations = view[:, :3, 3]  # Shape: (N, 3)
        avg_len = torch.norm(translations, dim=1).mean()  # Scalar
        # Global scaling factor (+1e-8 prevents division by zero)
        scale_factor = self.scale / (avg_len + 1e-8)

        # Apply scaling
        view[:, :3, 3] = translations * scale_factor

        return view

class OLATversePointLightEvalDataset(OLATverseEvalDataset):
    def __init__(self,
                 data_dir='/media/SSD1/hejun/Relighting/OLATverse',
                 pair_info='/media/SSD1/hejun/Relighting/OLATverse/data/experimental_pair/all_to_all_pl_mapping.json',
                 black_background=True,
                 resolution=(256, 256),
                 seed=180,
                 is_train=False,
                 **kwargs):

        self.data_dir = data_dir
        self.pair_info = pair_info
        self.black_background = black_background
        self.resolution = resolution
        self.seed = seed
        self.is_train = is_train
        self.scale = kwargs.pop("scale", 2.0)
        self.background = (torch.tensor([0.0, 0.0, 0.0]) if black_background
                           else torch.tensor([1.0, 1.0, 1.0]))

        with open(pair_info) as f:
            self.data_pairs = json.load(f)
        all_lights_json_path = os.path.join(self.data_dir, 'OLATverse', 'shared', 'all_lights.json')
        with open(all_lights_json_path) as f:
            _lights = json.load(f)
        self.light_idx_to_uid = {
            frame["light_idx"]: frame["file_path"].split('.')[-1]
            for frame in _lights["frames"]
        }

    # ------------------------------------------------------------------
    def _fetch_one_pair(self, object_name, source_lighting, target_lighting,
                        selected_views):

        # ---- Resolve source/target view lists -------------------------
        view_crop_mapping = {}
        if isinstance(selected_views, tuple):
            source_view_name, target_view_name = selected_views
        else:
            source_view_name = target_view_name = selected_views

        # ---- Load per-object camera info once -------------------------
        cam_info_path = os.path.join(self.data_dir, 'data/OLATverse_Upload_Val',
                                     object_name, 'all_cam.json')
        with open(cam_info_path) as f:
            cam_info_raw = json.load(f)
        # The JSON root is {"frames": [{"cam_idx": "Cam01", ...}, ...]}
        # Build a quick-lookup dict keyed by cam_idx for O(1) access.
        cam_info_dict = {cam["cam_idx"]: cam for cam in cam_info_raw["frames"]}

        source_lighting_name = source_lighting
        target_lighting_name = target_lighting

        # ==============================================================
        # Inner helper 1: Load HDR environment map from .exr
        # ==============================================================
        def _fetch_lightings(lighting_name, addition_rotation=None):
            """
            Load an environment map from:
                /media/SSD1/hejun/Relighting/OLATverse/OLATverse/shared/
                envmap_zspiral_mpi/{lighting_name:03d}.png

            lighting_name : int  — the light_idx recorded in the mapping JSON.

            Returns
            -------
            lightings : torch.Tensor  [1, H_env, W_env, 3]
                        Float32, values in [0, 1] (standard PNG range).
            rays      : torch.Tensor  [6, 1, H_env, W_env]
                        Per-pixel unit-sphere directions (dx, dy, dz) and
                        their squares, packed for network consumption.
            """
            png_path = os.path.join(
                f'{self.data_dir}/OLATverse/shared',
                'envmap_zspiral_mpi',
                f"{lighting_name:03d}.png"
            )

            raw = TF.to_tensor(Image.open(png_path).convert("RGB"))  # [3, H, W], float32 in [0,1]
            lightings = resize(raw.contiguous(), self.resolution).unsqueeze(0)  # [1, 3, H, W]

            return lightings, None

        # ==============================================================
        # Inner helper 2: Load images, masks, intrinsics, extrinsics
        # ==============================================================
        def _fetch_images_and_masks(lighting_name, view_name_list):
            """
            For each camera view, load the rendered AVIF image and its mask PNG,
            centre-crop both to square using the shorter side of the IMAGE,
            resize to self.resolution, and compute K.

            Image path : {data_dir}/data/OLATverse_Upload_Val/
                         {object_name}/masked_olat/{view_name}/
                         *.{lighting_uid}.avif   (unique match)
            Mask path  : {data_dir}/data/OLATverse_Upload_Val/
                         {object_name}/mask/{view_name}.png
            Camera pose: all_cam.json (loaded once above)

            Pipeline:
              (a) Decode AVIF → uint8 RGB  [H, W, 3]
              (b) Load mask PNG via PIL → numpy [H_m, W_m]
                  NOTE: PIL.size = (W, H), numpy.shape = (H, W) — convert carefully.
                  If mask size differs from image, resize mask to match image first.
              (c) Centre-crop BOTH to square: side = min(H_img, W_img)
              (d) Resize square to self.resolution (INTER_AREA for image, INTER_NEAREST for mask)
              (e) Convert image to float32 [0, 1]

            Returns
            -------
            images : [N, 3, H, W]
            masks  : [N, 1, H, W]
            Ks     : [N, 3, 3]   adjusted intrinsic matrices
            views  : [N, 4, 4]   c2w extrinsic matrices (OpenCV convention)
            """

            # ---- Resolve lighting_uid via pre-cached lookup table ------------
            lighting_uid = self.light_idx_to_uid[lighting_name]   # e.g. "000025"

            images, masks, Ks, views = [], [], [], []

            for view_name in view_name_list:

                # ==============================================================
                # Step A: Locate and decode AVIF → uint8 RGB
                # ==============================================================
                avif_dir = os.path.join(
                    self.data_dir, 'data/OLATverse_Upload_Val',
                    object_name, 'masked_olat', view_name)

                matches = list(Path(avif_dir).glob(f"*.{lighting_uid}.avif"))
                if len(matches) == 0:
                    raise FileNotFoundError(
                        f"No AVIF found matching *.{lighting_uid}.avif in {avif_dir}")
                if len(matches) > 1:
                    raise RuntimeError(
                        f"Multiple AVIFs matched *.{lighting_uid}.avif in {avif_dir}: {matches}")

                raw = imagecodecs.avif_decode(open(str(matches[0]), 'rb').read())
                if raw.ndim == 3 and raw.shape[2] == 4:
                    raw = raw[..., :3]          # drop alpha if RGBA
                # raw: [H, W, 3] uint8 RGB, background = (0, 0, 0)

                h_raw, w_raw = raw.shape[:2]   # numpy convention: shape = (H, W, C)

                # ==============================================================
                # Step B: Load mask PNG → numpy [H, W] uint8
                # PIL.size returns (W, H); numpy.shape returns (H, W).
                # If mask spatial size differs from image, resize mask to match.
                # ==============================================================
                mask_path = os.path.join(
                    self.data_dir, 'OLATverse_Upload_Val',
                    object_name, 'mask', f"{view_name}.png")

                if os.path.exists(mask_path):
                    mask_pil = Image.open(mask_path).convert("L")   # grayscale
                    # PIL.size = (W_m, H_m)  ←→  numpy.shape = (H_m, W_m)
                    w_m, h_m = mask_pil.size                         # PIL convention

                    if (w_m, h_m) != (w_raw, h_raw):
                        # Resize mask to match image dimensions.
                        # PIL.resize takes (W, H) — same as PIL.size convention.
                        mask_pil = mask_pil.resize((w_raw, h_raw), Image.NEAREST)

                    mask_np = np.array(mask_pil, dtype=np.uint8)     # [H, W] numpy convention
                else:
                    # No mask file: treat entire image as foreground
                    mask_np = np.full((h_raw, w_raw), 255, dtype=np.uint8)

                # ==============================================================
                # Step C: Centre-crop BOTH image and mask to square
                # side = min(H_img, W_img); crop is centred on the image.
                # ==============================================================
                side = min(h_raw, w_raw)
                top  = (h_raw - side) // 2
                left = (w_raw - side) // 2

                img_sq  = raw[top:top + side, left:left + side]       # [side, side, 3] uint8
                mask_sq = mask_np[top:top + side, left:left + side]   # [side, side] uint8

                # ==============================================================
                # Step D: Resize square → self.resolution
                # ==============================================================
                res_h, res_w = self.resolution[1], self.resolution[0]

                img_out  = cv2.resize(img_sq,  (res_w, res_h), interpolation=cv2.INTER_AREA)
                mask_out = cv2.resize(mask_sq, (res_w, res_h), interpolation=cv2.INTER_NEAREST)

                # ==============================================================
                # Step E: Convert to tensors
                # ==============================================================
                image = torch.from_numpy(
                    img_out.astype(np.float32) / 255.0
                ).permute(2, 0, 1).contiguous()                       # [3, H, W]

                mask = TF.to_tensor(Image.fromarray(mask_out))        # [1, H, W] in [0, 1]

                # ==============================================================
                # Camera extrinsics (c2w)
                # ==============================================================
                cam_data = cam_info_dict[view_name]
                c2w = torch.tensor(cam_data["transform_matrix"], dtype=torch.float32)
                views.append(c2w)

                # ==============================================================
                # Camera intrinsics — propagate centre-crop + resize into K
                # ==============================================================
                cx_orig, cy_orig, fx_orig, fy_orig = cam_data["camera_intrinsics"]

                # Step C: centre-crop shifts the principal point
                cx_crop = cx_orig - left
                cy_crop = cy_orig - top
                # fx, fy unchanged by a pure crop

                # Step D: resize square (side × side) → (res_w × res_h)
                scale    = res_w / side
                fx_final = fx_orig * scale
                fy_final = fy_orig * scale
                cx_final = cx_crop * scale
                cy_final = cy_crop * scale

                K = torch.tensor([
                    [fx_final, 0,        cx_final],
                    [0,        fy_final, cy_final],
                    [0,        0,        1       ]
                ], dtype=torch.float32)

                images.append(image)
                masks.append(mask)
                Ks.append(K)

            images = torch.stack(images, dim=0)                       # [N, 3, H, W]
            masks  = torch.stack(masks,  dim=0)                       # [N, 1, H, W]
            Ks     = torch.stack(Ks,     dim=0)                       # [N, 3, 3]
            views  = self.view_preprocess(torch.stack(views, dim=0))  # [N, 4, 4]

            images = images * masks + self.background.reshape(1, 3, 1, 1) * (1 - masks)
            return images, masks, Ks, views

        # ==============================================================
        # Rotation augmentation (training only)
        # ==============================================================
        if self.is_train:
            # get_random_rotation() is expected from the parent class or utility module.
            # It returns a scipy Rotation object; convert to a [3, 3] torch tensor.
            # addition_rotation = torch.tensor(
            #     self.get_random_rotation()[0].as_matrix(), dtype=torch.float32)
            addition_rotation = torch.eye(3, dtype=torch.float32)  # fallback
        else:
            addition_rotation = torch.eye(3, dtype=torch.float32)

        # ==============================================================
        # Fetch all data
        # ==============================================================
        source_images, source_mask, source_Ks, source_view = \
            _fetch_images_and_masks(source_lighting_name, source_view_name)
        target_images, target_mask, target_Ks, target_view = \
            _fetch_images_and_masks(target_lighting_name, target_view_name)

        # fetch environment map
        source_lighting, _ = _fetch_lightings(source_lighting_name, addition_rotation)
        target_lighting, _             = _fetch_lightings(target_lighting_name, addition_rotation)

        return_dict = {
            "source_lighting": source_lighting,
            "target_lighting": target_lighting,

            "source_images":   source_images,
            "target_images":   target_images,

            "source_view":     source_view,
            "target_view":     target_view,

            "source_mask":     source_mask,
            "target_mask":     target_mask,

            "source_Ks":       source_Ks,
            "target_Ks":       target_Ks,

            "addition_rotation": addition_rotation,
        }

        return return_dict