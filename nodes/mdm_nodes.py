"""
ComfyUI-MotionDiffusionModel Node Definitions

Text-to-motion generation using Motion Diffusion Model (MDM) and
SMPL skeleton fitting for humanoid meshes.

Based on: Human Motion Diffusion Model (ICLR 2023)
Repository: https://github.com/GuyTevet/motion-diffusion-model
"""

import os
import json
import numpy as np
import torch
from argparse import Namespace
from PIL import Image
import folder_paths

# Package root for data files
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDM_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mdm")

# Use ComfyUI's models directory for MDM models
COMFYUI_MODELS_DIR = folder_paths.models_dir
MDM_MODELS_DIR = os.path.join(COMFYUI_MODELS_DIR, "motion")

# ============================================================================
# SMPL Skeleton Utilities
# ============================================================================

SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]

SMPL_PARENT_INDICES = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
]

SMPL_BONE_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
    (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15),
    (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
]

# Canonical SMPL skeleton in T-pose (Y-up, ~1.7m tall)
SMPL_CANONICAL_JOINTS = np.array([
    [0.0, 0.92, 0.0], [0.09, 0.88, 0.0], [-0.09, 0.88, 0.0], [0.0, 1.0, 0.0],
    [0.09, 0.48, 0.0], [-0.09, 0.48, 0.0], [0.0, 1.12, 0.0], [0.09, 0.08, 0.0],
    [-0.09, 0.08, 0.0], [0.0, 1.26, 0.0], [0.09, 0.0, 0.06], [-0.09, 0.0, 0.06],
    [0.0, 1.46, 0.0], [0.06, 1.42, 0.0], [-0.06, 1.42, 0.0], [0.0, 1.62, 0.0],
    [0.18, 1.42, 0.0], [-0.18, 1.42, 0.0], [0.42, 1.42, 0.0], [-0.42, 1.42, 0.0],
    [0.66, 1.42, 0.0], [-0.66, 1.42, 0.0],
], dtype=np.float32)


def get_canonical_skeleton():
    return SMPL_CANONICAL_JOINTS.copy()


def create_skeleton_dict(joints, fps=20.0):
    return {
        "joints": joints,
        "joint_names": SMPL_JOINT_NAMES,
        "parent_indices": SMPL_PARENT_INDICES,
        "bone_connections": SMPL_BONE_CONNECTIONS,
        "fps": fps,
        "num_joints": 22,
    }


# ============================================================================
# MDM Model Utilities
# ============================================================================

class MinimalDataset:
    """Minimal dataset wrapper for inference (provides inv_transform only)"""
    def __init__(self, dataset_name='humanml', device='cpu'):
        self.dataset_name = dataset_name
        dataset_dir = os.path.join(MDM_DATA_PATH, 'dataset')

        if dataset_name == 'humanml':
            self.mean = np.load(os.path.join(dataset_dir, 't2m_mean.npy'))
            self.std = np.load(os.path.join(dataset_dir, 't2m_std.npy'))
        elif dataset_name == 'kit':
            self.mean = np.load(os.path.join(dataset_dir, 'kit_mean.npy'))
            self.std = np.load(os.path.join(dataset_dir, 'kit_std.npy'))
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.t2m_dataset = type('T2MDataset', (), {
            'mean': self.mean,
            'std': self.std,
            'inv_transform': lambda self, data: data * self.std + self.mean
        })()

        # Add dataset attribute for compatibility with get_model_args
        self.dataset = type('Dataset', (), {
            'num_actions': 1  # Default for text-only models (no action conditioning)
        })()


def get_default_args(dataset='humanml'):
    args = Namespace()
    args.cuda = True
    args.device = 0
    args.seed = 10
    args.batch_size = 1
    args.dataset = dataset
    args.data_dir = ""
    args.arch = 'trans_enc'
    args.text_encoder_type = 'clip'
    args.emb_trans_dec = False
    args.layers = 8
    args.latent_dim = 512
    args.cond_mask_prob = 0.1
    args.mask_frames = False
    args.lambda_rcxyz = 0.0
    args.lambda_vel = 0.0
    args.lambda_fc = 0.0
    args.lambda_target_loc = 0.0
    args.unconstrained = False
    args.pos_embed_max_len = 5000
    args.use_ema = True
    args.multi_target_cond = False
    args.multi_encoder_type = 'single'
    args.target_enc_layers = 1
    args.context_len = 0
    args.pred_len = 0
    args.noise_schedule = 'cosine'
    args.diffusion_steps = 1000
    args.sigma_small = True
    args.guidance_param = 2.5
    return args


# ============================================================================
# SMPL Body Model Auto-Download
# ============================================================================

def ensure_smpl_files():
    """Automatically download SMPL body model files if they don't exist"""
    from .mdm.utils.config import SMPL_DATA_PATH, SMPL_MODEL_PATH, JOINT_REGRESSOR_TRAIN_EXTRA

    # Check if required files exist
    required_files = [SMPL_MODEL_PATH, JOINT_REGRESSOR_TRAIN_EXTRA]
    all_exist = all(os.path.exists(f) for f in required_files)

    if all_exist:
        return  # Files already exist, nothing to do

    print("SMPL body model files not found. Downloading...")
    os.makedirs(SMPL_DATA_PATH, exist_ok=True)

    # Download SMPL files from MDM's official Google Drive
    smpl_gdrive_id = "1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2"
    zip_path = os.path.join(SMPL_DATA_PATH, "smpl.zip")

    try:
        # Use gdown for reliable Google Drive downloads
        try:
            import gdown
            print("  Downloading SMPL files from Google Drive...")
            gdown.download(id=smpl_gdrive_id, output=zip_path, quiet=False)
        except ImportError:
            # Fallback to urllib if gdown not available
            import urllib.request
            print("  Warning: gdown not installed, using urllib (less reliable)")
            print("  Install gdown for better downloads: pip install gdown")
            url = f"https://drive.google.com/uc?id={smpl_gdrive_id}"
            urllib.request.urlretrieve(url, zip_path)

        # Extract
        print(f"  Extracting SMPL files to {SMPL_DATA_PATH}...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(SMPL_DATA_PATH)

        # Move files from smpl/ subdirectory to SMPL_DATA_PATH
        smpl_subdir = os.path.join(SMPL_DATA_PATH, "smpl")
        if os.path.exists(smpl_subdir):
            for filename in os.listdir(smpl_subdir):
                src = os.path.join(smpl_subdir, filename)
                dst = os.path.join(SMPL_DATA_PATH, filename)
                if not os.path.exists(dst):
                    os.rename(src, dst)
            os.rmdir(smpl_subdir)

        # Cleanup
        os.remove(zip_path)
        print("  SMPL files downloaded successfully!")

    except Exception as e:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise RuntimeError(f"Failed to download SMPL files: {e}")


# ============================================================================
# NODE: (down)Load MDM Model - Combined Download + Load
# ============================================================================

# MDM model download URLs (from official Google Drive repository)
MDM_MODEL_URLS = {
    "humanml_trans_enc_512": {
        "url": "https://drive.google.com/uc?id=1PE0PK8e5a5j-7-Xhs5YET5U5pGh0c821",
        "description": "Original model (1000 steps, best quality, ~8sec/sample)",
        "folder": "humanml_trans_enc_512",  # Actual extracted folder name
        "gdrive_id": "1PE0PK8e5a5j-7-Xhs5YET5U5pGh0c821",
    },
    "humanml-encoder-512-50steps": {
        "url": "https://drive.google.com/uc?id=1cfadR1eZ116TIdXK7qDX1RugAerEiJXr",
        "description": "Fast model (50 steps, 20x faster ~0.4sec/sample) - RECOMMENDED",
        "folder": "humanml_enc_512_50steps",  # Actual extracted folder name
        "gdrive_id": "1cfadR1eZ116TIdXK7qDX1RugAerEiJXr",
    },
    "kit-encoder-512": {
        "url": "https://drive.google.com/uc?id=1SHCRcE0es31vkJMLGf9dyLe7YsWj7pNL",
        "description": "KIT dataset model",
        "folder": "kit_enc_512",  # Actual extracted folder name (assumed)
        "gdrive_id": "1SHCRcE0es31vkJMLGf9dyLe7YsWj7pNL",
    },
}


class DownLoadMDMModel:
    """Downloads (if needed) and loads MDM model for motion generation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(MDM_MODEL_URLS.keys()), {
                    "default": "humanml_trans_enc_512",
                    "tooltip": "Select MDM model to download/load"
                }),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "use_ema": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use exponential moving average weights (better quality)"
                }),
                "force_redownload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Re-download even if model exists"
                }),
            },
        }

    RETURN_TYPES = ("MDM_PIPELINE",)
    RETURN_NAMES = ("mdm_pipeline",)
    FUNCTION = "download_and_load"
    CATEGORY = "MDM/Model"

    def download_and_load(self, model_name, device, use_ema, force_redownload):
        import zipfile
        from .mdm.utils.model_util import create_model_and_diffusion, load_saved_model

        # Ensure SMPL body model files are available (auto-download if needed)
        ensure_smpl_files()

        model_info = MDM_MODEL_URLS[model_name]
        save_dir = MDM_MODELS_DIR  # Use ComfyUI models/motion/ directory
        os.makedirs(save_dir, exist_ok=True)  # Create if doesn't exist
        model_dir = os.path.join(save_dir, model_info["folder"])

        # Check if model already exists
        model_path = None
        if os.path.exists(model_dir) and not force_redownload:
            pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
            if pt_files:
                model_path = os.path.join(model_dir, pt_files[0])
                print(f"Found existing model: {model_path}")

        # Download if not found
        if model_path is None or force_redownload:
            os.makedirs(save_dir, exist_ok=True)
            zip_path = os.path.join(save_dir, f"{model_name}.zip")

            print(f"Downloading {model_name}...")
            print(f"  Description: {model_info['description']}")

            try:
                # Use gdown for reliable Google Drive downloads
                try:
                    import gdown
                    print(f"  Using gdown for Google Drive download...")
                    gdown.download(id=model_info["gdrive_id"], output=zip_path, quiet=False)
                except ImportError:
                    # Fallback to urllib if gdown not available
                    import urllib.request
                    print(f"  Warning: gdown not installed, using urllib (less reliable)")
                    print(f"  Install gdown for better downloads: pip install gdown")
                    url = model_info["url"]

                    def report_progress(block_num, block_size, total_size):
                        downloaded = block_num * block_size
                        if total_size > 0:
                            percent = min(100, downloaded * 100 / total_size)
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

                    urllib.request.urlretrieve(url, zip_path, reporthook=report_progress)
                    print()  # New line after progress

                # Extract
                print(f"Extracting to {save_dir}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(save_dir)

                os.remove(zip_path)
                print(f"Removed temporary zip file")

                # Find the .pt file
                if not os.path.exists(model_dir):
                    extracted_folders = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
                    for folder in extracted_folders:
                        if model_name.replace('-', '_') in folder or folder.replace('-', '_') in model_name:
                            model_dir = os.path.join(save_dir, folder)
                            break

                pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
                if not pt_files:
                    raise FileNotFoundError(f"No .pt files found in {model_dir}")

                model_path = os.path.join(model_dir, pt_files[0])
                print(f"Download complete: {model_path}")

            except Exception as e:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                raise RuntimeError(f"Failed to download {model_name}: {e}")

        # Load the model
        args_path = os.path.join(os.path.dirname(model_path), 'args.json')
        args = get_default_args()
        if os.path.exists(args_path):
            with open(args_path, 'r') as f:
                saved_args = json.load(f)
            for key, value in saved_args.items():
                if hasattr(args, key):
                    setattr(args, key, value)

        args.use_ema = use_ema

        if device == "cuda" and torch.cuda.is_available():
            dev = torch.device("cuda:0")
        else:
            dev = torch.device("cpu")

        print(f"Loading MDM model on {device}...")
        data = MinimalDataset(args.dataset, device=dev)
        model, diffusion = create_model_and_diffusion(args, data)
        load_saved_model(model, model_path, use_avg=use_ema)
        model.to(dev)
        model.eval()

        fps = 12.5 if args.dataset == 'kit' else 20
        n_joints = 22 if args.dataset == 'humanml' else 21

        pipeline = {
            'model': model, 'diffusion': diffusion, 'args': args,
            'data': data, 'device': dev, 'fps': fps,
            'max_frames': 196, 'n_joints': n_joints,
        }
        print(f"MDM model loaded: {n_joints} joints, {fps} FPS")
        return (pipeline,)


# ============================================================================
# NODE: MDM Generate Motion
# ============================================================================

class MDMGenerateMotion:
    """Generates human motion from text prompts using diffusion"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mdm_pipeline": ("MDM_PIPELINE",),
                "text_prompt": ("STRING", {
                    "default": "a person walks forward",
                    "multiline": True,
                }),
                "motion_length": ("FLOAT", {
                    "default": 6.0, "min": 0.5, "max": 9.8, "step": 0.1,
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1,
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
                "num_samples": ("INT", {"default": 1, "min": 1, "max": 10}),
            },
        }

    RETURN_TYPES = ("MDM_MOTION",)
    RETURN_NAMES = ("motion",)
    FUNCTION = "generate"
    CATEGORY = "MDM/Generation"

    def generate(self, mdm_pipeline, text_prompt, motion_length, guidance_scale, seed, num_samples):
        from .mdm.utils.fixseed import fixseed
        from .mdm.utils.sampler_util import ClassifierFreeSampleModel
        from .mdm.data_loaders.tensors import collate
        from .mdm.data_loaders.humanml.scripts.motion_process import recover_from_ric

        model = mdm_pipeline['model']
        diffusion = mdm_pipeline['diffusion']
        args = mdm_pipeline['args']
        data = mdm_pipeline['data']
        device = mdm_pipeline['device']
        fps = mdm_pipeline['fps']
        n_joints = mdm_pipeline['n_joints']

        fixseed(seed)
        n_frames = min(196, int(motion_length * fps))
        texts = [text_prompt] * num_samples

        if guidance_scale != 1.0 and args.cond_mask_prob > 0:
            sample_model = ClassifierFreeSampleModel(model)
        else:
            sample_model = model

        sample_model.to(device)
        sample_model.eval()

        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames, 'text': txt} for txt in texts]
        _, model_kwargs = collate(collate_args)
        model_kwargs['y'] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in model_kwargs['y'].items()}

        if guidance_scale != 1.0:
            model_kwargs['y']['scale'] = torch.ones(num_samples, device=device) * guidance_scale
        if 'text' in model_kwargs['y']:
            model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])

        motion_shape = (num_samples, model.njoints, model.nfeats, n_frames)
        print(f"Generating motion: {n_frames} frames, {motion_length:.1f}s")

        sample = diffusion.p_sample_loop(sample_model, motion_shape, clip_denoised=False,
                                          model_kwargs=model_kwargs, skip_timesteps=0, progress=True)

        if model.data_rep == 'hml_vec':
            sample = data.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
        sample = model.rot2xyz(x=sample, mask=None, pose_rep=rot2xyz_pose_rep, glob=True,
                               translation=True, jointstype='smpl', vertstrans=True)

        motion_output = {
            'motion': sample.cpu().numpy(),
            'text': texts,
            'lengths': np.array([n_frames] * num_samples),
            'fps': fps,
            'n_joints': n_joints,
            'num_samples': num_samples,
        }
        print(f"Generated: {sample.shape}")
        return (motion_output,)


# ============================================================================
# NODE: MDM Motion Preview
# ============================================================================

class MDMMotionPreview:
    """Visualizes motion as stick figure images"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion": ("MDM_MOTION",),
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 9}),
                "num_frames": ("INT", {"default": 8, "min": 1, "max": 32}),
                "elevation": ("FLOAT", {"default": 110.0, "min": 0.0, "max": 180.0, "step": 5.0}),
                "azimuth": ("FLOAT", {"default": -90.0, "min": -180.0, "max": 180.0, "step": 5.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview_images",)
    FUNCTION = "preview"
    CATEGORY = "MDM/Visualization"

    def preview(self, motion, sample_index, num_frames, elevation, azimuth):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return (torch.zeros((1, 256, 256, 3)),)

        motion_data = motion['motion']
        sample_index = min(sample_index, motion_data.shape[0] - 1)
        sample_motion = motion_data[sample_index]  # Shape: [num_joints, 3, num_frames]

        # Get total frames from the last dimension
        total_frames = sample_motion.shape[-1]
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        images = []
        for frame_idx in frame_indices:
            # Extract joints for this frame: [num_joints, 3] -> transpose to [3, num_joints] -> back to [num_joints, 3]
            # sample_motion shape is [22, 3, frames], so [:, :, frame_idx] gives [22, 3]
            joints = sample_motion[:, :, frame_idx]  # Shape: [22, 3] - already correct!
            fig = plt.figure(figsize=(6, 6), dpi=100)
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='red', s=30)
            for (i, j) in SMPL_BONE_CONNECTIONS:
                ax.plot3D([joints[i, 0], joints[j, 0]], [joints[i, 1], joints[j, 1]],
                         [joints[i, 2], joints[j, 2]], 'blue', linewidth=2)

            ax.view_init(elev=elevation, azim=azimuth)
            ax.set_title(f"Frame {frame_idx}")

            # Convert matplotlib figure to image tensor (modern API)
            fig.canvas.draw()
            # Get the RGBA buffer and convert to numpy array
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            img_array = buf.reshape(h, w, 4)[:, :, :3]  # Remove alpha channel
            img_tensor = torch.from_numpy(img_array.astype(np.float32) / 255.0).unsqueeze(0)
            images.append(img_tensor)
            plt.close(fig)

        return (torch.cat(images, dim=0),)


# ============================================================================
# NODE: MDM Motion to Numpy
# ============================================================================

class MDMMotionToNumpy:
    """Exports motion data to numpy file"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion": ("MDM_MOTION",),
                "output_path": ("STRING", {"default": "output/motion_data.npy"}),
                "sample_index": ("INT", {"default": -1, "min": -1, "max": 9}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "export"
    CATEGORY = "MDM/Export"
    OUTPUT_NODE = True

    def export(self, motion, output_path, sample_index):
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        motion_data = motion['motion']

        if sample_index >= 0:
            single_motion = motion_data[min(sample_index, motion_data.shape[0] - 1)].transpose(2, 0, 1)
            export_data = {'motion': single_motion, 'text': motion['text'][sample_index],
                          'fps': motion['fps'], 'n_joints': motion['n_joints']}
        else:
            export_data = {'motion': motion_data.transpose(0, 3, 1, 2), **motion}

        np.save(output_path, export_data)
        print(f"Saved: {output_path}")
        return (output_path,)


# ============================================================================
# NODE: Load Humanoid Mesh
# ============================================================================

class LoadHumanoidMesh:
    """Loads 3D humanoid mesh for skeleton fitting"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MESH_DATA",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "load_mesh"
    CATEGORY = "MDM/Skeleton"

    def load_mesh(self, mesh_path):
        import trimesh

        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        mesh = trimesh.load(mesh_path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = list(mesh.geometry.values())[0]

        mesh_data = {
            'mesh': mesh,
            'vertices': np.array(mesh.vertices),
            'faces': np.array(mesh.faces),
            'bounds': mesh.bounds,
            'centroid': mesh.centroid,
            'height': max(mesh.bounds[1][1] - mesh.bounds[0][1], mesh.bounds[1][2] - mesh.bounds[0][2]),
        }
        print(f"Loaded mesh: {len(mesh.vertices)} vertices, height={mesh_data['height']:.3f}")
        return (mesh_data,)


# ============================================================================
# NODE: Fit SMPL Skeleton to Mesh
# ============================================================================

class FitSMPLSkeletonToMesh:
    """Automatically positions SMPL skeleton inside humanoid mesh"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH_DATA",),
                "alignment_method": (["geometric", "icp", "hybrid"], {"default": "hybrid"}),
                "height_scale_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "vertical_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("SMPL_SKELETON",)
    RETURN_NAMES = ("skeleton",)
    FUNCTION = "fit_skeleton"
    CATEGORY = "MDM/Skeleton"

    def fit_skeleton(self, mesh, alignment_method, height_scale_factor, vertical_offset):
        skeleton = get_canonical_skeleton() * height_scale_factor

        # Geometric fit: scale and translate
        scale = mesh['height'] / 1.7
        skeleton = skeleton * scale
        target_pelvis = np.array([mesh['centroid'][0],
                                   mesh['bounds'][0][1] + (mesh['bounds'][1][1] - mesh['bounds'][0][1]) * 0.55,
                                   mesh['centroid'][2]])
        skeleton += target_pelvis - skeleton[0]

        # ICP refinement
        if alignment_method in ["icp", "hybrid"]:
            try:
                import open3d as o3d
                skeleton_pcd = o3d.geometry.PointCloud()
                skeleton_pcd.points = o3d.utility.Vector3dVector(skeleton)
                target_pcd = o3d.geometry.PointCloud()
                target_pcd.points = o3d.utility.Vector3dVector(mesh['vertices'])
                result = o3d.pipelines.registration.registration_icp(
                    skeleton_pcd, target_pcd, mesh['height'] * 0.1)
                skeleton_pcd.transform(result.transformation)
                skeleton = np.asarray(skeleton_pcd.points)
                print(f"ICP fitness: {result.fitness:.4f}")
            except ImportError:
                print("Open3D not available, using geometric fit only")

        skeleton[:, 1] += vertical_offset
        skeleton_dict = create_skeleton_dict(skeleton, fps=20.0)
        skeleton_dict['mesh_info'] = mesh
        print(f"Fitted skeleton: pelvis={skeleton[0]}, head={skeleton[15]}")
        return (skeleton_dict,)


# ============================================================================
# NODE: Apply Motion to Skeleton
# ============================================================================

class ApplyMotionToSkeleton:
    """Applies MDM motion to fitted skeleton"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton": ("SMPL_SKELETON",),
                "motion": ("MDM_MOTION",),
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 9}),
                "preserve_root_position": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("ANIMATED_SKELETON",)
    RETURN_NAMES = ("animated_skeleton",)
    FUNCTION = "apply_motion"
    CATEGORY = "MDM/Skeleton"

    def apply_motion(self, skeleton, motion, sample_index, preserve_root_position):
        fitted_joints = skeleton['joints']
        fps = skeleton['fps']
        motion_data = motion['motion']
        sample_index = min(sample_index, motion_data.shape[0] - 1)
        single_motion = motion_data[sample_index]
        n_joints, _, n_frames = single_motion.shape

        canonical_skeleton = get_canonical_skeleton()
        fitted_height = np.max(fitted_joints[:, 1]) - np.min(fitted_joints[:, 1])
        canonical_height = np.max(canonical_skeleton[:, 1]) - np.min(canonical_skeleton[:, 1])
        scale = fitted_height / canonical_height if canonical_height > 0 else 1.0

        animated_joints = np.zeros((n_frames, n_joints, 3))
        for frame_idx in range(n_frames):
            frame_joints = single_motion[:, :, frame_idx].T
            if preserve_root_position:
                relative_motion = frame_joints - frame_joints[0]
                animated_joints[frame_idx] = fitted_joints + relative_motion * scale
            else:
                centered_motion = frame_joints - canonical_skeleton[0]
                animated_joints[frame_idx] = centered_motion * scale + fitted_joints[0]

        animated_output = {
            'joints': animated_joints,
            'fps': fps,
            'n_frames': n_frames,
            'n_joints': n_joints,
            'text': motion['text'][sample_index],
            'joint_names': skeleton['joint_names'],
            'parent_indices': skeleton['parent_indices'],
            'bone_connections': skeleton['bone_connections'],
            'scale': scale,
            'fitted_rest_pose': fitted_joints,
        }
        print(f"Applied motion: {animated_joints.shape}")
        return (animated_output,)


# ============================================================================
# NODE: Export Animated Skeleton
# ============================================================================

class ExportAnimatedSkeleton:
    """Exports animated skeleton to BVH or JSON"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "animated_skeleton": ("ANIMATED_SKELETON",),
                "output_path": ("STRING", {"default": "output/animation.bvh"}),
                "format": (["bvh", "json"], {"default": "bvh"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "export"
    CATEGORY = "MDM/Export"
    OUTPUT_NODE = True

    def export(self, animated_skeleton, output_path, format):
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        if format == "bvh" and not output_path.endswith(".bvh"):
            output_path = output_path.rsplit(".", 1)[0] + ".bvh"
        elif format == "json" and not output_path.endswith(".json"):
            output_path = output_path.rsplit(".", 1)[0] + ".json"

        if format == "json":
            export_data = {
                'fps': float(animated_skeleton['fps']),
                'n_frames': int(animated_skeleton['n_frames']),
                'joint_names': animated_skeleton['joint_names'],
                'parent_indices': animated_skeleton['parent_indices'],
                'joints': animated_skeleton['joints'].tolist(),
                'rest_pose': animated_skeleton['fitted_rest_pose'].tolist(),
            }
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            # BVH export
            self._export_bvh(animated_skeleton, output_path)

        print(f"Exported: {output_path}")
        return (output_path,)

    def _export_bvh(self, anim, path):
        joints = anim['joints']
        fps = anim['fps']
        joint_names = anim['joint_names']
        parent_indices = anim['parent_indices']
        rest_pose = anim['fitted_rest_pose']
        n_frames, n_joints, _ = joints.shape

        with open(path, 'w') as f:
            f.write("HIERARCHY\n")

            def write_joint(idx, indent=""):
                name = joint_names[idx]
                parent_idx = parent_indices[idx]
                f.write(f"{indent}{'ROOT' if parent_idx == -1 else 'JOINT'} {name}\n{indent}{{\n")
                offset = rest_pose[idx] if parent_idx == -1 else rest_pose[idx] - rest_pose[parent_idx]
                f.write(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
                f.write(f"{indent}  CHANNELS {'6 Xposition Yposition Zposition' if parent_idx == -1 else '3'} Zrotation Xrotation Yrotation\n")
                children = [i for i in range(n_joints) if parent_indices[i] == idx]
                if children:
                    for child in children:
                        write_joint(child, indent + "  ")
                else:
                    f.write(f"{indent}  End Site\n{indent}  {{\n{indent}    OFFSET 0.0 0.0 0.0\n{indent}  }}\n")
                f.write(f"{indent}}}\n")

            write_joint(0)
            f.write(f"MOTION\nFrames: {n_frames}\nFrame Time: {1.0/fps:.6f}\n")

            for frame_idx in range(n_frames):
                frame_data = []
                for joint_idx in range(n_joints):
                    if parent_indices[joint_idx] == -1:
                        pos = joints[frame_idx, joint_idx]
                        frame_data.extend([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0])
                    else:
                        frame_data.extend([0.0, 0.0, 0.0])
                f.write(" ".join([f"{v:.6f}" for v in frame_data]) + "\n")




# ============================================================================
# NODE MAPPINGS
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "DownLoadMDMModel": DownLoadMDMModel,
    "MDMGenerateMotion": MDMGenerateMotion,
    "MDMMotionPreview": MDMMotionPreview,
    "MDMMotionToNumpy": MDMMotionToNumpy,
    "LoadHumanoidMesh": LoadHumanoidMesh,
    "FitSMPLSkeletonToMesh": FitSMPLSkeletonToMesh,
    "ApplyMotionToSkeleton": ApplyMotionToSkeleton,
    "ExportAnimatedSkeleton": ExportAnimatedSkeleton,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownLoadMDMModel": "(down)Load MDM Model",
    "MDMGenerateMotion": "MDM Generate Motion",
    "MDMMotionPreview": "MDM Motion Preview",
    "MDMMotionToNumpy": "MDM Motion to Numpy",
    "LoadHumanoidMesh": "Load Humanoid Mesh",
    "FitSMPLSkeletonToMesh": "Fit SMPL Skeleton to Mesh",
    "ApplyMotionToSkeleton": "Apply Motion to Skeleton",
    "ExportAnimatedSkeleton": "Export Animated Skeleton",
}
