"""
ComfyUI-MotionDiffusionModel

Text-to-motion generation and SMPL skeleton fitting for humanoid meshes.

Based on: Human Motion Diffusion Model (ICLR 2023)
Repository: https://github.com/GuyTevet/motion-diffusion-model
"""

from .nodes.mdm_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "1.0.0"
