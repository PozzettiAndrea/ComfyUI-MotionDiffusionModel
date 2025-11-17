# ComfyUI-MotionDiffusionModel

Text-to-motion generation using Motion Diffusion Model (MDM) with SMPL skeleton fitting for humanoid meshes.

![MDM](assets/mdm_banner.png)

## Features

- **Text-to-Motion Generation**: Generate realistic human motion from natural language descriptions
- **SMPL Skeleton Fitting**: Automatically position SMPL skeleton inside any humanoid mesh
- **Motion Visualization**: Preview generated motions as stick figure animations
- **Export Options**: Export to BVH (for Blender/Maya/Unity) or JSON formats

## Installation

1. Clone or copy to ComfyUI custom_nodes:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/ComfyUI-MotionDiffusionModel
```

2. Install dependencies (CLIP included automatically):
```bash
cd ComfyUI-MotionDiffusionModel
pip install -r requirements.txt
```

3. **Restart ComfyUI** - Models download automatically via the Download node!

## Nodes

### Model Management
| Node | Description |
|------|-------------|
| **Download MDM Checkpoint** | Auto-download pre-trained models from HuggingFace |
| **MDM Model Loader** | Load downloaded MDM model checkpoint |

### Motion Generation
| Node | Description |
|------|-------------|
| **MDM Generate Motion** | Generate motion from text prompt |
| **MDM Motion Preview** | Visualize motion as stick figures |
| **MDM Motion to Numpy** | Export raw motion data to .npy |

### Skeleton Fitting
| Node | Description |
|------|-------------|
| **Load Humanoid Mesh** | Load 3D mesh (OBJ, FBX, GLB, PLY) |
| **Fit SMPL Skeleton to Mesh** | Auto-position SMPL skeleton in mesh |
| **Apply Motion to Skeleton** | Apply generated motion to fitted skeleton |
| **Export Animated Skeleton** | Export to BVH or JSON format |

## Quick Start

### Basic Text-to-Motion
```
[Download MDM Checkpoint] ← select model
        ↓ model_path
[MDM Model Loader]
        ↓ MDM_PIPELINE
[MDM Generate Motion] ← "a person waves hello"
        ↓ MDM_MOTION
[MDM Motion Preview] → stick figure images
```

### Full Pipeline with Skeleton Fitting
```
[Download MDM Checkpoint] → model_path
        ↓
[MDM Model Loader] ─────────────────┐
                                     ↓
[MDM Generate Motion] ← text_prompt
        ↓
    MDM_MOTION ──────────────────────┐
                                     ↓
[Load Humanoid Mesh] ← character.obj
        ↓
[Fit SMPL Skeleton to Mesh]
        ↓
[Apply Motion to Skeleton] ←─────────┘
        ↓
[Export Animated Skeleton] → animation.bvh
```

## Pre-trained Models

**Automatic download via "Download MDM Checkpoint" node!**

Available models:
| Model | Speed | Quality |
|-------|-------|---------|
| `humanml_trans_enc_512` | Standard (1000 steps) | Best |
| `humanml-encoder-512-50steps` | Fast (50 steps, 20x faster) | Good |
| `kit-encoder-512` | KIT dataset | Specialized |

Models are automatically downloaded to: `lib/mdm/save/`

## Technical Details

- **Skeleton**: SMPL 22 joints
- **Max Duration**: ~9.8 seconds (196 frames @ 20 FPS)
- **Output Format**: Joint XYZ positions per frame
- **Guidance Scale**: 1.0 = no guidance, 2.5+ = better text adherence

## GPU Requirements

- CLIP encoder: ~400MB VRAM
- MDM model: ~1-2GB VRAM
- **Recommended**: 4-6GB VRAM total

## Troubleshooting

**CLIP not found:**
```bash
pip install git+https://github.com/openai/CLIP.git
```

**Open3D errors (for ICP fitting):**
```bash
pip install open3d
```

**Model not loading:**
- Ensure `args.json` exists alongside `.pt` file
- Check path is correct in node

## Credits

- [Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model) - Tevet et al., ICLR 2023
- [SMPL Body Model](https://smpl.is.tue.mpg.de/)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## License

This integration follows the licenses of MDM (MIT) and SMPL.
