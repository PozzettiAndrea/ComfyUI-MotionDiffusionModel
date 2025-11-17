# ComfyUI-MotionDiffusionModel nodes package

# Make chumpy_fork available as 'chumpy' for SMPL pickle compatibility
import sys
try:
    import chumpy_fork
    sys.modules['chumpy'] = chumpy_fork
except ImportError:
    pass  # chumpy_fork not installed yet
