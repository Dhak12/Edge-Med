from pathlib import Path
import torch

# ============================================================
# PATH CONFIGURATION
# ============================================================
# Project root: mri/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "dataset"

TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "validation"
TEST_DIR = DATASET_ROOT / "test"

# ============================================================
# MODEL CONFIGURATION
# ============================================================

NUM_CLASSES = 3

CLASS_NAMES = [
    "Benign",
    "Malignant",
    "Healthy"
]

IMAGE_SIZE = 224

# CPU-friendly batch size
BATCH_SIZE = 16

# REQUIRED: exactly 10 epochs
EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

# ============================================================
# OUTPUT CONFIGURATION
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOTS_DIR = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42
