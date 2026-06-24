from pathlib import Path
from typing import Dict, Tuple

"""
All constants used in this repository are collated here.
"""
# Dataset constants
DEFAULT_XVIEW_DATASET_ROOT: str = "data"
DEFAULT_XVIEW_ANNOTATION_FILE: str = "xView_train.geojson"
DEFAULT_XVIEW_IMAGE_DIR: str = "train_images"
DEFAULT_XVIEW_SPLIT_RATIOS: Tuple[float, float, float] = (0.70, 0.15, 0.15)
DEFAULT_SEED: int = 15
DEFAULT_BATCH_SIZE: int = 4
DEFAULT_TILE_SIZE: int = 640
DEFAULT_TILE_OVERLAP: float = 0.2
DEFAULT_TILE_MIN_AREA_RATIO: float = 0.5
DEFAULT_TILED_IMAGE_DIR: str = "sliced_images"
VESSEL_ID_TO_VESSEL_TYPES = {
    40: "Maritime Vessel",
    41: "Motorboat",
    42: "Sailboat",
    44: "Tugboat",
    45: "Barge",
    47: "Fishing Vessel",
    49: "Ferry",
    50: "Yacht",
    51: "Container Ship",
    52: "Oil Tanker",
}

# Vessel Detector Model constants
DEFAULT_IMAGE_SHAPE: Tuple[int, int, int] = (640, 640, 3)
DEFAULT_YOLO_MODEL: str = "yolo_v8_s_backbone_coco"
DEFAULT_YOLO_BBOX_FORMAT: str = "xyxy"
DEFAULT_MODEL_LEARNING_RATE: float = 0.00001
DEFAULT_IMAGE_DETECTOR_LOSS: str = "binary_crossentropy"
DEFAULT_IMAGE_BBOX_LOSS: str = "ciou"
VESSEL_TYPES_TO_CLASS_ID: Dict = {
    "Maritime Vessel": 0,
    "Motorboat": 1,
    "Sailboat": 2,
    "Tugboat": 3,
    "Barge": 4,
    "Fishing Vessel": 5,
    "Ferry": 6,
    "Yacht": 7,
    "Container Ship": 8,
    "Oil Tanker": 9,
}
DEFAULT_NUM_OUTPUT_CLASSES: int = len(list(VESSEL_TYPES_TO_CLASS_ID.keys()))

# Cargo Semantic Embedding Model constants
DEFAULT_CLIP_MODEL: str = "clip_vit_base_patch16"
DEFAULT_CROP_PAD_SIZE: int = 50

# Training/Testing constants
DEFAULT_NUM_EPOCHS: int = 15
DEFAULT_PROJECT_ROOT: Path = Path.home() / "Documents" / "GitHub" / "VesselAttributesPrediction"
DEFAULT_TRAINING_DIR: str = "detector_model_training"
DEFAULT_IOU_THRESHOLD: float = 0.7
CLASS_ID_TO_VESSEL_TYPES: Dict = {v: k for k, v in VESSEL_TYPES_TO_CLASS_ID.items()}
