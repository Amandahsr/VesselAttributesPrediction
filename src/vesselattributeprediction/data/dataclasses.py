from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import tensorflow as tf


@dataclass(frozen=True)
class XViewObjectAnnotation:
    """
    Dataclass to store original object annotations parsed from xView dataset.
    """

    image_path: Path
    feature_id: int
    object_id: int
    vessel_type: str
    bbox_dim: List[int]
    geometry: List[List[float]]
    detector_class_id: int


@dataclass(frozen=True)
class ImageAnnotationDatasets:
    """
    Dataclass for processed training, validation and testing datasets.
    """

    training: Union[Dict, tf.data.Dataset]
    validation: Union[Dict, tf.data.Dataset]
    testing: Union[Dict, tf.data.Dataset]
