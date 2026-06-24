import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
import tifffile

from vesselattributeprediction.constants import VESSEL_ID_TO_VESSEL_TYPES, VESSEL_TYPES_TO_CLASS_ID

TYPE_ERROR_MSG = "Only 3-channel RGB optical satellite images in TIFF formats are supported."


def get_abs_path(
    dataset_root: Union[str, Path],
    path: Union[str, Path],
) -> Path:
    """
    Resolves relative path to absolute paths.
    """
    resolved_path = Path(path)
    if resolved_path.is_absolute():
        return resolved_path
    return Path(dataset_root) / resolved_path


def load_tiff_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Loads images and returns RGB array from tiff images. Raises error if image is not in supported format.
    """
    image_path = Path(image_path)

    if image_path.suffix.lower() != ".tif":
        raise TypeError(TYPE_ERROR_MSG)

    image = tifffile.imread(image_path)

    if image.ndim != 3:
        raise TypeError(TYPE_ERROR_MSG)

    if image.shape[-1] == 3:
        return image

    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)

    if image.shape[-1] != 3:
        raise TypeError(TYPE_ERROR_MSG)

    return image


def collect_img_files(img_dir: Union[str, Path]) -> List[Union[str, Path]]:
    """
    Returns a list of all image filepaths in the image directory.
    """
    image_paths = sorted(img_dir.glob("*.tif"))
    if not image_paths:
        raise FileNotFoundError(f"No tiff images found in image directory: {img_dir}")

    return image_paths


def convert_image_to_tf(image: np.ndarray) -> tf.Tensor:
    """
    Convert image from numpy to tensor.
    """
    image_tf = tf.convert_to_tensor(image)
    image_tf = tf.image.convert_image_dtype(
        image_tf,
        dtype=tf.float32,
    )
    image_tf = tf.ensure_shape(
        image_tf,
        (None, None, 3),
    )

    return image_tf


def load_geojson_file(geojson_path: Union[str, Path]) -> Dict:
    """
    Loads a geojson file and returns content as a dictionary.
    """
    with geojson_path.open(encoding="utf-8") as f:
        return json.load(f)


def map_type_id_to_class_id_and_vessel(type_id: int) -> Tuple[int, str]:
    """
    Maps type id in xView dataset to class id and vessel type for vessel detection.
    """
    if type_id in VESSEL_ID_TO_VESSEL_TYPES.keys():
        vessel_type = VESSEL_ID_TO_VESSEL_TYPES[type_id]
        class_id = VESSEL_TYPES_TO_CLASS_ID[vessel_type]

    else:
        vessel_type = "Non-vessel"
        class_id = -1

    return class_id, vessel_type
