from typing import List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from vesselattributeprediction.constants import DEFAULT_CROP_PAD_SIZE, DEFAULT_IMAGE_SHAPE
from vesselattributeprediction.utils import load_tiff_image


def overlay_bboxes_on_image(
    image: np.ndarray,
    bboxes: np.ndarray,
    labels: List[str],
    bbox_line_width: int = 1,
    label_font_size: int = 2,
) -> np.ndarray:
    """
    Overlays bounding boxes and labels on image.
    """
    image_height, image_width = image.shape[:2]
    figure, axis = plt.subplots(
        figsize=(image_width / 100, image_height / 100),
        dpi=300,
    )
    axis.imshow(image.astype(np.uint8))
    axis.axis("off")

    for bbox, label in zip(bboxes, labels):
        x_min, y_min, x_max, y_max = bbox
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        rectangle = patches.Rectangle(
            (x_min, y_min),
            bbox_width,
            bbox_height,
            linewidth=bbox_line_width,
            edgecolor="red",
            facecolor="none",
        )
        axis.add_patch(rectangle)
        axis.text(
            x_max - 5,
            y_min - 5,
            label,
            color="black",
            fontsize=label_font_size,
            bbox={"facecolor": "red", "alpha": 0.75},
        )

    figure.canvas.draw()
    overlay_image = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(figure)

    return overlay_image


def crop_vessel_image(
    image: np.ndarray,
    bbox: np.ndarray,
    padding: int = DEFAULT_CROP_PAD_SIZE,
) -> Tuple[np.ndarray, List[int]]:
    """
    Crops image to detected vessels using specified bbox dimensions.
    """
    image_height, image_width = image.shape[:2]
    x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]

    crop_x_min = max(0, x_min - padding)
    crop_y_min = max(0, y_min - padding)
    crop_x_max = min(image_width, x_max + padding)
    crop_y_max = min(image_height, y_max + padding)

    vessel_crop = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max, :]
    crop_bbox = [
        x_min - crop_x_min,
        y_min - crop_y_min,
        x_max - crop_x_min,
        y_max - crop_y_min,
    ]

    return vessel_crop, crop_bbox


def convert_bbox_dim(bbox_dim_str: str) -> Optional[List[int]]:
    """
    Converts bbox xy dimensions of type string to xmin, ymin, xmax, ymax integers.
    """
    try:
        x_min, y_min, x_max, y_max = [int(value) for value in bbox_dim_str.split(",")]

    except ValueError:
        return None

    # Ensure bbox annotations are accurate
    if x_min >= x_max or y_min >= y_max:
        return None

    return [x_min, y_min, x_max, y_max]


def scale_image_and_bounding_boxes(
    img_array: np.ndarray,
    bbox_dims: List[List[int]],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Scales one image and all bounding boxes to the target size.
    """
    target_height, target_width = DEFAULT_IMAGE_SHAPE[:2]
    original_height, original_width = img_array.shape[:2]

    image = tf.image.convert_image_dtype(img_array, tf.float32)
    bounding_boxes = tf.convert_to_tensor(
        bbox_dims,
        dtype=tf.float32,
    )
    bounding_boxes = tf.reshape(bounding_boxes, (-1, 4))

    scale_factor = tf.minimum(
        tf.cast(target_height, tf.float32) / tf.cast(original_height, tf.float32),
        tf.cast(target_width, tf.float32) / tf.cast(original_width, tf.float32),
    )

    scaled_height = tf.cast(
        tf.round(tf.cast(original_height, tf.float32) * scale_factor),
        tf.int32,
    )
    scaled_width = tf.cast(
        tf.round(tf.cast(original_width, tf.float32) * scale_factor),
        tf.int32,
    )

    scaled_image = tf.image.resize(
        image,
        size=(scaled_height, scaled_width),
    )

    offset_y = (target_height - scaled_height) // 2
    offset_x = (target_width - scaled_width) // 2

    scaled_image_padded = tf.image.pad_to_bounding_box(
        scaled_image,
        offset_height=offset_y,
        offset_width=offset_x,
        target_height=target_height,
        target_width=target_width,
    )

    bbox_offsets = tf.cast([offset_x, offset_y, offset_x, offset_y], tf.float32)
    scaled_bboxes = (bounding_boxes * scale_factor) + bbox_offsets

    # Clip bbox dims to image dims
    scaled_bboxes = tf.clip_by_value(
        scaled_bboxes,
        clip_value_min=[
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        clip_value_max=[
            float(target_width),
            float(target_height),
            float(target_width),
            float(target_height),
        ],
    )

    return scaled_image_padded, scaled_bboxes


def load_and_scale_image(
    image_path: tf.Tensor,
    bbox_dims: tf.Tensor,
    class_ids: tf.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and scales one image and annotations.
    """
    image_filepath = image_path.numpy().decode("utf-8")
    image = load_tiff_image(image_filepath)
    scaled_image, scaled_bboxes = scale_image_and_bounding_boxes(
        image,
        bbox_dims.numpy().tolist(),
    )

    return (
        scaled_image.numpy().astype(np.float32),
        scaled_bboxes.numpy().astype(np.float32),
        class_ids.numpy().astype(np.float32),
    )
