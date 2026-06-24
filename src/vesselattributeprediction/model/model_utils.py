from typing import Dict

import numpy as np
import tensorflow as tf


def _get_non_encapsulated_bbox_indexes(
    boxes: np.ndarray,
) -> np.ndarray:
    """
    Returns indexes of boxes that are not fully encapsulated by a larger box.
    """
    if len(boxes) == 0:
        return np.asarray([], dtype=int)

    boxes = np.asarray(boxes, dtype=np.float32)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep_indexes = []
    for box_index, box in enumerate(boxes):
        contained_by_larger_box = np.any(
            (areas > areas[box_index])
            & (boxes[:, 0] <= box[0])
            & (boxes[:, 1] <= box[1])
            & (boxes[:, 2] >= box[2])
            & (boxes[:, 3] >= box[3])
        )
        if not contained_by_larger_box:
            keep_indexes.append(box_index)

    return np.asarray(keep_indexes, dtype=int)


def filter_overlapping_bbox_predictions(
    pred,
) -> Dict:
    """
    Retains highest-confidence non-overlapping bbox predictions.
    """
    pred_boxes = pred["boxes"]
    pred_classes = pred["classes"]
    pred_confidences = pred["confidence"]
    num_pred = pred["num_detections"]

    final_boxes = []
    final_classes = []
    final_confidences = []
    final_counts = []

    for i, num in enumerate(num_pred):
        # Retain valid predictions
        boxes = pred_boxes[i, :num]
        classes = pred_classes[i, :num]
        conf = pred_confidences[i, :num]

        # Use NMS to remove overlapping boxes based on IoU
        bboxes_yxyx = boxes[:, [1, 0, 3, 2]]
        filtered_indexes = tf.image.non_max_suppression(
            boxes=bboxes_yxyx, scores=conf, max_output_size=len(bboxes_yxyx)
        ).numpy()

        boxes = boxes[filtered_indexes]
        classes = classes[filtered_indexes]
        conf = conf[filtered_indexes]

        # Remove boxes fully contained inside larger boxes
        non_encapsulated_indexes = _get_non_encapsulated_bbox_indexes(boxes)

        final_boxes.append(boxes[non_encapsulated_indexes])
        final_classes.append(classes[non_encapsulated_indexes])
        final_confidences.append(conf[non_encapsulated_indexes])
        final_counts.append(len(non_encapsulated_indexes))

    return {
        "boxes": final_boxes,
        "classes": final_classes,
        "confidence": final_confidences,
        "num_detections": final_counts,
    }
