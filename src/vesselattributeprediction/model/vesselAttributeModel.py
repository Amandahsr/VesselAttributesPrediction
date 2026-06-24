from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from sahi.slicing import slice_image

from vesselattributeprediction.constants import (
    CLASS_ID_TO_VESSEL_TYPES,
    DEFAULT_IMAGE_SHAPE,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
)
from vesselattributeprediction.data.data_utils import (
    crop_vessel_image,
    overlay_bboxes_on_image,
    scale_image_and_bounding_boxes,
)
from vesselattributeprediction.model.cargoPredictorModel import CargoPredictorModel
from vesselattributeprediction.model.model_utils import filter_overlapping_bbox_predictions
from vesselattributeprediction.model.vesselDetectorModel import VesselDetectorModel
from vesselattributeprediction.utils import load_tiff_image


class VesselAttributeModel:
    """
    Multi-modal inference model that detects vessel objects in images and predicts cargo status based on detected vessel image and vessel type.
    """

    def __init__(self, model_file: Path):
        self.vessel_detector_filepath: Path = model_file
        self.vessel_detector_model: Optional[VesselDetectorModel] = None
        self.cargo_predictor_model: CargoPredictorModel = CargoPredictorModel()

        self.load_vessel_model()

    def load_vessel_model(self) -> None:
        self.vessel_detector_model = tf.keras.models.load_model(
            self.vessel_detector_filepath,
            compile=False,
        )

    def _detect_vessel_objects(
        self,
        image: np.ndarray,
    ) -> Dict:
        """
        Runs vessel detection on image and returns bounding boxes.
        """
        if self.vessel_detector_model is None:
            raise ValueError("No vessel detector model has been loaded.")

        sliced_result = slice_image(
            image=image,
            output_file_name="inference",
            output_dir=None,
            slice_height=DEFAULT_TILE_SIZE,
            slice_width=DEFAULT_TILE_SIZE,
            overlap_height_ratio=DEFAULT_TILE_OVERLAP,
            overlap_width_ratio=DEFAULT_TILE_OVERLAP,
            auto_slice_resolution=False,
            verbose=False,
        )
        tile_predictions = [
            self._predict_on_tile(
                tile_image=sliced_image.image,
                tile_starting_pixel=sliced_image.starting_pixel,
            )
            for sliced_image in sliced_result.sliced_image_list
        ]

        all_boxes = [prediction["boxes"] for prediction in tile_predictions]
        all_classes = [prediction["classes"] for prediction in tile_predictions]
        all_confidences = [prediction["confidence"] for prediction in tile_predictions]

        boxes = np.concatenate(all_boxes) if all_boxes else np.empty((0, 4), dtype=np.float32)
        classes = np.concatenate(all_classes) if all_classes else np.asarray([], dtype=int)
        confidences = np.concatenate(all_confidences) if all_confidences else np.asarray([], dtype=np.float32)

        if len(boxes) == 0:
            return {
                "boxes": [np.empty((0, 4), dtype=np.float32)],
                "classes": [np.asarray([], dtype=int)],
                "confidence": [np.asarray([], dtype=np.float32)],
                "num_detections": [0],
            }

        return filter_overlapping_bbox_predictions(
            {
                "boxes": np.expand_dims(boxes, axis=0),
                "classes": np.expand_dims(classes, axis=0),
                "confidence": np.expand_dims(confidences, axis=0),
                "num_detections": np.asarray([len(boxes)], dtype=int),
            }
        )

    def _predict_on_tile(
        self,
        tile_image: np.ndarray,
        tile_starting_pixel: list[int],
    ) -> Dict:
        """
        Runs vessel detection on one image tile and returns boxes in original image coordinates.
        """
        scaled_image, _ = scale_image_and_bounding_boxes(tile_image, [])
        image_batch = tf.expand_dims(scaled_image, axis=0)
        vessel_predictions = self.vessel_detector_model.predict(image_batch, verbose=0)
        vessel_predictions = filter_overlapping_bbox_predictions(vessel_predictions)

        boxes = self._rescale_bboxes_to_original_image(
            bboxes=vessel_predictions["boxes"][0],
            original_image_shape=tile_image.shape,
        )
        tile_x, tile_y = tile_starting_pixel
        boxes = boxes + np.asarray([tile_x, tile_y, tile_x, tile_y], dtype=np.float32)

        return {
            "boxes": boxes,
            "classes": np.asarray(vessel_predictions["classes"][0]).astype(int),
            "confidence": np.asarray(vessel_predictions["confidence"][0], dtype=np.float32),
        }

    def _rescale_bboxes_to_original_image(
        self,
        bboxes: np.ndarray,
        original_image_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Converts xyxy bboxes from scaled detector image coordinates to original image coordinates.
        """
        if len(bboxes) == 0:
            return np.asarray(bboxes, dtype=np.float32).reshape((-1, 4))

        target_height, target_width = DEFAULT_IMAGE_SHAPE[:2]
        original_height, original_width = original_image_shape[:2]
        scale_factor = min(
            target_height / original_height,
            target_width / original_width,
        )
        scaled_height = round(original_height * scale_factor)
        scaled_width = round(original_width * scale_factor)
        offset_x = (target_width - scaled_width) // 2
        offset_y = (target_height - scaled_height) // 2

        bboxes = np.asarray(bboxes, dtype=np.float32).reshape((-1, 4))
        bbox_offsets = np.asarray(
            [offset_x, offset_y, offset_x, offset_y],
            dtype=np.float32,
        )
        original_bboxes = (bboxes - bbox_offsets) / scale_factor
        original_bboxes[:, [0, 2]] = np.clip(
            original_bboxes[:, [0, 2]],
            0.0,
            float(original_width),
        )
        original_bboxes[:, [1, 3]] = np.clip(
            original_bboxes[:, [1, 3]],
            0.0,
            float(original_height),
        )

        return original_bboxes

    def _predict_cargo_status(
        self, image: np.ndarray, bbox_dim: np.ndarray, vessel_type: str
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Returns cropped vessel object with cargo status and cargo prediction score.
        """
        vessel_crop, _ = crop_vessel_image(image, bbox_dim)
        normalized_vessel_crop = tf.image.convert_image_dtype(
            vessel_crop,
            dtype=tf.float32,
        ).numpy()
        cargo_status, confidence_score = self.cargo_predictor_model.predict_vessel_cargo_status(
            normalized_vessel_crop,
            vessel_type,
        )

        return vessel_crop, cargo_status, confidence_score

    def detect_vessel_attributes(self, image_file: Union[str, Path]) -> Dict:
        """
        Runs full vessel detection + cargo status workflow.
        """
        image = load_tiff_image(image_file)

        # Detect vessel objects in image
        vessel_predictions = self._detect_vessel_objects(image)
        bboxes = np.asarray(vessel_predictions["boxes"][0], dtype=np.float32).reshape((-1, 4))
        class_ids = np.asarray(vessel_predictions["classes"][0]).astype(int)

        if len(bboxes) == 0:
            return {
                "original_image": image,
                "annotated_image": None,
                "bounding_boxes": None,
                "num_vessels_detected": None,
                "vessels": None,
            }

        vessel_results = []
        for bbox_dim, class_id in zip(
            bboxes,
            class_ids,
        ):
            vessel_type = CLASS_ID_TO_VESSEL_TYPES[int(class_id)]
            vessel_crop, cargo_status, cargo_confidence = self._predict_cargo_status(
                image,
                bbox_dim,
                vessel_type,
            )
            _, crop_bbox = crop_vessel_image(image, bbox_dim)
            vessel_label = f"vessel type: {vessel_type}\n" f"cargo status: {cargo_status}"

            vessel_results.append(
                {
                    "bbox": bbox_dim.tolist(),
                    "crop_bbox": crop_bbox,
                    "vessel_type": vessel_type,
                    "cargo_status": cargo_status,
                    "cargo_confidence": cargo_confidence,
                    "vessel_crop": vessel_crop,
                    "annotated_vessel_crop": overlay_bboxes_on_image(
                        image=vessel_crop,
                        bboxes=np.asarray([crop_bbox]),
                        labels=[vessel_label],
                    ),
                }
            )

        full_image_labels = [f"Vessel type: {vessel_result['vessel_type']}" for vessel_result in vessel_results]

        return {
            "original_image": image,
            "annotated_image": (
                overlay_bboxes_on_image(
                    image=image,
                    bboxes=bboxes,
                    labels=full_image_labels,
                )
                if len(bboxes) > 0
                else image
            ),
            "bounding_boxes": bboxes,
            "num_vessels_detected": len(bboxes),
            "vessels": vessel_results,
        }
