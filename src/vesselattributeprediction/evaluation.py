from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from vesselattributeprediction.constants import VESSEL_TYPES_TO_CLASS_ID
from vesselattributeprediction.model.model_utils import filter_overlapping_bbox_predictions


class EvaluationMetrics:
    def __init__(self):
        # Cache ground truth labels
        self._true_boxes: List[np.ndarray] = None
        self._true_classes: List[np.ndarray] = None

    def test_model(
        self,
        test_dataset: tf.data.Dataset,
        model_keras_datafile: Path,
    ) -> Dict:
        """
        Runs model prediction on test dataset.
        """
        if model_keras_datafile:
            self.model = tf.keras.models.load_model(
                model_keras_datafile,
                compile=False,
            )
        else:
            if self.model is None:
                raise ValueError(
                    "No vessel detector model has been initialized. Run initialize_vessel_detector_model() first."
                )

        # Prep ground truth labels
        true_boxes = []
        true_classes = []
        for test_batch in test_dataset:
            true_boxes.extend([np.asarray(bbox, dtype=np.float32) for bbox in test_batch["bounding_boxes"]["boxes"]])
            true_classes.extend(
                [np.asarray(classes).astype(int) for classes in test_batch["bounding_boxes"]["classes"]]
            )

        # Extract testing images for prediction
        testing_dataset = test_dataset.map(
            lambda test_batch: test_batch["images"],
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        vessel_predictions = self.model.predict(testing_dataset, verbose=0)

        # Post-process predictions
        vessel_predictions = filter_overlapping_bbox_predictions(vessel_predictions)
        self._true_boxes = true_boxes
        self._true_classes = true_classes

        return vessel_predictions

    def create_test_results_dict(
        self,
        vessel_predictions: Dict,
    ) -> Dict:
        """
        Aligns model predictions with ground truth labels to create final test results.
        """
        # Align pred to best matched gt for each image
        image_results = [
            self._align_prediction_to_groundtruth(
                pred_boxes=np.asarray(
                    vessel_predictions["boxes"][image_index],
                    dtype=np.float32,
                ).reshape((-1, 4)),
                pred_classes=np.asarray(vessel_predictions["classes"][image_index]).astype(int),
                pred_confidences=np.asarray(vessel_predictions["confidence"][image_index]),
                true_boxes=np.asarray(self._true_boxes[image_index], dtype=np.float32).reshape((-1, 4)),
                true_classes=np.asarray(self._true_classes[image_index]).astype(int),
            )
            for image_index in range(len(self._true_boxes))
        ]

        # Format final test results dict
        result_keys = ["matched_predictions", "unmatched_predictions"]
        value_keys = [
            "true_bboxes",
            "true_classes",
            "predicted_bboxes",
            "predicted_classes",
            "predicted_confidences",
        ]
        test_results = {}
        for result_key in result_keys:
            test_results[result_key] = {}
            for value_key in value_keys:
                values = [result[result_key][value_key] for result in image_results]
                if values:
                    test_results[result_key][value_key] = np.concatenate(values)
                elif "bboxes" in value_key:
                    test_results[result_key][value_key] = np.empty(
                        (0, 4),
                        dtype=np.float32,
                    )
                else:
                    test_results[result_key][value_key] = np.asarray([])

        return test_results

    def _compute_iou(
        self,
        bbox1: np.ndarray,
        bbox2: np.ndarray,
    ) -> float:
        """
        Computes IOU between two bounding boxes.
        """
        xmin1, ymin1, xmax1, ymax1 = bbox1
        xmin2, ymin2, xmax2, ymax2 = bbox2

        # Coordinates of intersecting area
        intersect_xmin = max(xmin1, xmin2)
        intersect_ymin = max(ymin1, ymin2)
        intersect_xmax = min(xmax1, xmax2)
        intersect_ymax = min(ymax1, ymax2)

        # Intersecting area
        intersect_width = max(0.0, intersect_xmax - intersect_xmin)
        intersect_height = max(0.0, intersect_ymax - intersect_ymin)
        intersect_area = intersect_width * intersect_height

        # Calculate intersecting area of bbox
        bbox1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
        bbox2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
        union_area = bbox1_area + bbox2_area - intersect_area

        if union_area == 0:
            return 0.0

        return float(intersect_area / union_area)

    def _get_best_truth_match(
        self,
        pred_box: np.ndarray,
        true_boxes: np.ndarray,
        truth_indexes: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Returns the ground-truth index with highest IOU.
        """
        ious = np.asarray(
            [
                self._compute_iou(
                    pred_box,
                    true_boxes[true_index],
                )
                for true_index in truth_indexes
            ]
        )
        best_index = int(np.argmax(ious))

        return int(truth_indexes[best_index]), float(ious[best_index])

    def _align_prediction_to_groundtruth(
        self,
        pred_boxes: np.ndarray,
        pred_classes: np.ndarray,
        pred_confidences: np.ndarray,
        true_boxes: np.ndarray,
        true_classes: np.ndarray,
        iou_threshold: float = 0.2,
    ) -> Dict:
        """
        Aligns predictions to ground truth using bbox IOUs.
        """
        unmatched_true_indexes = set(range(len(true_boxes)))
        matched_true_indexes: list[int] = []
        matched_prediction_indexes: list[int] = []
        unmatched_prediction_indexes: list[int] = []

        for pred_index in np.argsort(pred_confidences)[::-1]:
            # If image has no gt labels, all predictions are unmatched
            if not unmatched_true_indexes:
                unmatched_prediction_indexes.append(pred_index)
                continue

            # Get gt with the best IOU for prediction
            available_truth_indexes = np.fromiter(
                sorted(unmatched_true_indexes),
                dtype=int,
            )
            best_true_index, best_iou = self._get_best_truth_match(
                pred_box=pred_boxes[pred_index],
                true_boxes=true_boxes,
                truth_indexes=available_truth_indexes,
            )

            # Pred needs to have at least 50% overlap with gt
            if best_iou >= iou_threshold:
                matched_prediction_indexes.append(pred_index)
                matched_true_indexes.append(best_true_index)
                unmatched_true_indexes.remove(best_true_index)
            else:
                unmatched_prediction_indexes.append(pred_index)

        matched_true_indexes = np.asarray(matched_true_indexes, dtype=int)
        matched_prediction_indexes = np.asarray(matched_prediction_indexes, dtype=int)
        unmatched_true_indexes = np.fromiter(sorted(unmatched_true_indexes), dtype=int)
        unmatched_prediction_indexes = np.asarray(unmatched_prediction_indexes, dtype=int)

        return {
            "matched_predictions": {
                "true_bboxes": true_boxes[matched_true_indexes],
                "true_classes": true_classes[matched_true_indexes],
                "predicted_bboxes": pred_boxes[matched_prediction_indexes],
                "predicted_classes": pred_classes[matched_prediction_indexes],
                "predicted_confidences": pred_confidences[matched_prediction_indexes],
            },
            "unmatched_predictions": {
                "true_bboxes": true_boxes[unmatched_true_indexes],
                "true_classes": true_classes[unmatched_true_indexes],
                "predicted_bboxes": pred_boxes[unmatched_prediction_indexes],
                "predicted_classes": pred_classes[unmatched_prediction_indexes],
                "predicted_confidences": pred_confidences[unmatched_prediction_indexes],
            },
        }

    def _calculate_average_precision(
        self,
        confidence_scores: np.ndarray,
        match_labels: np.ndarray,
        num_ground_truths: int,
    ) -> float:
        """
        Calculates confidence-ranked average precision for one IOU threshold.
        """
        if len(confidence_scores) == 0 or num_ground_truths == 0:
            return 0.0

        sorted_indexes = np.argsort(confidence_scores)[::-1]
        sorted_matches = match_labels[sorted_indexes].astype(float)

        cumulative_true_positives = np.cumsum(sorted_matches)
        cumulative_false_positives = np.cumsum(1.0 - sorted_matches)
        recalls = cumulative_true_positives / num_ground_truths
        precisions = cumulative_true_positives / np.maximum(
            cumulative_true_positives + cumulative_false_positives,
            1.0,
        )

        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        for index in range(len(precisions) - 2, -1, -1):
            precisions[index] = max(precisions[index], precisions[index + 1])

        recall_changes = recalls[1:] - recalls[:-1]

        return float(np.sum(recall_changes * precisions[1:]))

    def calculate_class_precision(self, test_results_dict: Dict) -> float:
        """
        Calculates precision score for vessel class prediction.
        """
        matched_results = test_results_dict["matched_predictions"]
        unmatched_results = test_results_dict["unmatched_predictions"]
        num_correct_predictions = np.sum(matched_results["true_classes"] == matched_results["predicted_classes"])
        num_predictions = len(matched_results["predicted_classes"]) + len(unmatched_results["predicted_classes"])

        if num_predictions == 0:
            return 0.0

        return float(num_correct_predictions / num_predictions)

    def calculate_class_recall(self, test_results_dict: Dict) -> float:
        """
        Calculates recall score for vessel class prediction.
        """
        matched_results = test_results_dict["matched_predictions"]
        unmatched_results = test_results_dict["unmatched_predictions"]
        num_correct_predictions = np.sum(matched_results["true_classes"] == matched_results["predicted_classes"])
        num_ground_truths = len(matched_results["true_classes"]) + len(unmatched_results["true_classes"])

        if num_ground_truths == 0:
            return 0.0

        return float(num_correct_predictions / num_ground_truths)

    def calculate_class_false_positive_rate(self, test_results_dict: Dict) -> Dict:
        """
        Calculates false positive rate by vessel class.
        """
        matched_results = test_results_dict["matched_predictions"]
        unmatched_results = test_results_dict["unmatched_predictions"]

        true_classes = matched_results["true_classes"]
        pred_classes = matched_results["predicted_classes"]
        unmatched_pred_classes = unmatched_results["predicted_classes"]
        class_ids = np.union1d(
            np.union1d(true_classes, pred_classes),
            unmatched_pred_classes,
        )

        false_positive_rates = {}
        for class_id in class_ids:
            false_positives = np.sum((pred_classes == class_id) & (true_classes != class_id)) + np.sum(
                unmatched_pred_classes == class_id
            )
            true_negatives = np.sum((pred_classes != class_id) & (true_classes != class_id))
            denominator = false_positives + true_negatives
            if denominator == 0:
                false_positive_rates[int(class_id)] = 0.0
            else:
                false_positive_rates[int(class_id)] = float(false_positives / denominator)

        return false_positive_rates

    def calculate_bbox_map(self, test_results_dict: Dict) -> List[float]:
        """
        Calculates mAP scores for bounding box predictions for IOU thresholds of 0.5-0.95.
        """
        matched_results = test_results_dict["matched_predictions"]
        unmatched_results = test_results_dict["unmatched_predictions"]

        matched_true_boxes = matched_results["true_bboxes"]
        matched_pred_boxes = matched_results["predicted_bboxes"]
        num_ground_truths = len(matched_true_boxes) + len(unmatched_results["true_bboxes"])

        if num_ground_truths == 0:
            return 0.0

        matched_iou_scores = np.asarray(
            [
                self._compute_iou(true_box, pred_box)
                for true_box, pred_box in zip(matched_true_boxes, matched_pred_boxes)
            ]
        )
        confidence_scores = np.concatenate(
            [
                matched_results["predicted_confidences"],
                unmatched_results["predicted_confidences"],
            ]
        )
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        average_precisions = [
            self._calculate_average_precision(
                confidence_scores=confidence_scores,
                match_labels=np.concatenate(
                    [
                        matched_iou_scores >= iou_threshold,
                        np.zeros(
                            len(unmatched_results["predicted_confidences"]),
                            dtype=bool,
                        ),
                    ]
                ),
                num_ground_truths=num_ground_truths,
            )
            for iou_threshold in iou_thresholds
        ]

        return average_precisions

    def calculate_vessel_detector_metrics(self, test_results_dict: Dict) -> pd.DataFrame:
        """
        Calculates precision, recall, FPR and MAP scores by vessel class.
        """
        matched_results = test_results_dict["matched_predictions"]
        unmatched_results = test_results_dict["unmatched_predictions"]
        class_ids = sorted(VESSEL_TYPES_TO_CLASS_ID.values())
        false_positive_rates = self.calculate_class_false_positive_rate(test_results_dict)

        metrics_by_class = {}
        for class_id in class_ids:
            class_id = int(class_id)
            true_class_mask = matched_results["true_classes"] == class_id
            pred_class_mask = matched_results["predicted_classes"] == class_id
            correct_class_mask = true_class_mask & pred_class_mask
            misclassified_true_class_mask = true_class_mask & ~pred_class_mask
            misclassified_pred_class_mask = ~true_class_mask & pred_class_mask

            unmatched_true_class_mask = unmatched_results["true_classes"] == class_id
            unmatched_pred_class_mask = unmatched_results["predicted_classes"] == class_id

            class_results = {
                "matched_predictions": {
                    value_key: values[correct_class_mask] for value_key, values in matched_results.items()
                },
                "unmatched_predictions": {
                    "true_bboxes": np.concatenate(
                        [
                            unmatched_results["true_bboxes"][unmatched_true_class_mask],
                            matched_results["true_bboxes"][misclassified_true_class_mask],
                        ]
                    ),
                    "true_classes": np.concatenate(
                        [
                            unmatched_results["true_classes"][unmatched_true_class_mask],
                            matched_results["true_classes"][misclassified_true_class_mask],
                        ]
                    ),
                    "predicted_bboxes": np.concatenate(
                        [
                            unmatched_results["predicted_bboxes"][unmatched_pred_class_mask],
                            matched_results["predicted_bboxes"][misclassified_pred_class_mask],
                        ]
                    ),
                    "predicted_classes": np.concatenate(
                        [
                            unmatched_results["predicted_classes"][unmatched_pred_class_mask],
                            matched_results["predicted_classes"][misclassified_pred_class_mask],
                        ]
                    ),
                    "predicted_confidences": np.concatenate(
                        [
                            unmatched_results["predicted_confidences"][unmatched_pred_class_mask],
                            matched_results["predicted_confidences"][misclassified_pred_class_mask],
                        ]
                    ),
                },
            }

            bbox_average_precisions = self.calculate_bbox_map(class_results)
            metrics_by_class[class_id] = {
                "precision": self.calculate_class_precision(class_results),
                "recall": self.calculate_class_recall(class_results),
                "false_positive_rate": false_positive_rates.get(class_id, 0.0),
                "bbox_mAP": float(np.mean(bbox_average_precisions)),
            }

        return pd.DataFrame(metrics_by_class)

    def get_classification_matrix_by_class(self, test_results_dict: Dict) -> Dict:
        """
        Returns classification matrix for each class ID as a dictionary.
        """
        matched_results = test_results_dict["matched_predictions"]
        unmatched_results = test_results_dict["unmatched_predictions"]

        true_classes = matched_results["true_classes"]
        pred_classes = matched_results["predicted_classes"]
        unmatched_true_classes = unmatched_results["true_classes"]
        unmatched_pred_classes = unmatched_results["predicted_classes"]

        classification_matrix_by_class = {}
        for class_id in sorted(VESSEL_TYPES_TO_CLASS_ID.values()):
            class_id = int(class_id)
            true_class_mask = true_classes == class_id
            pred_class_mask = pred_classes == class_id

            true_positives = np.sum(true_class_mask & pred_class_mask)
            false_positives = np.sum(~true_class_mask & pred_class_mask) + np.sum(unmatched_pred_classes == class_id)
            false_negatives = np.sum(true_class_mask & ~pred_class_mask) + np.sum(unmatched_true_classes == class_id)

            classification_matrix_by_class[class_id] = {
                "TP": int(true_positives),
                "FP": int(false_positives),
                "FN": int(false_negatives),
            }

        return classification_matrix_by_class
